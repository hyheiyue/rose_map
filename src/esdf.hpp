#pragma once
#include "acc_map.hpp"

#include <cmath>
#include <queue>
#include <vector>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_invoke.h>

namespace rose_map {

class ESDF: public AccMap {
public:
    using Ptr = std::shared_ptr<ESDF>;

    explicit ESDF(const YAML::Node& config): AccMap(config) {
        const int size = nx_ * ny_;
        esdf_.resize(size);
        dist_to_occ_.resize(size);
        dist_to_free_.resize(size);

        const float vs = voxel_size_;
        step_cost_[0] = vs;
        step_cost_[1] = vs * std::sqrt(2.0f);
    }

    static Ptr create(const YAML::Node& config) {
        return std::make_shared<ESDF>(config);
    }

    void update(Clock now) {
        AccMap::update(now);
        fullRebuildSignedTBB();
    }

    float getDistance(const VoxelKey2D& key) const {
        int idx = key2DToIndex2D(key);
        return (idx < 0) ? kInf : esdf_[idx];
    }
    std::vector<Eigen::Vector4f> getOccupiedPoints(float sample_resolution_m = 0.1) const {
        std::vector<Eigen::Vector4f> pts;

        if (sample_resolution_m <= 0.0f)
            sample_resolution_m = voxel_size_;

        // 物理尺度 → stride
        int stride = std::max(1, static_cast<int>(std::round(sample_resolution_m / voxel_size_)));

        pts.reserve((nx_ / stride) * (ny_ / stride));

        auto robo = getRoboBase();

        for (int y = 0; y < ny_; y += stride) {
            for (int x = 0; x < nx_; x += stride) {
                int idx = y * nx_ + x;
                VoxelKey2D k = index2DToKey2D(idx);
                Eigen::Vector3f p = key2DToWorld(k);
                pts.emplace_back(p.x(), p.y(), p.z(), esdf_[idx]);
            }
        }

        pts.emplace_back(robo.x(), robo.y(), robo.z(), 0.0f);
        return pts;
    }

private:
    static constexpr float kInf = 1e6f;

    struct Node {
        int idx;
        VoxelKey2D key;
        float dist;
        bool operator<(const Node& o) const {
            return dist > o.dist; // min-heap
        }
    };

    std::vector<float> esdf_;
    std::vector<float> dist_to_occ_;
    std::vector<float> dist_to_free_;

    static constexpr int dx_[8] = { 1, -1, 0, 0, 1, 1, -1, -1 };
    static constexpr int dy_[8] = { 0, 0, 1, -1, 1, -1, 1, -1 };

    float step_cost_[2];

    inline bool isDiagonal(int k) const {
        return k >= 4;
    }

    // ---------------- Dijkstra（单线程，线程安全） ----------------
    void
    runDijkstra(const std::vector<uint8_t>& acc, std::vector<float>& dist, bool source_is_occ) {
        const int size = nx_ * ny_;
        std::priority_queue<Node> pq;

        // 1. 并行初始化
        tbb::parallel_for(tbb::blocked_range<int>(0, size), [&](const tbb::blocked_range<int>& r) {
            for (int i = r.begin(); i != r.end(); ++i) {
                bool is_occ = (acc[i] == 0);
                dist[i] = (is_occ == source_is_occ) ? 0.0f : kInf;
            }
        });

        // 2. source 入队（串行，数量相对少）
        for (int i = 0; i < size; ++i) {
            if (dist[i] == 0.0f) {
                pq.push({ i, index2DToKey2D(i), 0.0f });
            }
        }

        // 3. Dijkstra 主循环（不可并行）
        while (!pq.empty()) {
            Node cur = pq.top();
            pq.pop();

            if (cur.dist > dist[cur.idx])
                continue;

            const VoxelKey2D& ck = cur.key;

            for (int k = 0; k < 8; ++k) {
                VoxelKey2D nk { ck.x + dx_[k], ck.y + dy_[k] };
                int nb = key2DToIndex2D(nk);
                if (nb < 0)
                    continue;

                float nd = cur.dist + step_cost_[isDiagonal(k)];
                if (nd < dist[nb]) {
                    dist[nb] = nd;
                    pq.push({ nb, nk, nd });
                }
            }
        }
    }

    // ---------------- Signed ESDF 并行构建 ----------------
    void fullRebuildSignedTBB() {
        const auto& acc = acc_grid_view();

        // 1. 两次 Dijkstra 并行执行
        tbb::parallel_invoke(
            [&]() { runDijkstra(acc, dist_to_occ_, true); },
            [&]() { runDijkstra(acc, dist_to_free_, false); }
        );

        // 2. 合成 signed ESDF（并行）
        const int size = nx_ * ny_;
        tbb::parallel_for(tbb::blocked_range<int>(0, size), [&](const tbb::blocked_range<int>& r) {
            for (int i = r.begin(); i != r.end(); ++i) {
                esdf_[i] = dist_to_occ_[i] - dist_to_free_[i];
            }
        });
    }
};

} // namespace rose_map
