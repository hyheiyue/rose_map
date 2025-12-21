#pragma once
#include "acc_map.hpp"

#include <cmath>
#include <limits>
#include <queue>

namespace rose_map {

class ESDF: public AccMap {
public:
    using Ptr = std::shared_ptr<ESDF>;

    explicit ESDF(const YAML::Node& config): AccMap(config) {
        esdf_.resize(nx_ * ny_, kInf);
    }

    static Ptr create(const YAML::Node& config) {
        return std::make_shared<ESDF>(config);
    }

    void update(Clock now) {
        AccMap::update(now);

        // std::fill(esdf_.begin(), esdf_.end(), kInf);
        // computeESDF();
    }

    float getDistance(const VoxelKey2D& key) const {
        int idx = key2DToIndex2D(key);
        if (idx < 0)
            return kInf;
        return esdf_[idx];
    }

    std::vector<Eigen::Vector4f> getOccupiedPoints() const {
        std::vector<Eigen::Vector4f> pts;
        pts.reserve(nx_ * ny_);

        auto robo_base = getRoboBase();

        for (int i = 0; i < static_cast<int>(esdf_.size()); ++i) {
            VoxelKey2D key2d = index2DToKey2D(i);
            Eigen::Vector3f p = key2DToWorld(key2d);
            pts.emplace_back(p.x(), p.y(), p.z(), esdf_[i]);
        }

        pts.emplace_back(robo_base.x(), robo_base.y(), robo_base.z(), 0.0f);
        return pts;
    }

private:
    static constexpr float kInf = 1e6f;

    std::vector<float> esdf_;

    // 使用索引而不是 Key 以减少重复转换
    struct Node {
        int idx;
        float dist;
        bool operator<(const Node& other) const {
            return dist > other.dist; // min-heap
        }
    };

    void computeESDF() {
        const float res = voxel_size_;
        const float diag = res * 1.41421356f;

        std::priority_queue<Node> pq;

        const int size2d = nx_ * ny_;
        const auto& acc = acc_grid_view();

        for (int idx = 0; idx < size2d; ++idx) {
            if (acc[idx] == 0) { // occupied/obstacle
                esdf_[idx] = 0.0f;
                pq.push({ idx, 0.0f });
            } else {
                esdf_[idx] = kInf;
            }
        }

        // 8 连通邻居偏移（基于 key 坐标）
        const int dx[8] = { 1, -1, 0, 0, 1, 1, -1, -1 };
        const int dy[8] = { 0, 0, 1, -1, 1, -1, 1, -1 };
        const float cost[8] = { res, res, res, res, diag, diag, diag, diag };

        while (!pq.empty()) {
            Node cur = pq.top();
            pq.pop();

            int cur_idx = cur.idx;
            float cur_dist = cur.dist;

            if (cur_dist > esdf_[cur_idx])
                continue;

            VoxelKey2D cur_key = index2DToKey2D(cur_idx);

            for (int k = 0; k < 8; ++k) {
                VoxelKey2D nb_key(cur_key.x + dx[k], cur_key.y + dy[k]);
                int nb_idx = key2DToIndex2D(nb_key);
                if (nb_idx < 0)
                    continue;

                float nd = cur_dist + cost[k];
                if (nd < esdf_[nb_idx]) {
                    esdf_[nb_idx] = nd;
                    pq.push({ nb_idx, nd });
                }
            }
        }
    }
};

} // namespace rose_map
