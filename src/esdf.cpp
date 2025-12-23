#include "esdf.hpp"
#include <queue>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_invoke.h>

namespace rose_map {
ESDF::ESDF(rclcpp::Node& node): AccMap(node) {
    const int size = nx_ * ny_;
    esdf_.resize(size);
    dist_to_occ_.resize(size);
    dist_to_free_.resize(size);

    const float vs = voxel_size_;
    step_cost_[0] = vs; // straight
    step_cost_[1] = vs * std::sqrt(2.0f); // diagonal
}
void ESDF::update(Clock now) {
    AccMap::update(now);
    rebuildSigned();
}
std::vector<Eigen::Vector4f> ESDF::getOccupiedPoints(float sample_resolution_m) const {
    std::vector<Eigen::Vector4f> pts;

    if (sample_resolution_m <= 0.0f)
        sample_resolution_m = voxel_size_;

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
void ESDF::runKeyDijkstra(
    const std::vector<uint8_t>& acc,
    std::vector<float>& dist,
    bool source_is_occ
) {
    const int size = nx_ * ny_;

    // local pointers for hot arrays
    float* dist_ptr = dist.data();
    const uint8_t* acc_ptr = acc.data();

    // --- parallel initialize dist ---
    tbb::parallel_for(tbb::blocked_range<int>(0, size), [&](const tbb::blocked_range<int>& r) {
        for (int i = r.begin(); i != r.end(); ++i) {
            bool is_occ = (acc_ptr[i] == 0);
            dist_ptr[i] = (is_occ == source_is_occ) ? 0.0f : kInf;
        }
    });

    // Reserve underlying container for PQ to reduce allocations
    std::vector<QNode> heap_data;
    heap_data.reserve(std::max(1024, size / 16)); // heuristic reserve
    std::priority_queue<QNode, std::vector<QNode>, QComp> pq(QComp(), std::move(heap_data));

    // push initial sources (use cached idx and key)
    for (int i = 0; i < size; ++i) {
        if (dist_ptr[i] == 0.0f) {
            VoxelKey2D k = index2DToKey2D(i);
            pq.push(QNode { k, i, 0.0f });
        }
    }

    // Hot loop: Dijkstra expanding by VoxelKey only (idx cached)
    while (!pq.empty()) {
        QNode cur = pq.top();
        pq.pop();

        // fast index check (use cached idx)
        int cur_idx = cur.idx;
        if (cur_idx < 0)
            continue;

        // stale entry check
        if (cur.dist > dist_ptr[cur_idx])
            continue;

        // expand neighbors
        // when diagonals disabled, only iterate first 4 entries
        const int Kmax = params_.esdf_params.use_diagonals ? 8 : 4;

        for (int kk = 0; kk < Kmax; ++kk) {
            VoxelKey2D nb_key { cur.key.x + dx8_[kk], cur.key.y + dy8_[kk] };
            int nb_idx = key2DToIndex2D(nb_key);
            if (nb_idx < 0)
                continue;

            float step = step_cost_[isDiagonalIdx(kk) ? 1 : 0];
            float nd = cur.dist + step;

            // relax
            if (nd < dist_ptr[nb_idx]) {
                dist_ptr[nb_idx] = nd;
                pq.push(QNode { nb_key, nb_idx, nd });
            }
        }
    }
}
void ESDF::rebuildSigned() {
    const auto& acc = acc_grid_view();
    // run two independent Dijkstras in parallel (each uses own dist buffer)
    tbb::parallel_invoke(
        [&]() { runKeyDijkstra(acc, dist_to_occ_, true); },
        [&]() { runKeyDijkstra(acc, dist_to_free_, false); }
    );

    const int size = nx_ * ny_;
    // combine -> esdf (parallel)
    tbb::parallel_for(tbb::blocked_range<int>(0, size), [&](const tbb::blocked_range<int>& r) {
        for (int i = r.begin(); i != r.end(); ++i) {
            esdf_[i] = dist_to_occ_[i] - dist_to_free_[i];
        }
    });
}

} // namespace rose_map