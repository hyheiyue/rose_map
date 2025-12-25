#include "rose_map/esdf.hpp"
#include <queue>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_invoke.h>

namespace rose_map {
ESDF::ESDF(rclcpp::Node& node): AccMap(node) {
    const int size = occ_map_info_.nx_ * occ_map_info_.ny_;
    esdf_.resize(size);
    dist_to_occ_.resize(size);
    dist_to_free_.resize(size);

    const float vs = occ_map_info_.voxel_size_;
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
        sample_resolution_m = occ_map_info_.voxel_size_;

    int stride = std::max(1, static_cast<int>(std::round(sample_resolution_m / occ_map_info_.voxel_size_)));
    pts.reserve((occ_map_info_.nx_ / stride) * (occ_map_info_.ny_ / stride));

    auto robo = getRoboBase();

    for (int y = 0; y < occ_map_info_.ny_; y += stride) {
        for (int x = 0; x < occ_map_info_.nx_; x += stride) {
            int idx = y * occ_map_info_.nx_ + x;
            VoxelKey2D k = index2DToKey2D(idx);
            Eigen::Vector3f p = key2DToWorld(k);
            pts.emplace_back(p.x(), p.y(), p.z(), esdf_[idx]);
        }
    }

    pts.emplace_back(robo.x(), robo.y(), robo.z(), 0.0f);
    return pts;
}

void ESDF::propagateKeyDistanceFieldTwoPass(
    const std::vector<uint8_t>& acc,
    std::vector<float>& dist,
    bool source_is_occ
) {
    const int size = occ_map_info_.nx_ *occ_map_info_. ny_;
    float* dist_ptr = dist.data();
    const uint8_t* acc_ptr = acc.data();

    tbb::parallel_for(tbb::blocked_range<int>(0, size), [&](const tbb::blocked_range<int>& r) {
        for (int i = r.begin(); i != r.end(); ++i) {
            bool is_occ = (acc_ptr[i] == 1);
            dist_ptr[i] = (is_occ == source_is_occ) ? 0.0f : kInf;
        }
    });

    bool has_source = false;
    for (int i = 0; i < size; ++i) {
        if (dist_ptr[i] == 0.0f) {
            has_source = true;
            break;
        }
    }
    if (!has_source) {
        std::fill(dist.begin(), dist.end(), kInf);
        return;
    }

    const int iterations = 2;
    for (int it = 0; it < iterations; ++it) {
        // Forward pass（左上 → 右下）
        for (int y = 0; y < occ_map_info_.ny_; ++y) {
            for (int x = 0; x < occ_map_info_.nx_; ++x) {
                int idx = y * occ_map_info_.nx_ + x;
                float cur = dist_ptr[idx];
                if (!std::isfinite(cur))
                    continue;

                VoxelKey2D key = index2DToKey2D(idx);
                float best = cur;

                const int Kmax = params_.esdf_params.use_diagonals ? 8 : 4;
                for (int k = 0; k < Kmax; ++k) {
                    if (dx8_[k] > 0 || dy8_[k] > 0)
                        continue; // 仅用上/左方向邻居传播

                    VoxelKey2D nb { key.x + dx8_[k], key.y + dy8_[k] };
                    int nb_idx = key2DToIndex2D(nb);
                    if (nb_idx < 0)
                        continue;

                    float nd = dist_ptr[nb_idx];
                    if (!std::isfinite(nd))
                        continue;

                    float step = step_cost_[isDiagonalIdx(k) ? 1 : 0];
                    float cand = nd + step;
                    if (cand < best)
                        best = cand;
                }
                dist_ptr[idx] = best;
            }
        }

        // Backward pass（右下 → 左上）
        for (int y = occ_map_info_.ny_ - 1; y >= 0; --y) {
            for (int x = occ_map_info_.nx_ - 1; x >= 0; --x) {
                int idx = y * occ_map_info_.nx_ + x;
                float cur = dist_ptr[idx];
                if (!std::isfinite(cur))
                    continue; // 跳过 inf/NaN，不参与扩散

                VoxelKey2D key = index2DToKey2D(idx);
                float best = cur;

                const int Kmax = params_.esdf_params.use_diagonals ? 8 : 4;
                for (int k = 0; k < Kmax; ++k) {
                    if (dx8_[k] < 0 || dy8_[k] < 0)
                        continue; // 仅用下/右方向邻居传播

                    VoxelKey2D nb { key.x + dx8_[k], key.y + dy8_[k] };
                    int nb_idx = key2DToIndex2D(nb);
                    if (nb_idx < 0)
                        continue;

                    float nd = dist_ptr[nb_idx];
                    if (!std::isfinite(nd))
                        continue; // 邻居是 inf 也跳过，避免 inf-污染

                    float step = step_cost_[isDiagonalIdx(k) ? 1 : 0];
                    float cand = nd + step;
                    if (cand < best)
                        best = cand;
                }
                dist_ptr[idx] = best;
            }
        }
    }
}

void ESDF::rebuildSigned() {
    const auto& acc = acc_grid_view();
    if (acc.empty()) {
        std::cout << "acc map empty!!!" << std::endl;
    }
    // run two independent Dijkstras in parallel (each uses own dist buffer)
    tbb::parallel_invoke(
        [&]() { propagateKeyDistanceFieldTwoPass(acc, dist_to_occ_, true); },
        [&]() { propagateKeyDistanceFieldTwoPass(acc, dist_to_free_, false); }
    );

    const int size = occ_map_info_.nx_ * occ_map_info_.ny_;
    // combine -> esdf (parallel)
    tbb::parallel_for(tbb::blocked_range<int>(0, size), [&](const tbb::blocked_range<int>& r) {
        for (int i = r.begin(); i != r.end(); ++i) {
            esdf_[i] = dist_to_occ_[i] - dist_to_free_[i];
        }
    });
}

} // namespace rose_map