#include "occ_map.hpp"
#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_invoke.h>
namespace rose_map {
OccMap::OccMap(rclcpp::Node& node) {
    voxel_size_ = node.declare_parameter<float>("rose_map.voxel_size", 0.05);
    std::vector<double> size_vec = node.declare_parameter<std::vector<double>>(
        "rose_map.size",
        std::vector<double> { 5.0, 5.0, 5.0 }
    );
    size_ = Eigen::Vector3f(size_vec[0], size_vec[1], size_vec[2]);
    std::vector<double> origin_vec = node.declare_parameter<std::vector<double>>(
        "rose_map.origin",
        std::vector<double> { 5.0, 5.0, 5.0 }
    );
    origin_ = Eigen::Vector3f(origin_vec[0], origin_vec[1], origin_vec[2]);
    params_.load(node);

    Eigen::Vector3f half = size_ * 0.5f;
    min_key_ = worldToKey3D(origin_ - half);
    max_key_ = worldToKey3D(origin_ + half);

    nx_ = max_key_.x - min_key_.x + 1;
    ny_ = max_key_.y - min_key_.y + 1;
    nz_ = max_key_.z - min_key_.z + 1;

    const size_t N = size_t(nx_) * ny_ * nz_;
    grid_.resize(N);

    hit_buf_.stamp.resize(N, 0);
    free_buf_.stamp.resize(N, 0);
    ray_buf_.stamp.resize(N, 0);
    occupied_pos_.assign(N, -1);
    origin_key_ = worldToKey3D(origin_);
    ox_ = oy_ = oz_ = 0;
    occupied_buffer_idx_.reserve(N);
}
void OccMap::insertPointCloud(
    const std::vector<Eigen::Vector3f>& pts,
    const Eigen::Vector3f& sensor_origin,
    Clock t
) {
    ++stamp_now_;

    hit_buf_.clear();
    free_buf_.clear();
    ray_buf_.clear();

    const float max_r2 =
        params_.occ_map_params.max_ray_range * params_.occ_map_params.max_ray_range;
    const VoxelKey3D sensor_key = worldToKey3D(sensor_origin);
    struct LocalBuf {
        std::vector<int> ray;
        std::vector<int> hit;
    };

    tbb::enumerable_thread_specific<LocalBuf> tls;

    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, pts.size()),
        [&](const tbb::blocked_range<size_t>& r) {
            auto& local = tls.local();

            for (size_t i = r.begin(); i < r.end(); ++i) {
                const auto& p = pts[i];
                if ((p - sensor_origin).squaredNorm() > max_r2)
                    continue;

                VoxelKey3D k_hit = worldToKey3D(p);
                int hit_idx = key3DToIndex3D(k_hit);
                if (hit_idx < 0)
                    continue;

                for (auto& n: HIT_D) {
                    int idx = key3DToIndex3D({ k_hit.x + n[0], k_hit.y + n[1], k_hit.z + n[2] });
                    if (idx >= 0) {
                        local.hit.push_back(idx);
                    }
                }
                if (params_.occ_map_params.use_ray) {
                    for (auto n: RAY_D) {
                        int idx =
                            key3DToIndex3D({ k_hit.x + n[0], k_hit.y + n[1], k_hit.z + n[2] });
                        if (idx >= 0) {
                            local.ray.push_back(idx);
                        }
                    }
                }
            }
        }
    );

    for (auto& local: tls) {
        if (params_.occ_map_params.use_ray) {
            for (int idx: local.ray)
                ray_buf_.tryPush(idx, stamp_now_);
        }
        for (int idx: local.hit)
            hit_buf_.tryPush(idx, stamp_now_);
    }
    if (params_.occ_map_params.use_ray) {
        // for(int idx: ray_buf_.indices)
        // {
        //     raycastFreeKey(sensor_key, index3DToKey3D(idx));
        // }
        tbb::enumerable_thread_specific<std::vector<int>> tls_free;

        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, ray_buf_.indices.size()),
            [&](const tbb::blocked_range<size_t>& r) {
                auto& local = tls_free.local();
                for (size_t i = r.begin(); i < r.end(); ++i) {
                    int hit_idx = ray_buf_.indices[i];
                    raycastFreeKeyTLS(sensor_key, index3DToKey3D(hit_idx), local);
                }
            }
        );
        size_t total = 0;
        for (auto& v: tls_free)
            total += v.size();

        free_buf_.indices.reserve(free_buf_.indices.size() + total);
        for (auto& v: tls_free) {
            for (int idx: v) {
                free_buf_.tryPush(idx, stamp_now_);
            }
        }
    }

    tbb::parallel_invoke([&] { commitFree(t); }, [&] { commitHits(t); });
}
void OccMap::update(Clock now) {
    now_ = now;

    size_t write = 0;
    for (size_t read = 0; read < occupied_buffer_idx_.size(); ++read) {
        int idx = occupied_buffer_idx_[read];
        if (isOccupied(idx, now)) {
            occupied_buffer_idx_[write++] = idx;
        } else {
            occupied_pos_[idx] = -1;
        }
    }
    occupied_buffer_idx_.resize(write);
}
std::vector<Eigen::Vector4f> OccMap::getOccupiedPoints(float sample_resolution_m) const {
    std::vector<Eigen::Vector4f> pts;

    if (sample_resolution_m <= 0.0f)
        sample_resolution_m = voxel_size_;

    // 物理尺度 → stride
    int stride = std::max(1, static_cast<int>(std::round(sample_resolution_m / voxel_size_)));

    pts.reserve(occupied_buffer_idx_.size() / stride + 1);

    for (size_t i = 0; i < occupied_buffer_idx_.size(); i += stride) {
        int idx = occupied_buffer_idx_[i];

        if (!isOccupied(idx, now_))
            continue;

        auto p = key3DToWorld(index3DToKey3D(idx));
        pts.emplace_back(p.x(), p.y(), p.z(), p.z() - origin_.z());
    }

    return pts;
}
void OccMap::setOrigin(const Eigen::Vector3f& o) {
    VoxelKey3D new_origin = worldToKey3D(o);
    VoxelKey3D shift { new_origin.x - origin_key_.x,
                       new_origin.y - origin_key_.y,
                       new_origin.z - origin_key_.z };
    const int min_shift = params_.occ_map_params.min_shift;
    if (std::abs(shift.x) < min_shift && std::abs(shift.y) < min_shift
        && std::abs(shift.z) < min_shift)
        return;
    if (std::abs(shift.x) >= nx_ || std::abs(shift.y) >= ny_ || std::abs(shift.z) >= nz_) {
        resetAll();
        origin_key_ = new_origin;
        origin_ = o;
        return;
    }
    slideAxis(0, shift.x);
    slideAxis(1, shift.y);
    slideAxis(2, shift.z);
    origin_key_ = new_origin;
    origin_ = o;
}
void OccMap::slideAxis(int axis, int shift) {
    if (shift == 0)
        return;
    int steps = std::abs(shift);
    int dir = shift > 0 ? 1 : -1;
    for (int s = 0; s < steps; ++s) {
        int slice = (axis == 0 ? ox_ : axis == 1 ? oy_ : oz_);
        clearSlice(axis, slice);
        if (axis == 0)
            ox_ = (ox_ + dir + nx_) % nx_;
        if (axis == 1)
            oy_ = (oy_ + dir + ny_) % ny_;
        if (axis == 2)
            oz_ = (oz_ + dir + nz_) % nz_;
    }
}
void OccMap::clearSlice(int axis, int slice) {
    for (int i = 0; i < ny_; ++i)
        for (int j = 0; j < nz_; ++j) {
            int idx;
            if (axis == 0)
                idx = (slice * ny_ + i) * nz_ + j;
            else if (axis == 1)
                idx = (i * ny_ + slice) * nz_ + j;
            else
                idx = (i * ny_ + j) * nz_ + slice;
            grid_[idx] = Cell();
            hit_buf_.stamp[idx] = free_buf_.stamp[idx] = ray_buf_.stamp[idx] = 0;
        }
}
void OccMap::resetAll() {
    std::fill(grid_.begin(), grid_.end(), Cell());
    std::fill(hit_buf_.stamp.begin(), hit_buf_.stamp.end(), 0);
    std::fill(free_buf_.stamp.begin(), free_buf_.stamp.end(), 0);
    std::fill(ray_buf_.stamp.begin(), ray_buf_.stamp.end(), 0);
}
void OccMap::commitHits(Clock t) {
    for (int idx: hit_buf_.indices) {
        Cell& c = grid_[idx];
        bool was = isOccupied(idx, t);

        c.log_odds =
            std::min(c.log_odds + params_.occ_map_params.log_hit, params_.occ_map_params.log_max);
        c.last_update = t;

        bool now = isOccupied(idx, t);
        trackState(was, now, idx);
    }
    hit_buf_.clear();
}
void OccMap::commitFree(Clock t) {
    for (int idx: free_buf_.indices) {
        Cell& c = grid_[idx];
        bool was = isOccupied(idx, t);

        c.log_odds =
            std::max(c.log_odds + params_.occ_map_params.log_free, params_.occ_map_params.log_min);
        c.last_update = t;

        bool now = isOccupied(idx, t);
        trackState(was, now, idx);
    }
    free_buf_.clear();
}
} // namespace rose_map