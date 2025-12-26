#include "rose_map/occ_map.hpp"
#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

namespace rose_map {

OccMap::OccMap(rclcpp::Node& node) {
    params_.load(node);
    occ_map_info_.voxel_size_ = params_.occ_map_params.voxel_size;

    occ_map_info_.size_ = params_.occ_map_params.size;
    occ_map_info_.origin_ = params_.occ_map_params.origin;

    Eigen::Vector3f half = occ_map_info_.size_ * 0.5f;
    occ_map_info_.min_key_ = worldToKey3D(occ_map_info_.origin_ - half);
    occ_map_info_.max_key_ = worldToKey3D(occ_map_info_.origin_ + half);

    occ_map_info_.nx_ = occ_map_info_.max_key_.x - occ_map_info_.min_key_.x + 1;
    occ_map_info_.ny_ = occ_map_info_.max_key_.y - occ_map_info_.min_key_.y + 1;
    occ_map_info_.nz_ = occ_map_info_.max_key_.z - occ_map_info_.min_key_.z + 1;

    const size_t N = static_cast<size_t>(occ_map_info_.nx_) * occ_map_info_.ny_ * occ_map_info_.nz_;
    occ_map_info_.grid_.resize(N);

    hit_buf_.resize(N);
    free_buf_.resize(N);
    ray_buf_.resize(N);
    rise_buf_.resize(N);
    fall_buf_.resize(N);
    prev_occupied_.assign(N, 0);

    occupied_pos_.assign(N, -1);
    occ_map_info_.origin_key_ = worldToKey3D(occ_map_info_.origin_);
    occ_map_info_.ox_ = occ_map_info_.oy_ = occ_map_info_.oz_ = 0;
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
            local.hit.reserve(256);
            local.ray.reserve(256);

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
        std::vector<VoxelKey3D> h_list;
        std::vector<size_t> step_limit;
        size_t max_range_vox =
            std::ceil(params_.occ_map_params.max_ray_range / occ_map_info_.voxel_size_);

        for (int idx: ray_buf_.indices) {
            auto k = index3DToKey3D(idx);
            h_list.push_back(k);
            step_limit.push_back(
                std::abs(k.x - sensor_key.x) + std::abs(k.y - sensor_key.y)
                + std::abs(k.z - sensor_key.z)
            );
        }
        tbb::enumerable_thread_specific<std::vector<int>> tls_free;
        raycastFreeKeyTLS_SyncStep_Parallel(
            sensor_key,
            h_list,
            step_limit,
            max_range_vox,
            tls_free
        );
        for (auto& local: tls_free) {
            for (int idx: local) {
                free_buf_.tryPush(idx, stamp_now_);
            }
        }

        ray_buf_.clear();
    }
    commitFree(t);
    commitHits(t);
}

void OccMap::update(Clock now) {
    now_ = now;
    rise_buf_.clear();
    fall_buf_.clear();

    size_t write = 0;
    for (size_t read = 0; read < occupied_buffer_idx_.size(); ++read) {
        int idx = occupied_buffer_idx_[read];
        bool was = prev_occupied_[idx];
        bool now_occ = isOccupied(idx, now);

        if (!was && now_occ) {
            rise_buf_.tryPush(idx, stamp_now_);
        } else if (was && !now_occ) {
            fall_buf_.tryPush(idx, stamp_now_);
            occ_map_info_.grid_[idx].reset();
        }

        if (now_occ) {
            occupied_buffer_idx_[write++] = idx;
        } else {
            occupied_pos_[idx] = -1;
        }

        prev_occupied_[idx] = now_occ;
    }

    occupied_buffer_idx_.resize(write);
}

std::vector<Eigen::Vector4f> OccMap::getOccupiedPoints(float sample_resolution_m) const {
    std::vector<Eigen::Vector4f> pts;

    if (sample_resolution_m <= 0.0f)
        sample_resolution_m = occ_map_info_.voxel_size_;

    int stride =
        std::max(1, static_cast<int>(std::round(sample_resolution_m / occ_map_info_.voxel_size_)));

    pts.reserve(occupied_buffer_idx_.size() / stride + 1);

    for (size_t i = 0; i < occupied_buffer_idx_.size(); i += stride) {
        int idx = occupied_buffer_idx_[i];
        if (!isOccupied(idx, now_))
            continue;
        auto p = key3DToWorld(index3DToKey3D(idx));
        pts.emplace_back(p.x(), p.y(), p.z(), p.z() - occ_map_info_.origin_.z());
    }

    return pts;
}

void OccMap::setOrigin(const Eigen::Vector3f& o) {
    VoxelKey3D new_origin = worldToKey3D(o);
    VoxelKey3D shift { new_origin.x - occ_map_info_.origin_key_.x,
                       new_origin.y - occ_map_info_.origin_key_.y,
                       new_origin.z - occ_map_info_.origin_key_.z };

    const int min_shift = params_.occ_map_params.min_shift;
    if (std::abs(shift.x) < min_shift && std::abs(shift.y) < min_shift
        && std::abs(shift.z) < min_shift)
        return;

    if (std::abs(shift.x) >= occ_map_info_.nx_ || std::abs(shift.y) >= occ_map_info_.ny_
        || std::abs(shift.z) >= occ_map_info_.nz_)
    {
        resetAll();
        occ_map_info_.origin_key_ = new_origin;
        occ_map_info_.origin_ = o;
        return;
    }

    slideAxis(0, shift.x);
    slideAxis(1, shift.y);
    slideAxis(2, shift.z);
    occ_map_info_.origin_key_ = new_origin;
    occ_map_info_.origin_ = o;
    Eigen::Vector3f half = occ_map_info_.size_ * 0.5f;
    occ_map_info_.min_key_ = worldToKey3D(occ_map_info_.origin_ - half);
    occ_map_info_.max_key_ = worldToKey3D(occ_map_info_.origin_ + half);
}

void OccMap::slideAxis(int axis, int shift) {
    if (shift == 0)
        return;
    int dim = (axis == 0 ? occ_map_info_.nx_ : axis == 1 ? occ_map_info_.ny_ : occ_map_info_.nz_);
    int start = (axis == 0 ? occ_map_info_.ox_ : axis == 1 ? occ_map_info_.oy_ : occ_map_info_.oz_);

    std::vector<int> slices_to_clear;
    slices_to_clear.reserve(std::abs(shift));
    int steps = std::abs(shift);
    int dir = shift > 0 ? 1 : -1;
    for (int s = 0; s < steps; ++s) {
        int slice = (start + dir * s + dim) % dim;
        if (slice < 0)
            slice += dim;
        slices_to_clear.push_back(slice);
    }
    for (int slice: slices_to_clear)
        clearSlice(axis, slice);

    if (axis == 0)
        occ_map_info_.ox_ = (occ_map_info_.ox_ + shift + occ_map_info_.nx_) % occ_map_info_.nx_;
    else if (axis == 1)
        occ_map_info_.oy_ = (occ_map_info_.oy_ + shift + occ_map_info_.ny_) % occ_map_info_.ny_;
    else if (axis == 2)
        occ_map_info_.oz_ = (occ_map_info_.oz_ + shift + occ_map_info_.nz_) % occ_map_info_.nz_;
}

void OccMap::clearSlice(int axis, int slice) {
    if (axis == 0) {
        for (int y = 0; y < occ_map_info_.ny_; ++y)
            for (int z = 0; z < occ_map_info_.nz_; ++z) {
                int idx = (slice * occ_map_info_.ny_ + y) * occ_map_info_.nz_ + z;
                occ_map_info_.grid_[idx].reset();
                hit_buf_.clearOne(idx);
                free_buf_.clearOne(idx);
                ray_buf_.clearOne(idx);
            }
    } else if (axis == 1) {
        for (int x = 0; x < occ_map_info_.nx_; ++x)
            for (int z = 0; z < occ_map_info_.nz_; ++z) {
                int idx = (x * occ_map_info_.ny_ + slice) * occ_map_info_.nz_ + z;
                occ_map_info_.grid_[idx].reset();
                hit_buf_.clearOne(idx);
                free_buf_.clearOne(idx);
                ray_buf_.clearOne(idx);
            }
    } else if (axis == 2) {
        for (int x = 0; x < occ_map_info_.nx_; ++x)
            for (int y = 0; y < occ_map_info_.ny_; ++y) {
                int idx = (x * occ_map_info_.ny_ + y) * occ_map_info_.nz_ + slice;
                occ_map_info_.grid_[idx].reset();
                hit_buf_.clearOne(idx);
                free_buf_.clearOne(idx);
                ray_buf_.clearOne(idx);
            }
    }
}

void OccMap::resetAll() {
    for (auto& c: occ_map_info_.grid_)
        c.reset();
    hit_buf_.reset();
    free_buf_.reset();
    ray_buf_.reset();
}

void OccMap::commitHits(Clock t) {
    for (int idx: hit_buf_.indices) {
        bool was = isOccupied(idx, t);
        auto& count = hit_buf_.count[idx];
        Cell& c = occ_map_info_.grid_[idx];
        c.log_odds = std::min(
            c.log_odds + params_.occ_map_params.log_hit * count,
            params_.occ_map_params.log_max
        );
        c.last_update = t;
        count = 0;
        bool now = isOccupied(idx, t);
        trackOccupied(was, now, idx);
    }
    hit_buf_.clear();
}

void OccMap::commitFree(Clock t) {
    for (int idx: free_buf_.indices) {
        bool was = isOccupied(idx, t);
        auto& count = free_buf_.count[idx];
        Cell& c = occ_map_info_.grid_[idx];
        c.log_odds = std::max(
            c.log_odds + params_.occ_map_params.log_free * count,
            params_.occ_map_params.log_min
        );
        c.last_update = t;
        count = 0;
        bool now = isOccupied(idx, t);
        trackOccupied(was, now, idx);
    }
    free_buf_.clear();
}

} // namespace rose_map
