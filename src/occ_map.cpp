#include "rose_map/occ_map.hpp"
#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_invoke.h>

namespace rose_map {

OccMap::OccMap(rclcpp::Node& node) {
    params_.load(node);
    occ_map_info_.voxel_size = params_.occ_map_params.voxel_size;

    occ_map_info_.size = params_.occ_map_params.size;
    occ_map_info_.origin = params_.occ_map_params.origin;

    Eigen::Vector3f half = occ_map_info_.size * 0.5f;
    occ_map_info_.min_key = worldToKey3D(occ_map_info_.origin - half);
    occ_map_info_.max_key = worldToKey3D(occ_map_info_.origin + half);

    occ_map_info_.nx = occ_map_info_.max_key.x - occ_map_info_.min_key.x + 1;
    occ_map_info_.ny = occ_map_info_.max_key.y - occ_map_info_.min_key.y + 1;
    occ_map_info_.nz = occ_map_info_.max_key.z - occ_map_info_.min_key.z + 1;

    const size_t N = static_cast<size_t>(occ_map_info_.nx) * occ_map_info_.ny * occ_map_info_.nz;
    occ_map_info_.grid.resize(N);

    hit_buf_.resize(N);
    free_buf_.resize(N);
    ray_buf_.resize(N);
    occ_map_info_.origin_key = worldToKey3D(occ_map_info_.origin);
    occ_map_info_.ox = occ_map_info_.oy = occ_map_info_.oz = 0;
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

    const VoxelKey3D sensor_key = worldToKey3D(sensor_origin);

    struct LocalBuf {
        std::vector<int> ray;
        std::vector<VoxelKey3D> outmap_ray;
        std::vector<int> hit;
        bool inited = false;
    };

    tbb::enumerable_thread_specific<LocalBuf> tls;

    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, pts.size(), 64),
        [&](const tbb::blocked_range<size_t>& r) {
            auto& local = tls.local();
            if (!local.inited) {
                local.hit.reserve(512);
                local.ray.reserve(512);
                local.outmap_ray.reserve(256);
                local.inited = true;
            }
            for (size_t i = r.begin(); i < r.end(); ++i) {
                const Eigen::Vector3f& p = pts[i];
                const VoxelKey3D k_hit = worldToKey3D(p);

                int idx = key3DToIndex3D(k_hit);
                if (idx >= 0)
                    local.hit.push_back(idx);

                if (params_.occ_map_params.use_ray) {
                    int idx = key3DToIndex3D(k_hit);
                    if (idx >= 0)
                        local.ray.push_back(idx);
                    else
                        local.outmap_ray.push_back(k_hit);
                }
            }
        }
    );

    size_t total_hit = 0, total_ray = 0, total_outmap = 0;
    for (const auto& local: tls) {
        total_hit += local.hit.size();
        total_ray += local.ray.size();
        total_outmap += local.outmap_ray.size();
    }

    std::vector<VoxelKey3D> outmap_ray;
    outmap_ray.reserve(total_outmap);
    hit_buf_.reserve(total_hit);
    ray_buf_.reserve(total_ray);

    for (auto& local: tls) {
        for (int idx: local.hit)
            hit_buf_.tryPush(idx, stamp_now_);

        if (params_.occ_map_params.use_ray) {
            for (int idx: local.ray)
                ray_buf_.tryPush(idx, stamp_now_);

            outmap_ray.insert(outmap_ray.end(), local.outmap_ray.begin(), local.outmap_ray.end());
        }
    }
    commitHits(t);

    if (params_.occ_map_params.use_ray) {
        const size_t max_range_vox = static_cast<size_t>(
            std::ceil(params_.occ_map_params.max_ray_range / occ_map_info_.voxel_size)
        );
        tbb::enumerable_thread_specific<RayResultSOA> tls_outmap([&] {
            RayResultSOA v;
            v.reserve(256);
            return v;
        });

        tbb::enumerable_thread_specific<RayResultSOA> tls_map([&] {
            RayResultSOA v;
            v.reserve(256);
            return v;
        });

        tbb::parallel_invoke(
            [&] { raycastOutmapParallel(sensor_key, outmap_ray, max_range_vox, tls_outmap); },
            [&] { raycastParallel(sensor_key, ray_buf_, max_range_vox, tls_map); }
        );

        for (auto& local: tls_outmap) {
            for (size_t i = 0; i < local.size(); ++i) {
                free_buf_.tryPush(local.free_idx[i], local.count[i], stamp_now_);
            }
        }

        for (auto& local: tls_map) {
            for (size_t i = 0; i < local.size(); ++i) {
                free_buf_.tryPush(local.free_idx[i], local.count[i], stamp_now_);
            }
        }

        ray_buf_.clear();
        commitFree(t);
    }
}
inline void OccMap::raycastOutmapParallel(
    const VoxelKey3D& o,
    const std::vector<VoxelKey3D>& outmap_targets,
    size_t max_range_vox,
    tbb::enumerable_thread_specific<RayResultSOA>& tls_free
) {
    constexpr size_t GRAIN = 256;

    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, outmap_targets.size(), GRAIN),
        [&](const tbb::blocked_range<size_t>& r) {
            auto& out = tls_free.local();
            for (size_t i = r.begin(); i < r.end(); ++i) {
                const auto& h = outmap_targets[i];

                float dx = (h.x + 0.5f) - (o.x + 0.5f);
                float dy = (h.y + 0.5f) - (o.y + 0.5f);
                float dz = (h.z + 0.5f) - (o.z + 0.5f);

                const float invLen = 1.f / std::max({ std::abs(dx), std::abs(dy), std::abs(dz) });
                dx *= invLen;
                dy *= invLen;
                dz *= invLen;

                DDAOutmapPolicy policy;
                ddaRaycastKernel(o, dx, dy, dz, max_range_vox, policy, out);
            }
        }
    );
}

inline void OccMap::raycastParallel(
    const VoxelKey3D& o,
    const StampedIndexBuffer& ray,
    size_t max_range_vox,
    tbb::enumerable_thread_specific<RayResultSOA>& tls_free
) {
    constexpr size_t GRAIN = 256;

    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, ray.indices.size(), GRAIN),
        [&](const tbb::blocked_range<size_t>& r) {
            auto& out = tls_free.local();

            for (size_t i = r.begin(); i < r.end(); ++i) {
                const auto& k = index3DToKey3D(ray.indices[i]);

                float dx = (k.x + 0.5f) - (o.x + 0.5f);
                float dy = (k.y + 0.5f) - (o.y + 0.5f);
                float dz = (k.z + 0.5f) - (o.z + 0.5f);

                const float invLen = 1.f / std::max({ std::abs(dx), std::abs(dy), std::abs(dz) });
                dx *= invLen;
                dy *= invLen;
                dz *= invLen;

                DDARayPolicy policy { k.x, k.y, k.z, ray.count[ray.indices[i]] };

                ddaRaycastKernel(o, dx, dy, dz, max_range_vox, policy, out);
            }
        }
    );
}

void OccMap::update(Clock now) {
    now_ = now;

    const int N = occ_map_info_.grid.size();

    occupied_buffer_idx_.clear();
    occupied_buffer_idx_.reserve(N / 8);

    tbb::enumerable_thread_specific<std::vector<int>> tls([&] {
        std::vector<int> v;
        v.reserve(256);
        return v;
    });
    tbb::parallel_for(tbb::blocked_range<int>(0, N, 512), [&](const tbb::blocked_range<int>& r) {
        auto& local = tls.local();
        for (int idx = r.begin(); idx != r.end(); ++idx) {
            if (isOccupied(idx, now)) {
                local.push_back(idx);
            }
        }
    });
    for (auto& local: tls) {
        occupied_buffer_idx_.insert(occupied_buffer_idx_.end(), local.begin(), local.end());
    }
}

std::vector<Eigen::Vector4f> OccMap::getOccupiedPoints(float sample_resolution_m) const {
    std::vector<Eigen::Vector4f> pts;

    if (sample_resolution_m <= 0.0f)
        sample_resolution_m = occ_map_info_.voxel_size;

    int stride =
        std::max(1, static_cast<int>(std::round(sample_resolution_m / occ_map_info_.voxel_size)));

    pts.reserve(occupied_buffer_idx_.size() / stride + 1);

    for (size_t i = 0; i < occupied_buffer_idx_.size(); i += stride) {
        int idx = occupied_buffer_idx_[i];
        if (!isOccupied(idx, now_))
            continue;
        auto p = key3DToWorld(index3DToKey3D(idx));
        pts.emplace_back(p.x(), p.y(), p.z(), p.z() - occ_map_info_.origin.z());
    }

    return pts;
}

void OccMap::setOrigin(const Eigen::Vector3f& o) {
    VoxelKey3D new_origin = worldToKey3D(o);
    VoxelKey3D shift { new_origin.x - occ_map_info_.origin_key.x,
                       new_origin.y - occ_map_info_.origin_key.y,
                       new_origin.z - occ_map_info_.origin_key.z };

    const int min_shift = params_.occ_map_params.min_shift;
    if (std::abs(shift.x) < min_shift && std::abs(shift.y) < min_shift
        && std::abs(shift.z) < min_shift)
        return;

    if (std::abs(shift.x) >= occ_map_info_.nx || std::abs(shift.y) >= occ_map_info_.ny
        || std::abs(shift.z) >= occ_map_info_.nz)
    {
        resetAll();
        occ_map_info_.origin_key = new_origin;
        occ_map_info_.origin = o;
        return;
    }

    slideAxis(0, shift.x);
    slideAxis(1, shift.y);
    slideAxis(2, shift.z);
    occ_map_info_.origin_key = new_origin;
    occ_map_info_.origin = o;
    Eigen::Vector3f half = occ_map_info_.size * 0.5f;
    occ_map_info_.min_key = worldToKey3D(occ_map_info_.origin- half);
    occ_map_info_.max_key = worldToKey3D(occ_map_info_.origin + half);
}

void OccMap::slideAxis(int axis, int shift) {
    if (shift == 0)
        return;
    int dim = (axis == 0 ? occ_map_info_.nx : axis == 1 ? occ_map_info_.ny : occ_map_info_.nz);
    int start = (axis == 0 ? occ_map_info_.ox : axis == 1 ? occ_map_info_.oy : occ_map_info_.oz);

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
        occ_map_info_.ox = (occ_map_info_.ox + shift + occ_map_info_.nx) % occ_map_info_.nx;
    else if (axis == 1)
        occ_map_info_.oy = (occ_map_info_.oy + shift + occ_map_info_.ny) % occ_map_info_.ny;
    else if (axis == 2)
        occ_map_info_.oz = (occ_map_info_.oz + shift + occ_map_info_.nz) % occ_map_info_.nz;
}

void OccMap::clearSlice(int axis, int slice) {
    if (axis == 0) {
        for (int y = 0; y < occ_map_info_.ny; ++y)
            for (int z = 0; z < occ_map_info_.nz; ++z) {
                int idx = (slice * occ_map_info_.ny + y) * occ_map_info_.nz + z;
                occ_map_info_.grid[idx].reset();
                hit_buf_.clearOne(idx);
                free_buf_.clearOne(idx);
                ray_buf_.clearOne(idx);
            }
    } else if (axis == 1) {
        for (int x = 0; x < occ_map_info_.nx; ++x)
            for (int z = 0; z < occ_map_info_.nz; ++z) {
                int idx = (x * occ_map_info_.ny + slice) * occ_map_info_.nz + z;
                occ_map_info_.grid[idx].reset();
                hit_buf_.clearOne(idx);
                free_buf_.clearOne(idx);
                ray_buf_.clearOne(idx);
            }
    } else if (axis == 2) {
        for (int x = 0; x < occ_map_info_.nx; ++x)
            for (int y = 0; y < occ_map_info_.ny; ++y) {
                int idx = (x * occ_map_info_.ny + y) * occ_map_info_.nz + slice;
                occ_map_info_.grid[idx].reset();
                hit_buf_.clearOne(idx);
                free_buf_.clearOne(idx);
                ray_buf_.clearOne(idx);
            }
    }
}

void OccMap::resetAll() {
    for (auto& c: occ_map_info_.grid)
        c.reset();
    hit_buf_.reset();
    free_buf_.reset();
    ray_buf_.reset();
}
struct OccupancyDelta {
    int idx;
    bool was;
    bool now;
};

void OccMap::commitHits(Clock t) {
    const auto& indices = hit_buf_.indices;
    if (indices.empty()) {
        hit_buf_.clear();
        return;
    }
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, indices.size(), 128),
        [&](const tbb::blocked_range<size_t>& r) {
            for (size_t i = r.begin(); i != r.end(); ++i) {
                const int idx = indices[i];

                auto& count = hit_buf_.count[idx];
                if (count == 0)
                    continue;

                Cell& c = occ_map_info_.grid[idx];

                c.log_odds = std::min(
                    c.log_odds + params_.occ_map_params.log_hit * count,
                    params_.occ_map_params.log_max
                );
                c.last_update = t;

                count = 0;
            }
        }
    );

    hit_buf_.clear();
}

void OccMap::commitFree(Clock t) {
    const auto& indices = free_buf_.indices;
    if (indices.empty()) {
        free_buf_.clear();
        return;
    }

    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, indices.size(), 128),
        [&](const tbb::blocked_range<size_t>& r) {
            for (size_t i = r.begin(); i != r.end(); ++i) {
                const int idx = indices[i];

                auto& count = free_buf_.count[idx];
                if (count == 0)
                    continue;

                Cell& c = occ_map_info_.grid[idx];

                c.log_odds = std::max(
                    c.log_odds + params_.occ_map_params.log_free * count,
                    params_.occ_map_params.log_min
                );
                c.last_update = t;

                count = 0;
            }
        }
    );

    free_buf_.clear();
}

} // namespace rose_map
