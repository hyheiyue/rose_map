#pragma once

#include "common.hpp"
#include "parameters.hpp"

#include <Eigen/Core>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <queue>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <vector>

namespace rose_map {

struct VoxelKey3D {
    int x, y, z;
    VoxelKey3D(): x(0), y(0), z(0) {}
    VoxelKey3D(int x_, int y_, int z_): x(x_), y(y_), z(z_) {}
    bool operator==(const VoxelKey3D& rhs) const {
        return x == rhs.x && y == rhs.y && z == rhs.z;
    }
};

struct StampedIndexBuffer {
    std::vector<int> indices;
    std::vector<uint32_t> stamp;
    std::vector<uint16_t> count;
    inline void tryPush(int idx, uint32_t now) {
        if (stamp[idx] != now) {
            stamp[idx] = now;
            indices.push_back(idx);
        }
        count[idx]++;
    }
    inline void tryPush(int idx, int count_val, uint32_t now) {
        if (stamp[idx] != now) {
            stamp[idx] = now;
            indices.push_back(idx);
        }
        count[idx] += count_val;
    }
    inline void reserve(size_t n) {
        indices.reserve(n);
    }
    inline void resize(size_t n) {
        stamp.resize(n, 0);
        count.resize(n, 0);
    }
    inline void clearOne(int idx) {
        stamp[idx] = 0;
        count[idx] = 0;
    }
    inline void reset() {
        indices.clear();
        std::fill(stamp.begin(), stamp.end(), 0);
        std::fill(count.begin(), count.end(), 0);
    }
    inline void clear() {
        indices.clear();
        std::fill(count.begin(), count.end(), 0);
    }
};
class OccMap {
public:
    using Ptr = std::shared_ptr<OccMap>;

    explicit OccMap(rclcpp::Node& node);

    static Ptr create(rclcpp::Node& node) {
        return std::make_shared<OccMap>(node);
    }

    struct Cell {
        float log_odds = 0.f;
        Clock last_update = 0.0;
        void reset() {
            log_odds = 0.f;
            last_update = 0.0;
        }
    };
    static constexpr int FREE_D[1][3] = { { 0, 0, 0 } };
    void insertPointCloud(
        const std::vector<Eigen::Vector3f>& pts,
        const Eigen::Vector3f& sensor_origin,
        Clock t
    );
    inline bool isOccupied(int idx, Clock now) const {
        if (idx < 0)
            return params_.occ_map_params.unknown_is_occupied;

        const Cell& c = occ_map_info_.grid_[idx];
        if (now - c.last_update > params_.occ_map_params.timeout)
            return false;

        return c.log_odds > params_.occ_map_params.occ_th;
    }

    void update(Clock now);
    Eigen::Vector3f origin() const {
        return occ_map_info_.origin_;
    }
    std::vector<Eigen::Vector4f> getOccupiedPoints(float sample_resolution_m = -0.05) const;

    void setOrigin(const Eigen::Vector3f& o);
    void slideAxis(int axis, int shift);
    void clearSlice(int axis, int slice);
    void resetAll();
    inline void trackOccupied(bool was, bool now, int idx) {
        if (!was && now) {
            occupied_pos_[idx] = occupied_buffer_idx_.size();
            occupied_buffer_idx_.push_back(idx);
        } else if (was && !now) {
            int pos = occupied_pos_[idx];
            if (pos < 0)
                return;
            int last = occupied_buffer_idx_.back();
            occupied_buffer_idx_[pos] = last;
            occupied_pos_[last] = pos;
            occupied_buffer_idx_.pop_back();
            occupied_pos_[idx] = -1;
        }
    }

    void commitHits(Clock t);

    void commitFree(Clock t);
    struct RayResultSOA {
        std::vector<int> free_idx;
        std::vector<int> count;

        inline void reserve(size_t n) {
            free_idx.reserve(n);
            count.reserve(n);
        }

        inline void clear() {
            free_idx.clear();
            count.clear();
        }

        inline void push(int idx, int c) {
            free_idx.push_back(idx);
            count.push_back(c);
        }

        inline size_t size() const {
            return free_idx.size();
        }
    };

    struct DDAOutmapPolicy {
        inline bool shouldStop(int cx, int cy, int cz) const {
            return false; // 只由 map boundary 控制
        }

        inline void emit(int idx, RayResultSOA& out) const {
            out.push(idx, 1);
        }
    };

    struct DDARayPolicy {
        int tx, ty, tz;
        int count;

        inline bool shouldStop(int cx, int cy, int cz) const {
            return (cx == tx && cy == ty && cz == tz);
        }

        inline void emit(int idx, RayResultSOA& out) const {
            out.push(idx, count);
        }
    };
    template<typename Policy>
    inline void ddaRaycastKernel(
        const VoxelKey3D& o,
        float dx,
        float dy,
        float dz,
        size_t max_range_vox,
        const Policy& policy,
        RayResultSOA& out
    ) {
        const int min_x = occ_map_info_.min_key_.x;
        const int min_y = occ_map_info_.min_key_.y;
        const int min_z = occ_map_info_.min_key_.z;
        const int max_x = occ_map_info_.max_key_.x;
        const int max_y = occ_map_info_.max_key_.y;
        const int max_z = occ_map_info_.max_key_.z;

        const float px = o.x + 0.5f;
        const float py = o.y + 0.5f;
        const float pz = o.z + 0.5f;

        int cx = o.x;
        int cy = o.y;
        int cz = o.z;

        const int stepX = (dx > 0.f) - (dx < 0.f);
        const int stepY = (dy > 0.f) - (dy < 0.f);
        const int stepZ = (dz > 0.f) - (dz < 0.f);

        const float tDeltaX =
            (std::abs(dx) > 1e-6f) ? std::abs(1.f / dx) : std::numeric_limits<float>::infinity();
        const float tDeltaY =
            (std::abs(dy) > 1e-6f) ? std::abs(1.f / dy) : std::numeric_limits<float>::infinity();
        const float tDeltaZ =
            (std::abs(dz) > 1e-6f) ? std::abs(1.f / dz) : std::numeric_limits<float>::infinity();

        float tMaxX = (std::abs(dx) > 1e-6f) ? ((stepX > 0 ? cx + 1.f : cx) - px) / dx
                                             : std::numeric_limits<float>::infinity();
        float tMaxY = (std::abs(dy) > 1e-6f) ? ((stepY > 0 ? cy + 1.f : cy) - py) / dy
                                             : std::numeric_limits<float>::infinity();
        float tMaxZ = (std::abs(dz) > 1e-6f) ? ((stepZ > 0 ? cz + 1.f : cz) - pz) / dz
                                             : std::numeric_limits<float>::infinity();

        for (size_t step = 0; step < max_range_vox; ++step) {
            if (tMaxX < tMaxY) {
                if (tMaxX < tMaxZ) {
                    cx += stepX;
                    tMaxX += tDeltaX;
                } else {
                    cz += stepZ;
                    tMaxZ += tDeltaZ;
                }
            } else {
                if (tMaxY < tMaxZ) {
                    cy += stepY;
                    tMaxY += tDeltaY;
                } else {
                    cz += stepZ;
                    tMaxZ += tDeltaZ;
                }
            }

            if (cx < min_x || cx > max_x || cy < min_y || cy > max_y || cz < min_z || cz > max_z)
                break;

            if (policy.shouldStop(cx, cy, cz))
                break;

            const int idx = key3DToIndex3D({ cx, cy, cz });
            if (idx >= 0)
                policy.emit(idx, out);
        }
    }
    inline void raycastOutmapParallel(
        const VoxelKey3D& o,
        const std::vector<VoxelKey3D>& outmap_targets,
        size_t max_range_vox,
        tbb::enumerable_thread_specific<RayResultSOA>& tls_free
    );

    inline void raycastParallel(
        const VoxelKey3D& o,
        const StampedIndexBuffer& ray,
        size_t max_range_vox,
        tbb::enumerable_thread_specific<RayResultSOA>& tls_free
    );
    inline int key3DToIndex3D(const VoxelKey3D& k) const {
        const int ox = occ_map_info_.origin_key_.x;
        const int oy = occ_map_info_.origin_key_.y;
        const int oz = occ_map_info_.origin_key_.z;
        const int half_x = occ_map_info_.nx_ >> 1;
        const int half_y = occ_map_info_.ny_ >> 1;
        const int half_z = occ_map_info_.nz_ >> 1;

        int dx = k.x - ox + half_x;
        int dy = k.y - oy + half_y;
        int dz = k.z - oz + half_z;

        if (dx < 0 || dx >= occ_map_info_.nx_ || dy < 0 || dy >= occ_map_info_.ny_ || dz < 0
            || dz >= occ_map_info_.nz_)
            return -1;

        int rx = dx + occ_map_info_.ox_;
        if (rx >= occ_map_info_.nx_)
            rx -= occ_map_info_.nx_;
        else if (rx < 0)
            rx += occ_map_info_.nx_;

        int ry = dy + occ_map_info_.oy_;
        if (ry >= occ_map_info_.ny_)
            ry -= occ_map_info_.ny_;
        else if (ry < 0)
            ry += occ_map_info_.ny_;

        int rz = dz + occ_map_info_.oz_;
        if (rz >= occ_map_info_.nz_)
            rz -= occ_map_info_.nz_;
        else if (rz < 0)
            rz += occ_map_info_.nz_;

        return (rx * occ_map_info_.ny_ + ry) * occ_map_info_.nz_ + rz;
    }

    inline VoxelKey3D index3DToKey3D(int idx) const {
        const int half_x = occ_map_info_.nx_ >> 1;
        const int half_y = occ_map_info_.ny_ >> 1;
        const int half_z = occ_map_info_.nz_ >> 1;

        int rz = idx;
        int qxy = rz / occ_map_info_.nz_;
        rz -= qxy * occ_map_info_.nz_;

        int ry = qxy;
        int qx = ry / occ_map_info_.ny_;
        ry -= qx * occ_map_info_.ny_;

        int rx = qx;

        int dx = rx - occ_map_info_.ox_;
        if (dx < 0)
            dx += occ_map_info_.nx_;
        else if (dx >= occ_map_info_.nx_)
            dx -= occ_map_info_.nx_;

        int dy = ry - occ_map_info_.oy_;
        if (dy < 0)
            dy += occ_map_info_.ny_;
        else if (dy >= occ_map_info_.ny_)
            dy -= occ_map_info_.ny_;

        int dz = rz - occ_map_info_.oz_;
        if (dz < 0)
            dz += occ_map_info_.nz_;
        else if (dz >= occ_map_info_.nz_)
            dz -= occ_map_info_.nz_;

        return { occ_map_info_.origin_key_.x + dx - half_x,
                 occ_map_info_.origin_key_.y + dy - half_y,
                 occ_map_info_.origin_key_.z + dz - half_z };
    }

    inline VoxelKey3D worldToKey3D(const Eigen::Vector3f& p) const {
        Eigen::Vector3f q = p / occ_map_info_.voxel_size_;
        return { int(std::floor(q.x())), int(std::floor(q.y())), int(std::floor(q.z())) };
    }
    inline Eigen::Vector3f key3DToWorld(const VoxelKey3D& k) const {
        return Eigen::Vector3f(k.x, k.y, k.z) * occ_map_info_.voxel_size_;
    }
    struct OccMapInfo {
        float voxel_size_;
        Eigen::Vector3f origin_, size_;

        int nx_, ny_, nz_;
        std::vector<Cell> grid_;

        VoxelKey3D origin_key_;
        int ox_, oy_, oz_;
        VoxelKey3D min_key_, max_key_;
    } occ_map_info_;

    std::vector<int> occupied_buffer_idx_;
    std::vector<int> occupied_pos_;
    StampedIndexBuffer hit_buf_, free_buf_, ray_buf_;
    StampedIndexBuffer rise_buf_; // free/unknown → occupied
    StampedIndexBuffer fall_buf_; // occupied → free/unknown
    std::vector<int8_t> prev_occupied_; // 记录上一帧是否 occupied
    uint32_t stamp_now_ = 1;
    Clock now_;

    Parameters params_;
};

} // namespace rose_map
