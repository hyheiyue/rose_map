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
};

/**
 * @brief 每帧唯一索引收集器（buffer + stamp 封装）
 */
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
    const std::vector<Eigen::Vector3i> HIT_D = {
        { 0, 0, 0 },
        //  { -1, 0, 0 },
        //  { 1, 0, 0 },
        //  { 0, -1, 0 },
        //  { 0, 1, 0 }
    };
    const std::vector<Eigen::Vector3i> RAY_D = {
        { 0, 0, 0 },
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

        const Cell& c = grid_[idx];
        if (now - c.last_update > params_.occ_map_params.timeout)
            return false;

        return c.log_odds > params_.occ_map_params.occ_th;
    }

    void update(Clock now);
    Eigen::Vector3f origin() const {
        return origin_;
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

    void raycastFreeKey(const VoxelKey3D& o, const VoxelKey3D& h) {
        if (o.x == h.x && o.y == h.y && o.z == h.z)
            return;

        int x = o.x, y = o.y, z = o.z;
        int dx = std::abs(h.x - x);
        int dy = std::abs(h.y - y);
        int dz = std::abs(h.z - z);

        int sx = (h.x > x) ? 1 : -1;
        int sy = (h.y > y) ? 1 : -1;
        int sz = (h.z > z) ? 1 : -1;

        float tMaxX = 0.5f, tMaxY = 0.5f, tMaxZ = 0.5f;
        float tDeltaX = dx ? 1.f / dx : std::numeric_limits<float>::infinity();
        float tDeltaY = dy ? 1.f / dy : std::numeric_limits<float>::infinity();
        float tDeltaZ = dz ? 1.f / dz : std::numeric_limits<float>::infinity();

        int max_steps = dx + dy + dz + 1;
        while ((x != h.x || y != h.y || z != h.z) && max_steps--) {
            for (const auto& n: FREE_D) {
                int idx = key3DToIndex3D({ x + n[0], y + n[1], z + n[2] });
                if (idx >= 0)
                    free_buf_.tryPush(idx, stamp_now_);
            }

            if (tMaxX < tMaxY) {
                if (tMaxX < tMaxZ) {
                    x += sx;
                    tMaxX += tDeltaX;
                } else {
                    z += sz;
                    tMaxZ += tDeltaZ;
                }
            } else {
                if (tMaxY < tMaxZ) {
                    y += sy;
                    tMaxY += tDeltaY;
                } else {
                    z += sz;
                    tMaxZ += tDeltaZ;
                }
            }
        }
    }

    inline void raycastFreeKeyTLS(const VoxelKey3D& o, const VoxelKey3D& h, std::vector<int>& out) {
        if (o.x == h.x && o.y == h.y && o.z == h.z)
            return;

        int x = o.x, y = o.y, z = o.z;
        int dx = std::abs(h.x - x);
        int dy = std::abs(h.y - y);
        int dz = std::abs(h.z - z);

        int sx = (h.x > x) ? 1 : -1;
        int sy = (h.y > y) ? 1 : -1;
        int sz = (h.z > z) ? 1 : -1;

        float tMaxX = 0.5f, tMaxY = 0.5f, tMaxZ = 0.5f;
        float tDeltaX = dx ? 1.f / dx : std::numeric_limits<float>::infinity();
        float tDeltaY = dy ? 1.f / dy : std::numeric_limits<float>::infinity();
        float tDeltaZ = dz ? 1.f / dz : std::numeric_limits<float>::infinity();

        int max_steps = dx + dy + dz + 1;

        while ((x != h.x || y != h.y || z != h.z) && max_steps--) {
            for (const auto& n: FREE_D) {
                int idx = key3DToIndex3D({ x + n[0], y + n[1], z + n[2] });
                if (idx >= 0) {
                    out.push_back(idx);
                }
            }
            // 3D DDA
            if (tMaxX < tMaxY) {
                if (tMaxX < tMaxZ) {
                    x += sx;
                    tMaxX += tDeltaX;
                } else {
                    z += sz;
                    tMaxZ += tDeltaZ;
                }
            } else {
                if (tMaxY < tMaxZ) {
                    y += sy;
                    tMaxY += tDeltaY;
                } else {
                    z += sz;
                    tMaxZ += tDeltaZ;
                }
            }
        }
    }
    inline void raycastFreeKeyTLS_SyncStep_Parallel(
        const VoxelKey3D& o,
        const std::vector<VoxelKey3D>& h_list,
        const std::vector<size_t>& max_steps,
        size_t max_range_vox,
        tbb::enumerable_thread_specific<std::vector<int>>& tls_free
    ) {
        tbb::parallel_for(size_t(0), h_list.size(), size_t(32), [&](size_t i) {
            auto& out = tls_free.local();
            int x = o.x, y = o.y, z = o.z;
            int sx = (h_list[i].x > x) ? 1 : -1;
            int sy = (h_list[i].y > y) ? 1 : -1;
            int sz = (h_list[i].z > z) ? 1 : -1;

            size_t steps = max_steps[i];
            if (steps > max_range_vox)
                steps = max_range_vox;

            for (size_t s = 0; s < steps; ++s) {
                if (x == h_list[i].x && y == h_list[i].y && z == h_list[i].z)
                    break;

                x += sx;
                y += sy;
                z += sz;

                int idx = key3DToIndex3D({ x, y, z });
                if (idx >= 0)
                    out.push_back(idx);
                else
                    break;
            }
        });
    }

    inline int key3DToIndex3D(const VoxelKey3D& k) const {
        const int ox = origin_key_.x;
        const int oy = origin_key_.y;
        const int oz = origin_key_.z;
        const int half_x = nx_ >> 1;
        const int half_y = ny_ >> 1;
        const int half_z = nz_ >> 1;

        int dx = k.x - ox + half_x;
        int dy = k.y - oy + half_y;
        int dz = k.z - oz + half_z;

        if (dx < 0 || dx >= nx_ || dy < 0 || dy >= ny_ || dz < 0 || dz >= nz_)
            return -1;

        int rx = dx + ox_;
        if (rx >= nx_)
            rx -= nx_;
        else if (rx < 0)
            rx += nx_;

        int ry = dy + oy_;
        if (ry >= ny_)
            ry -= ny_;
        else if (ry < 0)
            ry += ny_;

        int rz = dz + oz_;
        if (rz >= nz_)
            rz -= nz_;
        else if (rz < 0)
            rz += nz_;

        return (rx * ny_ + ry) * nz_ + rz;
    }

    inline VoxelKey3D index3DToKey3D(int idx) const {
        const int half_x = nx_ >> 1;
        const int half_y = ny_ >> 1;
        const int half_z = nz_ >> 1;

        int rz = idx;
        int qxy = rz / nz_;
        rz -= qxy * nz_; // 代替 % nz_

        int ry = qxy;
        int qx = ry / ny_;
        ry -= qx * ny_; // 代替 % ny_

        int rx = qx;

        int dx = rx - ox_;
        if (dx < 0)
            dx += nx_;
        else if (dx >= nx_)
            dx -= nx_;

        int dy = ry - oy_;
        if (dy < 0)
            dy += ny_;
        else if (dy >= ny_)
            dy -= ny_;

        int dz = rz - oz_;
        if (dz < 0)
            dz += nz_;
        else if (dz >= nz_)
            dz -= nz_;

        return { origin_key_.x + dx - half_x,
                 origin_key_.y + dy - half_y,
                 origin_key_.z + dz - half_z };
    }
    // inline int key3DToIndex3D(const VoxelKey3D& k) const {
    //     int dx = k.x - origin_key_.x + nx_ / 2;
    //     int dy = k.y - origin_key_.y + ny_ / 2;
    //     int dz = k.z - origin_key_.z + nz_ / 2;

    //     if ((unsigned)dx >= (unsigned)nx_ || (unsigned)dy >= (unsigned)ny_
    //         || (unsigned)dz >= (unsigned)nz_)
    //         return -1;

    //     int rx = (dx + ox_) % nx_;
    //     int ry = (dy + oy_) % ny_;
    //     int rz = (dz + oz_) % nz_;

    //     return (rx * ny_ + ry) * nz_ + rz;
    // }

    // inline VoxelKey3D index3DToKey3D(int idx) const {
    //     int rz = idx % nz_;
    //     int ry = (idx / nz_) % ny_;
    //     int rx = idx / (ny_ * nz_);

    //     int dx = (rx - ox_ + nx_) % nx_;
    //     int dy = (ry - oy_ + ny_) % ny_;
    //     int dz = (rz - oz_ + nz_) % nz_;

    //     return { origin_key_.x + dx - nx_ / 2,
    //              origin_key_.y + dy - ny_ / 2,
    //              origin_key_.z + dz - nz_ / 2 };
    // }

    inline VoxelKey3D worldToKey3D(const Eigen::Vector3f& p) const {
        Eigen::Vector3f q = p / voxel_size_;
        return { int(std::floor(q.x())), int(std::floor(q.y())), int(std::floor(q.z())) };
    }
    inline Eigen::Vector3f key3DToWorld(const VoxelKey3D& k) const {
        return Eigen::Vector3f(k.x, k.y, k.z) * voxel_size_;
    }

    float voxel_size_;
    Eigen::Vector3f origin_, size_;

    int nx_, ny_, nz_;
    std::vector<Cell> grid_;

    VoxelKey3D origin_key_;
    int ox_, oy_, oz_;

    std::vector<int> occupied_buffer_idx_;
    std::vector<int> occupied_pos_;
    StampedIndexBuffer hit_buf_, free_buf_, ray_buf_;
    StampedIndexBuffer rise_buf_; // free/unknown → occupied
    StampedIndexBuffer fall_buf_; // occupied → free/unknown
    std::vector<int8_t> prev_occupied_; // 记录上一帧是否 occupied
    uint32_t stamp_now_ = 1;
    VoxelKey3D min_key_, max_key_;
    Clock now_;

    Parameters params_;
};

} // namespace rose_map
