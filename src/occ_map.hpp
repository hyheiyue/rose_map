#pragma once

#include "parameters.hpp"
#include "rose_map/common.hpp"

#include <Eigen/Core>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
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
    inline void tryPush(int idx, uint32_t now) {
        if (stamp[idx] != now) {
            stamp[idx] = now;
            indices.push_back(idx);
        }
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
    };
    static constexpr int HIT_D[5][3] = { { 0, 0, 0 },
                                         { -1, 0, 0 },
                                         { 1, 0, 0 },
                                         { 0, -1, 0 },
                                         { 0, 1, 0 } };
    static constexpr int RAY_D[1][3] = {
        { 0, 0, 0 },
        //  { -1, 0, 0 },
        //  { 1, 0, 0 },
        //  { 0, -1, 0 },
        //  { 0, 1, 0 }
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
    std::vector<Eigen::Vector4f> getOccupiedPoints(float sample_resolution_m = 0.05) const;

    void setOrigin(const Eigen::Vector3f& o);
    void slideAxis(int axis, int shift);
    void clearSlice(int axis, int slice);
    void resetAll();
    void markHitWithNeighbors(const VoxelKey3D& k) {
        for (auto& n: HIT_D) {
            int idx = key3DToIndex3D({ k.x + n[0], k.y + n[1], k.z + n[2] });
            if (idx >= 0)
                hit_buf_.tryPush(idx, stamp_now_);
        }
    }

    inline void trackState(bool was, bool now, int idx) {
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
                if (idx >= 0)
                    out.push_back(idx);
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

    inline int key3DToIndex3D(const VoxelKey3D& k) const {
        int dx = k.x - origin_key_.x + nx_ / 2;
        int dy = k.y - origin_key_.y + ny_ / 2;
        int dz = k.z - origin_key_.z + nz_ / 2;

        if ((unsigned)dx >= (unsigned)nx_ || (unsigned)dy >= (unsigned)ny_
            || (unsigned)dz >= (unsigned)nz_)
            return -1;

        int rx = (dx + ox_) % nx_;
        int ry = (dy + oy_) % ny_;
        int rz = (dz + oz_) % nz_;

        return (rx * ny_ + ry) * nz_ + rz;
    }

    inline VoxelKey3D index3DToKey3D(int idx) const {
        int rz = idx % nz_;
        int ry = (idx / nz_) % ny_;
        int rx = idx / (ny_ * nz_);

        int dx = (rx - ox_ + nx_) % nx_;
        int dy = (ry - oy_ + ny_) % ny_;
        int dz = (rz - oz_ + nz_) % nz_;

        return { origin_key_.x + dx - nx_ / 2,
                 origin_key_.y + dy - ny_ / 2,
                 origin_key_.z + dz - nz_ / 2 };
    }

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
    uint32_t stamp_now_ = 1;
    VoxelKey3D min_key_, max_key_;
    Clock now_;

    Parameters params_;
};

} // namespace rose_map
