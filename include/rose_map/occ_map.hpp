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

    inline std::vector<int>
    raycastClipToMapParallel(const VoxelKey3D& o, const std::vector<VoxelKey3D>& outmap_list) {
        std::vector<int> clamped_indices(outmap_list.size(), -1);

        const size_t MAX_STEP = occ_map_info_.nx_ + occ_map_info_.ny_ + occ_map_info_.nz_;

        constexpr size_t GRAIN = 32;

        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, outmap_list.size(), GRAIN),
            [&](const tbb::blocked_range<size_t>& r) {
                for (size_t i = r.begin(); i != r.end(); ++i) {
                    const auto& h = outmap_list[i];

                    // --- 体素中心 ---
                    const float px = o.x + 0.5f;
                    const float py = o.y + 0.5f;
                    const float pz = o.z + 0.5f;
                    const float tpx = h.x + 0.5f;
                    const float tpy = h.y + 0.5f;
                    const float tpz = h.z + 0.5f;

                    float dx = tpx - px;
                    float dy = tpy - py;
                    float dz = tpz - pz;

                    const float len = std::sqrt(dx * dx + dy * dy + dz * dz);
                    if (len < 1e-6f)
                        continue;

                    const float invLen = 1.0f / len;
                    dx *= invLen;
                    dy *= invLen;
                    dz *= invLen;

                    int cx = o.x, cy = o.y, cz = o.z;

                    const int stepX = (dx > 0.f) - (dx < 0.f);
                    const int stepY = (dy > 0.f) - (dy < 0.f);
                    const int stepZ = (dz > 0.f) - (dz < 0.f);

                    const float tDeltaX = (std::abs(dx) > 1e-6f)
                        ? std::abs(1.0f / dx)
                        : std::numeric_limits<float>::infinity();
                    const float tDeltaY = (std::abs(dy) > 1e-6f)
                        ? std::abs(1.0f / dy)
                        : std::numeric_limits<float>::infinity();
                    const float tDeltaZ = (std::abs(dz) > 1e-6f)
                        ? std::abs(1.0f / dz)
                        : std::numeric_limits<float>::infinity();

                    float tMaxX = (std::abs(dx) > 1e-6f) ? ((stepX > 0 ? cx + 1.f : cx) - px) / dx
                                                         : std::numeric_limits<float>::infinity();
                    float tMaxY = (std::abs(dy) > 1e-6f) ? ((stepY > 0 ? cy + 1.f : cy) - py) / dy
                                                         : std::numeric_limits<float>::infinity();
                    float tMaxZ = (std::abs(dz) > 1e-6f) ? ((stepZ > 0 ? cz + 1.f : cz) - pz) / dz
                                                         : std::numeric_limits<float>::infinity();

                    const size_t maxStep = std::min(static_cast<size_t>(std::ceil(len)), MAX_STEP);

                    size_t stepCount = 0;
                    int last_valid_idx = -1;

                    while (stepCount < maxStep) {
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

                        const int idx = key3DToIndex3D({ cx, cy, cz });
                        if (idx < 0)
                            break;

                        last_valid_idx = idx;
                        ++stepCount;
                    }

                    clamped_indices[i] = last_valid_idx;
                }
            }
        );

        return clamped_indices;
    }

    inline void raycastFreeKeyTLS_SyncStep_Parallel(
        const VoxelKey3D& o,
        const std::vector<VoxelKey3D>& h_list,
        const std::vector<size_t>& max_steps,
        size_t max_range_vox,
        tbb::enumerable_thread_specific<std::vector<int>>& tls_free
    ) {
        constexpr size_t GRAIN = 32;
        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, h_list.size(), GRAIN),
            [&](const tbb::blocked_range<size_t>& r) {
                for (size_t i = r.begin(); i != r.end(); ++i) {
                    auto& out = tls_free.local();

                    const int tx = h_list[i].x;
                    const int ty = h_list[i].y;
                    const int tz = h_list[i].z;

                    const float px = o.x + 0.5f;
                    const float py = o.y + 0.5f;
                    const float pz = o.z + 0.5f;
                    const float tpx = tx + 0.5f;
                    const float tpy = ty + 0.5f;
                    const float tpz = tz + 0.5f;

                    float dx = tpx - px;
                    float dy = tpy - py;
                    float dz = tpz - pz;
                    const float len = std::sqrt(dx * dx + dy * dy + dz * dz);
                    if (len < 1e-6f)
                        return;
                    const float invLen = 1.0f / len;
                    dx *= invLen;
                    dy *= invLen;
                    dz *= invLen;

                    int cx = o.x, cy = o.y, cz = o.z;

                    const int stepX = (dx > 0.f) - (dx < 0.f);
                    const int stepY = (dy > 0.f) - (dy < 0.f);
                    const int stepZ = (dz > 0.f) - (dz < 0.f);

                    const float tDeltaX = (std::abs(dx) > 1e-6f)
                        ? std::abs(1.0f / dx)
                        : std::numeric_limits<float>::infinity();
                    const float tDeltaY = (std::abs(dy) > 1e-6f)
                        ? std::abs(1.0f / dy)
                        : std::numeric_limits<float>::infinity();
                    const float tDeltaZ = (std::abs(dz) > 1e-6f)
                        ? std::abs(1.0f / dz)
                        : std::numeric_limits<float>::infinity();

                    float tMaxX = (std::abs(dx) > 1e-6f) ? ((stepX > 0 ? cx + 1.f : cx) - px) / dx
                                                         : std::numeric_limits<float>::infinity();
                    float tMaxY = (std::abs(dy) > 1e-6f) ? ((stepY > 0 ? cy + 1.f : cy) - py) / dy
                                                         : std::numeric_limits<float>::infinity();
                    float tMaxZ = (std::abs(dz) > 1e-6f) ? ((stepZ > 0 ? cz + 1.f : cz) - pz) / dz
                                                         : std::numeric_limits<float>::infinity();

                    size_t maxStep = max_steps[i];
                    if (maxStep > max_range_vox)
                        maxStep = max_range_vox;

                    size_t stepCount = 0;

                    while (stepCount < maxStep) {
                        if (cx == tx && cy == ty && cz == tz)
                            break;
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

                        const int idx = key3DToIndex3D({ cx, cy, cz });
                        if (idx < 0)
                            break;
                        out.push_back(idx);
                        ++stepCount;
                    }
                }
            }
        );
    }

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
