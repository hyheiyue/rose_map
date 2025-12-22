#pragma once

#include "ankerl/unordered_dense.h"
#include "rose_map/common.hpp"
#include "rose_map/yaml_eigen.hpp"

#include "execution"
#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_invoke.h>
#include <thread>
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

    explicit OccMap(const YAML::Node& config) {
        voxel_size_ = config["occ_map"]["voxel_size"].as<float>();
        size_ = config["occ_map"]["size"].as<Eigen::Vector3f>();
        origin_ = config["occ_map"]["origin"].as<Eigen::Vector3f>();
        params_.load(config["occ_map"]);

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
        slide_cleared_idx_.reserve(nx_ * ny_ + nx_ * nz_ + ny_ * nz_);
        occupied_buffer_idx_.reserve(N);
    }

    static Ptr create(const YAML::Node& config) {
        return std::make_shared<OccMap>(config);
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
    ) {
        ++stamp_now_;

        hit_buf_.clear();
        free_buf_.clear();
        ray_buf_.clear();

        const float max_r2 = params_.max_ray_range * params_.max_ray_range;
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
                        int idx =
                            key3DToIndex3D({ k_hit.x + n[0], k_hit.y + n[1], k_hit.z + n[2] });
                        if (idx >= 0) {
                            local.hit.push_back(idx);
                        }
                    }
                    if (params_.use_ray) {
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
            if (params_.use_ray) {
                for (int idx: local.ray)
                    ray_buf_.tryPush(idx, stamp_now_);
            }
            for (int idx: local.hit)
                hit_buf_.tryPush(idx, stamp_now_);
        }
        if (params_.use_ray) {
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

    inline bool isOccupied(int idx, Clock now) const {
        if (idx < 0)
            return params_.unknown_is_occupied;

        const Cell& c = grid_[idx];
        if (now - c.last_update > params_.timeout)
            return false;

        return c.log_odds > params_.occ_th;
    }

    void update(Clock now) {
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

    Eigen::Vector3f origin() const {
        return origin_;
    }
    std::vector<Eigen::Vector4f> getOccupiedPoints(float sample_resolution_m = 0.05) const {
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

    void setOrigin(const Eigen::Vector3f& o) {
        slide_cleared_idx_.clear();
        VoxelKey3D new_origin = worldToKey3D(o);
        VoxelKey3D shift { new_origin.x - origin_key_.x,
                           new_origin.y - origin_key_.y,
                           new_origin.z - origin_key_.z };
        const int min_shift = params_.min_shift;
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
    void slideAxis(int axis, int shift) {
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
    void clearSlice(int axis, int slice) {
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
                slide_cleared_idx_.push_back(idx);
            }
    }
    void resetAll() {
        slide_cleared_idx_.clear(); // === NEW ===
        for (int i = 0; i < (int)grid_.size(); ++i)
            slide_cleared_idx_.push_back(i);
        std::fill(grid_.begin(), grid_.end(), Cell());
        std::fill(hit_buf_.stamp.begin(), hit_buf_.stamp.end(), 0);
        std::fill(free_buf_.stamp.begin(), free_buf_.stamp.end(), 0);
        std::fill(ray_buf_.stamp.begin(), ray_buf_.stamp.end(), 0);
    }

    void markHitWithNeighbors(const VoxelKey3D& k) {
        for (auto& n: HIT_D) {
            int idx = key3DToIndex3D({ k.x + n[0], k.y + n[1], k.z + n[2] });
            if (idx >= 0)
                hit_buf_.tryPush(idx, stamp_now_);
        }
    }
    const std::vector<int>& slideClearedIndices() const {
        return slide_cleared_idx_;
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

    void commitHits(Clock t) {
        for (int idx: hit_buf_.indices) {
            Cell& c = grid_[idx];
            bool was = isOccupied(idx, t);

            c.log_odds = std::min(c.log_odds + params_.log_hit, params_.log_max);
            c.last_update = t;

            bool now = isOccupied(idx, t);
            trackState(was, now, idx);
        }
        hit_buf_.clear();
    }

    void commitFree(Clock t) {
        for (int idx: free_buf_.indices) {
            Cell& c = grid_[idx];
            bool was = isOccupied(idx, t);

            c.log_odds = std::max(c.log_odds + params_.log_free, params_.log_min);
            c.last_update = t;

            bool now = isOccupied(idx, t);
            trackState(was, now, idx);
        }
        free_buf_.clear();
    }

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
    std::vector<int> slide_cleared_idx_;
    uint32_t stamp_now_ = 1;
    VoxelKey3D min_key_, max_key_;
    Clock now_;

    struct Params {
        float log_hit = 0.85f;
        float log_free = -0.4f;
        float log_min = -5.f;
        float log_max = 5.f;
        float occ_th = 1.2f;
        int min_shift = 1;
        double timeout = 0.8;
        double max_ray_range = 20.0;
        bool unknown_is_occupied = false;
        bool use_ray = true;

        void load(const YAML::Node& c) {
            if (c["log_hit"])
                log_hit = c["log_hit"].as<float>();
            if (c["log_free"])
                log_free = c["log_free"].as<float>();
            if (c["log_min"])
                log_min = c["log_min"].as<float>();
            if (c["log_max"])
                log_max = c["log_max"].as<float>();
            if (c["occ_th"])
                occ_th = c["occ_th"].as<float>();
            if (c["min_shift"])
                min_shift = c["min_shift"].as<int>();
            if (c["timeout"])
                timeout = c["timeout"].as<double>();
            if (c["max_ray_range"])
                max_ray_range = c["max_ray_range"].as<double>();
            if (c["unknown_is_occupied"])
                unknown_is_occupied = c["unknown_is_occupied"].as<bool>();
            if (c["use_ray"])
                use_ray = c["use_ray"].as<bool>();
        }
    } params_;
};

} // namespace rose_map
