#pragma once

#include "ankerl/unordered_dense.h"
#include "rose_map/common.hpp"
#include "rose_map/yaml_eigen.hpp"

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

namespace rose_map {

struct VoxelKey {
    int x, y, z;
};

class OccMap {
public:
    using Ptr = std::shared_ptr<OccMap>;

    explicit OccMap(const YAML::Node& config) {
        voxel_size_ = config["voxel_size"].as<float>();
        size_ = config["size"].as<Eigen::Vector3f>();
        origin_ = config["origin"].as<Eigen::Vector3f>();
        params_.load(config);

        Eigen::Vector3f half = size_ * 0.5f;

        min_key_ = worldToKey(origin_ - half);
        max_key_ = worldToKey(origin_ + half);

        nx_ = max_key_.x - min_key_.x + 1;
        ny_ = max_key_.y - min_key_.y + 1;
        nz_ = max_key_.z - min_key_.z + 1;

        grid_.resize(size_t(nx_) * ny_ * nz_);
        hit_mask_.resize(grid_.size(), 0);
        free_mask_.resize(grid_.size(), 0);
        up_mask_.resize(grid_.size(), 0);
        down_mask_.resize(grid_.size(), 0);
        ray_mask_.resize(grid_.size(), 0);

        origin_key_ = worldToKey(origin_);
        ox_ = oy_ = oz_ = 0;

        active_idx_.reserve(50000);
        hit_buffer_idx_.reserve(50000);
        free_buffer_idx_.reserve(50000);
        ray_buffer_idx_.reserve(50000);
        
    }

    static Ptr create(const YAML::Node& config) {
        return std::make_shared<OccMap>(config);
    }
    std::vector<Eigen::Vector4f> getOccupiedPoints() const {
        std::vector<Eigen::Vector4f> pts;
        pts.reserve(active_idx_.size());
        for (int idx: active_idx_) {
            const Cell& c = grid_[idx];
            if (!c.active)
                continue;
            if (c.log_odds <= params_.occ_th)
                continue;
            auto p = keyToWorld(indexToKey(idx));
            pts.emplace_back(p.x(), p.y(), p.z(), p.z() - origin_.z());
        }
        return pts;
    }

    void setOrigin(const Eigen::Vector3f& o) {
        VoxelKey new_origin = worldToKey(o);
        VoxelKey shift { new_origin.x - origin_key_.x,
                         new_origin.y - origin_key_.y,
                         new_origin.z - origin_key_.z };
        const int min_shift = params_.min_shift;
        if (std::abs(shift.x) < min_shift && std::abs(shift.y) < min_shift
            && std::abs(shift.z) < min_shift) {
            return;
        }
        // shift 太大 → 全清
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

    void insertPointCloud(
        const std::vector<Eigen::Vector3f>& pts,
        const Eigen::Vector3f& sensor_origin,
        Clock t
    ) {
        const float max_r2 = params_.max_ray_range * params_.max_ray_range;
        ray_buffer_idx_.clear();
        hit_buffer_idx_.clear();
        free_buffer_idx_.clear();
        const VoxelKey sensor_key = worldToKey(sensor_origin);
        for (const auto& p: pts) {
            if ((p - sensor_origin).squaredNorm() > max_r2)
                continue;

            VoxelKey k_hit = worldToKey(p);
            int hit_idx = keyToIndex(k_hit);
            if (hit_idx < 0)
                continue;
            
            RayKey ray { sensor_key, k_hit };
            if (!ray_mask_[hit_idx]) {
                ray_buffer_idx_.push_back(hit_idx);
                ray_mask_[hit_idx] = 1;
            }
            markHitWithNeighbors(k_hit);
        }
        for (const auto& idx: ray_buffer_idx_) {
            auto key = indexToKey(idx);
            raycastFreeKey(sensor_key,key);
            ray_mask_[idx] = 0;
        }

        commitFree(t);
        commitHits(t);
    }
    bool isOccupied(int idx, Clock now) const {
        if (idx < 0)
            return params_.unknown_is_occupied;

        const Cell& c = grid_[idx];
        if (!c.active || now - c.last_update > params_.timeout)
            return false;

        return c.log_odds > params_.occ_th;
    }
    bool isOccupied(const VoxelKey& k, Clock now) const {
        int idx = keyToIndex(k);
        return isOccupied(idx, now);
    }
    bool isOccupied(const Eigen::Vector3f& p, Clock now) const {
        VoxelKey k = worldToKey(p);
        return isOccupied(k, now);
    }
    void update(Clock now) {
        decayActive(now);
        now_ = now;
    }
    void updateEnd() {
        up_buffer_idx_.clear();
        down_buffer_idx_.clear();
        std::fill(up_mask_.begin(), up_mask_.end(), 0);
        std::fill(down_mask_.begin(), down_mask_.end(), 0);
    }
    Eigen::Vector3f origin() const {
        return origin_;
    }

    struct Cell {
        float log_odds = 0.f;
        Clock last_update = 0.0;
        bool active = false;
    };
    struct RayKey {
        VoxelKey o;
        VoxelKey h;
    };

    struct RayKeyHash {
        size_t operator()(const RayKey& r) const noexcept {
            size_t h1 = ((uint64_t)(r.o.x & 0x1FFFFF) << 42) | ((uint64_t)(r.o.y & 0x1FFFFF) << 21)
                | ((uint64_t)(r.o.z & 0x1FFFFF));
            size_t h2 = ((uint64_t)(r.h.x & 0x1FFFFF) << 42) | ((uint64_t)(r.h.y & 0x1FFFFF) << 21)
                | ((uint64_t)(r.h.z & 0x1FFFFF));
            return h1 ^ (h2 * 1315423911ull);
        }
    };

    struct RayKeyEq {
        bool operator()(const RayKey& a, const RayKey& b) const noexcept {
            return a.o.x == b.o.x && a.o.y == b.o.y && a.o.z == b.o.z && a.h.x == b.h.x
                && a.h.y == b.h.y && a.h.z == b.h.z;
        }
    };

    inline int keyToIndex(const VoxelKey& k) const {
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

    inline VoxelKey indexToKey(int idx) const {
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
                if (axis == 1)
                    idx = (i * ny_ + slice) * nz_ + j;
                if (axis == 2)
                    idx = (i * ny_ + j) * nz_ + slice;

                grid_[idx] = Cell();
                hit_mask_[idx] = 0;
                free_mask_[idx] = 0;
                up_mask_[idx] = 0;
                down_mask_[idx] = 0;
            }
    }

    void resetAll() {
        std::fill(grid_.begin(), grid_.end(), Cell());
        std::fill(hit_mask_.begin(), hit_mask_.end(), 0);
        std::fill(free_mask_.begin(), free_mask_.end(), 0);
        std::fill(up_mask_.begin(), up_mask_.end(), 0);
        std::fill(down_mask_.begin(), down_mask_.end(), 0);
        active_idx_.clear();
    }

    void markHitWithNeighbors(const VoxelKey& k) {
        static const int neigh[5][3] = { { 0, 0, 0 },
                                         { -1, 0, 0 },
                                         { 1, 0, 0 },
                                         { 0, -1, 0 },
                                         { 0, 1, 0 } };

        for (auto& d: neigh) {
            int idx = keyToIndex({ k.x + d[0], k.y + d[1], k.z + d[2] });
            if (idx < 0 || hit_mask_[idx])
                continue;
            hit_mask_[idx] = 1;
            hit_buffer_idx_.push_back(idx);
        }
    }
    void trackState(bool was_occ, bool now_occ, int idx) {
        if (!was_occ && now_occ) {
            if (idx >= 0 && !up_mask_[idx]) {
                up_buffer_idx_.push_back(idx);
                up_mask_[idx] = 1;
            }
        }
        if (was_occ && !now_occ) {
            if (idx >= 0 && !down_mask_[idx]) {
                down_buffer_idx_.push_back(idx);
                down_mask_[idx] = 1;
            }
        }
    }

    void commitHits(Clock t) {
        for (int idx: hit_buffer_idx_) {
            Cell& c = grid_[idx];
            bool was_occ = isOccupied(idx, t);
            c.log_odds = std::min(c.log_odds + params_.log_hit, params_.log_max);
            c.last_update = t;
            if (!c.active) {
                c.active = true;
                active_idx_.push_back(idx);
            }
            bool now_occ = isOccupied(idx, t);
            trackState(was_occ, now_occ, idx);
            hit_mask_[idx] = 0;
        }
        hit_buffer_idx_.clear();
    }

    void commitFree(Clock t) {
        for (int idx: free_buffer_idx_) {
            Cell& c = grid_[idx];
            bool was_occ = isOccupied(idx, t);
            c.log_odds = std::max(c.log_odds + params_.log_free, params_.log_min);
            c.last_update = t;
            if (!c.active) {
                c.active = true;
                active_idx_.push_back(idx);
            }
            bool now_occ = isOccupied(idx, t);
            trackState(was_occ, now_occ, idx);
            free_mask_[idx] = 0;
        }
        free_buffer_idx_.clear();
    }
    void decayActive(Clock now) {
        for (size_t i = 0; i < active_idx_.size();) {
            int idx = active_idx_[i];
            Cell& c = grid_[idx];
            bool was_occ = isOccupied(idx, now);
            c.log_odds = std::max(
                c.log_odds + float(params_.log_decay * (now - last_decay_)),
                params_.log_min
            );
            bool now_occ = isOccupied(idx, now);
            trackState(was_occ, now_occ, idx);
            if (now - c.last_update > params_.timeout) {
                c.active = false;
                active_idx_[i] = active_idx_.back();
                active_idx_.pop_back();
            } else {
                ++i;
            }
        }
        last_decay_ = now;
    }

    void raycastFreeKey(const VoxelKey& o, const VoxelKey& h) {
        int x = o.x, y = o.y, z = o.z;
        const int x1 = h.x, y1 = h.y, z1 = h.z;

        int dx = std::abs(x1 - x);
        int dy = std::abs(y1 - y);
        int dz = std::abs(z1 - z);

        int sx = (x1 > x) ? 1 : -1;
        int sy = (y1 > y) ? 1 : -1;
        int sz = (z1 > z) ? 1 : -1;
        float tMaxX = 0.5f;
        float tMaxY = 0.5f;
        float tMaxZ = 0.5f;

        float tDeltaX = (dx > 0) ? (1.0f / dx) : std::numeric_limits<float>::infinity();
        float tDeltaY = (dy > 0) ? (1.0f / dy) : std::numeric_limits<float>::infinity();
        float tDeltaZ = (dz > 0) ? (1.0f / dz) : std::numeric_limits<float>::infinity();
        const int max_steps = dx + dy + dz + 1;
        int steps = 0;
        while ((x != x1 || y != y1 || z != z1) && steps++ < max_steps) {
            int idx = keyToIndex({ x, y, z });
            if (idx >= 0 && !free_mask_[idx]) {
                free_mask_[idx] = 1;
                free_buffer_idx_.push_back(idx);
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

    inline VoxelKey worldToKey(const Eigen::Vector3f& p) const {
        Eigen::Vector3f q = p / voxel_size_;
        return { int(std::floor(q.x())), int(std::floor(q.y())), int(std::floor(q.z())) };
    }

    inline Eigen::Vector3f keyToWorld(const VoxelKey& k) const {
        return Eigen::Vector3f(k.x, k.y, k.z) * voxel_size_;
    }

    static inline float intBound(float s, float ds) {
        if (ds == 0)
            return 1e6f;
        return ds > 0 ? (std::floor(s + 1) - s) / ds : (s - std::floor(s)) / -ds;
    }

    float voxel_size_;
    Eigen::Vector3f origin_, size_;

    // fixed grid
    int nx_, ny_, nz_;
    std::vector<Cell> grid_;

    // ring offset
    VoxelKey origin_key_;
    int ox_, oy_, oz_;

    // buffers
    std::vector<int> active_idx_;
    std::vector<int> hit_buffer_idx_;
    std::vector<int> free_buffer_idx_;
    std::vector<int> up_buffer_idx_;
    std::vector<int> down_buffer_idx_;
    std::vector<int> ray_buffer_idx_;

    std::vector<uint8_t> hit_mask_;
    std::vector<uint8_t> free_mask_;
    std::vector<uint8_t> up_mask_;
    std::vector<uint8_t> down_mask_;
    std::vector<uint8_t> ray_mask_;
    
    
    VoxelKey min_key_, max_key_;
    Clock last_decay_ = 0.0;
    Clock now_;
    struct Params {
        float log_hit = 0.85f;
        float log_free = -0.4f;
        float log_decay = -0.05f;
        float log_min = -5.f;
        float log_max = 5.f;
        float occ_th = 1.2f;
        int min_shift = 1;
        double timeout = 0.8;
        double max_ray_range = 20.0;
        bool unknown_is_occupied = false;

        void load(const YAML::Node& c) {
            if (c["log_hit"])
                log_hit = c["log_hit"].as<float>();
            if (c["log_free"])
                log_free = c["log_free"].as<float>();
            if (c["log_decay"])
                log_decay = c["log_decay"].as<float>();
            if (c["log_min"])
                log_min = c["log_min"].as<float>();
            if (c["log_max"])
                log_max = c["log_max"].as<float>();
            if (c["occ_th"])
                occ_th = c["occ_th"].as<float>();
            if (c["min_shift"])
                min_shift = c["min_shift"].as<float>();
            if (c["timeout"])
                timeout = c["timeout"].as<double>();
            if (c["max_ray_range"])
                max_ray_range = c["max_ray_range"].as<double>();
            if (c["unknown_is_occupied"])
                unknown_is_occupied = c["unknown_is_occupied"].as<bool>();
        }
    } params_;
};

} // namespace rose_map
