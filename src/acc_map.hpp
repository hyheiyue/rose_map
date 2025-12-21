#pragma once

#include "occ_map.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

namespace rose_map {

struct VoxelKey2D {
    int x { 0 };
    int y { 0 };

    VoxelKey2D() = default;
    VoxelKey2D(int x_, int y_): x(x_), y(y_) {}
};

class AccMap: public OccMap {
public:
    using Ptr = std::shared_ptr<AccMap>;

    explicit AccMap(const YAML::Node& config): OccMap(config) {
        params_.load(config["acc_map"]);

        const int size2d = nx_ * ny_;
        buf0_.assign(size2d, 1);
        buf1_.assign(size2d, 1);

        curr_ = &buf0_;

        upper_idx_.reserve(1024);
        lower_idx_.reserve(1024);
    }

    static Ptr create(const YAML::Node& config) {
        return std::make_shared<AccMap>(config);
    }

    Eigen::Vector3f getRoboBase() const {
        // Robot base position in world frame
        return origin_ - Eigen::Vector3f(0.0f, 0.0f, params_.origin2base);
    }

    inline int key2DToIndex2D(const VoxelKey2D& k) const {
        int dx = k.x - origin_key_.x + nx_ / 2;
        int dy = k.y - origin_key_.y + ny_ / 2;

        if ((unsigned)dx >= (unsigned)nx_ || (unsigned)dy >= (unsigned)ny_) {
            return -1;
        }

        int rx = (dx + ox_) % nx_;
        int ry = (dy + oy_) % ny_;

        return rx + ry * nx_;
    }

    inline VoxelKey2D index2DToKey2D(int idx) const {
        int ry = idx / nx_;
        int rx = idx % nx_;

        int dx = (rx - ox_ + nx_) % nx_;
        int dy = (ry - oy_ + ny_) % ny_;

        return { origin_key_.x + dx - nx_ / 2, origin_key_.y + dy - ny_ / 2 };
    }

    inline VoxelKey2D worldToKey2D(const Eigen::Vector2f& p) const {
        Eigen::Vector2f q = p / voxel_size_;
        return { static_cast<int>(std::floor(q.x())), static_cast<int>(std::floor(q.y())) };
    }

    inline Eigen::Vector3f key2DToWorld(const VoxelKey2D& k) const {
        Eigen::Vector3f p(static_cast<float>(k.x), static_cast<float>(k.y), 0.0f);
        p *= voxel_size_;
        p.z() = getRoboBase().z();
        return p;
    }
    inline bool isPassableCached(int idx, const Eigen::Vector3f& robo_base) const {
        const Cell& c = grid_[idx];
        if (!isOccupied(idx, now_))
            return true;
        Eigen::Vector3f p = key3DToWorld(index3DToKey3D(idx));
        Eigen::Vector3f diff = p - robo_base;
        if (diff.z() < params_.min_diff_z)
            return true;
        return false;
    }
    void update(Clock now) {
        OccMap::update(now);

        upper_idx_.clear();
        lower_idx_.clear();

        const int size2d = nx_ * ny_;
        std::vector<uint8_t>* other = (curr_ == &buf0_) ? &buf1_ : &buf0_;

        std::fill(other->begin(), other->end(), static_cast<uint8_t>(1));

        std::vector<int> block_cnt(size2d, 0);

        const Eigen::Vector3f robo_base = getRoboBase();

        for (int idx3d: occupied_buffer_idx_) {
            if (!isOccupied(idx3d, now))
                continue;

            VoxelKey3D k3 = index3DToKey3D(idx3d);
            VoxelKey2D k2 { k3.x, k3.y };
            int idx2d = key2DToIndex2D(k2);
            if (idx2d < 0)
                continue;
            if (!isPassableCached(idx3d, robo_base)) {
                block_cnt[idx2d]++;
            }
        }

        for (int i = 0; i < size2d; ++i) {
            bool blocked = false;

            if (block_cnt[i] >= params_.min_block_count)
                blocked = true;
            if (params_.block_ratio > 0.0f) {
                float ratio = static_cast<float>(block_cnt[i]) / static_cast<float>(nz_);
                if (ratio >= params_.block_ratio) {
                    blocked = true;
                } else {
                    blocked = false;
                }
            }

            (*other)[i] = blocked ? 0 : 1;
        }

        for (int i = 0; i < size2d; ++i) {
            bool was_passable = ((*curr_)[i] != 0);
            bool now_passable = ((*other)[i] != 0);
            if (was_passable && !now_passable) {
                upper_idx_.push_back(i);
            } else if (!was_passable && now_passable) {
                lower_idx_.push_back(i);
            }
        }

        curr_ = other;
    }

    std::vector<Eigen::Vector4f> getOccupiedPoints() const {
        std::vector<Eigen::Vector4f> pts;
        pts.reserve(nx_ * ny_ / 8 + 16);

        for (int i = 0; i < static_cast<int>(curr_->size()); ++i) {
            if ((*curr_)[i] == 0) {
                VoxelKey2D key2d = index2DToKey2D(i);
                Eigen::Vector3f p = key2DToWorld(key2d);
                pts.emplace_back(p.x(), p.y(), p.z(), 0.0f);
            }
        }

        Eigen::Vector3f robo_base = getRoboBase();
        pts.emplace_back(robo_base.x(), robo_base.y(), robo_base.z(), 0.0f);

        return pts;
    }

    struct Params {
        float origin2base { 0.0f };
        float min_diff_z { 0.0f };

        int min_block_count { 1 };
        float block_ratio { 0.4f };

        void load(const YAML::Node& config) {
            origin2base = config["origin2base"].as<float>();
            min_diff_z = config["min_diff_z"].as<float>();
            min_block_count = config["min_block_count"].as<int>(1);
            block_ratio = config["block_ratio"].as<float>(0.0f);
        }
    } params_;

    const std::vector<uint8_t>& acc_grid_view() const {
        return *curr_;
    }

    const std::vector<uint8_t>& last_acc_grid_view() const {
        return (curr_ == &buf0_) ? buf1_ : buf0_;
    }

private:
    std::vector<uint8_t> buf0_;
    std::vector<uint8_t> buf1_;
    std::vector<uint8_t>* curr_ { nullptr };

    std::vector<int> upper_idx_;
    std::vector<int> lower_idx_;
};

} // namespace rose_map
