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

    explicit AccMap(rclcpp::Node& node);

    static Ptr create(rclcpp::Node& node) {
        return std::make_shared<AccMap>(node);
    }

    Eigen::Vector3f getRoboBase() const {
        // Robot base position in world frame
        return origin_ - Eigen::Vector3f(0.0f, 0.0f, params_.acc_map_params.origin2base);
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
        if (diff.z() < params_.acc_map_params.min_diff_z)
            return true;
        return false;
    }
    void update(Clock now);

    std::vector<Eigen::Vector4f> getOccupiedPoints() const;
    const std::vector<uint8_t>& acc_grid_view() const {
        return *curr_;
    }

    const std::vector<uint8_t>& last_acc_grid_view() const {
        return (curr_ == &buf0_) ? buf1_ : buf0_;
    }

    std::vector<uint8_t> buf0_;
    std::vector<uint8_t> buf1_;
    std::vector<uint8_t>* curr_ { nullptr };

    std::vector<int> upper_idx_;
    std::vector<int> lower_idx_;
};

} // namespace rose_map
