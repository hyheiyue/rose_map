#pragma once

#include "occ_map.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <opencv2/core/mat.hpp>
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
        return occ_map_info_.origin_
            - Eigen::Vector3f(0.0f, 0.0f, params_.acc_map_params.origin2base);
    }

    inline int key2DToIndex2D(const VoxelKey2D& k) const {
        const int ox = occ_map_info_.origin_key_.x;
        const int oy = occ_map_info_.origin_key_.y;
        const int half_x = occ_map_info_.nx_ >> 1;
        const int half_y = occ_map_info_.ny_ >> 1;

        int dx = k.x - ox + half_x;
        int dy = k.y - oy + half_y;

        if (dx < 0 || dx >= occ_map_info_.nx_ || dy < 0 || dy >= occ_map_info_.ny_)
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

        return rx + ry * occ_map_info_.nx_;
    }

    inline VoxelKey2D index2DToKey2D(int idx) const {
        const int half_x = occ_map_info_.nx_ >> 1;
        const int half_y = occ_map_info_.ny_ >> 1;

        int ry = idx / occ_map_info_.nx_;
        int rx = idx - ry * occ_map_info_.nx_;

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

        return { occ_map_info_.origin_key_.x + dx - half_x,
                 occ_map_info_.origin_key_.y + dy - half_y };
    }
    // inline int key2DToIndex2D(const VoxelKey2D& k) const {
    //     int dx = k.x - origin_key_.x + nx_ / 2;
    //     int dy = k.y - origin_key_.y + ny_ / 2;

    //     if ((unsigned)dx >= (unsigned)nx_ || (unsigned)dy >= (unsigned)ny_) {
    //         return -1;
    //     }

    //     int rx = (dx + ox_) % nx_;
    //     int ry = (dy + oy_) % ny_;

    //     return rx + ry * nx_;
    // }

    // inline VoxelKey2D index2DToKey2D(int idx) const {
    //     int ry = idx / nx_;
    //     int rx = idx % nx_;

    //     int dx = (rx - ox_ + nx_) % nx_;
    //     int dy = (ry - oy_ + ny_) % ny_;

    //     return { origin_key_.x + dx - nx_ / 2, origin_key_.y + dy - ny_ / 2 };
    // }
    inline VoxelKey2D worldToKey2D(const Eigen::Vector2f& p) const {
        Eigen::Vector2f q = p / occ_map_info_.voxel_size_;
        return { static_cast<int>(std::floor(q.x())), static_cast<int>(std::floor(q.y())) };
    }

    inline Eigen::Vector3f key2DToWorld(const VoxelKey2D& k) const {
        Eigen::Vector3f p(static_cast<float>(k.x), static_cast<float>(k.y), 0.0f);
        p *= occ_map_info_.voxel_size_;
        p.z() = getRoboBase().z();
        return p;
    }

    inline bool isPassableCached(int idx, const Eigen::Vector3f& robo_base) const {
        const Cell& c = occ_map_info_.grid_[idx];
        if (!isOccupied(idx, now_))
            return true;
        Eigen::Vector3f p = key3DToWorld(index3DToKey3D(idx));
        Eigen::Vector3f diff = p - robo_base;
        if (std::abs(diff.z()) < params_.acc_map_params.min_diff_z)
            return true;
        return false;
    }
    bool isBlockedWorld(const Eigen::Vector2f& world_xy) const {
        // 1. 先尝试 OccMap 判断（地图内）
        auto key = worldToKey2D(world_xy);
        int idx2d = key2DToIndex2D(key);
        if (idx2d >= 0) {
            return (*curr_).at<uint8_t>(0, idx2d) == 1;
        }

        // 2. OccMap 外但有静态地图 → 用图像判断
        if (has_image_map_) {
            int ix, iy;
            if (worldToImage(world_xy, ix, iy)) {
                // 在图像范围内 → 直接查 mask
                if (iy >= 0 && iy < image_height_ && ix >= 0 && ix < image_width_) {
                    return image_mask_[iy * image_width_ + ix] == 0;
                }
            }
        }

        // 3. 图像变换失败或图像外 → 视为 free
        return false;
    }
    void update(Clock now);

    std::vector<Eigen::Vector4f> getOccupiedPoints() const;
    const cv::Mat& acc_grid_view() const {
        return *curr_;
    }

    const cv::Mat& last_acc_grid_view() const {
        return (curr_ == &buf0_) ? buf1_ : buf0_;
    }
    void applyMorphology(cv::Mat& src);
    bool loadRosMapYaml(const std::string& yaml_path);
    inline bool worldToImage(const Eigen::Vector2f& pw, int& ix, int& iy) const {
        Eigen::Vector2f p = pw - image_origin_;

        ix = static_cast<int>(std::floor(p.x() / image_resolution_));
        iy = static_cast<int>(std::floor(p.y() / image_resolution_));

        // ROS map: (0,0) 在左下 → OpenCV 在左上
        iy = image_height_ - 1 - iy;

        if (ix < 0 || iy < 0 || ix >= image_width_ || iy >= image_height_)
            return false;

        return true;
    }
    cv::Mat buf0_;
    cv::Mat buf1_;
    cv::Mat* curr_ { nullptr };

    std::vector<uint8_t> image_mask_; // 0 blocked, 1 free
    bool has_image_map_ = false;

    float image_resolution_ = 0.0f;
    Eigen::Vector2f image_origin_; // 左下角 (x,y)
    int image_width_ = 0;
    int image_height_ = 0;
};

} // namespace rose_map
