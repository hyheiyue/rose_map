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
    bool operator==(const VoxelKey2D& rhs) const {
        return x == rhs.x && y == rhs.y;
    }
};

class AccMap: public OccMap {
public:
    using Ptr = std::shared_ptr<AccMap>;

    explicit AccMap(rclcpp::Node& node);

    static Ptr create(rclcpp::Node& node) {
        return std::make_shared<AccMap>(node);
    }
    void updateRoboBase() {
        Eigen::Vector3f robo_base = occ_map_info_.origin;
        const VoxelKey3D o = worldToKey3D(robo_base);
        const float voxel_size = occ_map_info_.voxel_size;
        const int search_range = static_cast<int>(0.5f / voxel_size); // ±0.5m

        float min_z = std::numeric_limits<float>::max();
        bool found = false;
        const float base_z_world = occ_map_info_.origin.z();
        for (int i = -search_range; i <= search_range; ++i) {
            const int x = o.x + i;
            for (int j = -search_range; j <= search_range; ++j) {
                const int y = o.y + j;
                for (int k = -search_range; k < 0; ++k) {
                    const int z = o.z + k;

                    VoxelKey3D ky { x, y, z };
                    const int idx = key3DToIndex3D(ky);
                    if (idx < 0)
                        continue;

                    if (!isOccupied(idx, now_))
                        continue;

                    const float pz = base_z_world + k * voxel_size;
                    if (pz < min_z) {
                        min_z = pz;
                        found = true;
                    }
                }
            }
        }
        if (found) {
            robo_base.z() = min_z;
        } else {
            robo_base.z() -= params_.acc_map_params.origin2base;
        }

        robo_base_ = robo_base;
    }

    Eigen::Vector3f getRoboBase() const {
        return robo_base_;
    }
    void setOrigin(const Eigen::Vector3f& o) {
        OccMap::setOrigin(o);
        acc_map_info_.tmp_origin = Eigen::Vector2f(o.x(), o.y());
    }
    inline int key2DToIndex2D(const VoxelKey2D& k) const {
        const int ox = acc_map_info_.origin_key.x;
        const int oy = acc_map_info_.origin_key.y;
        const int half_x = acc_map_info_.nx >> 1;
        const int half_y = acc_map_info_.ny >> 1;

        int dx = k.x - ox + half_x;
        int dy = k.y - oy + half_y;

        if (dx < 0 || dx >= acc_map_info_.nx || dy < 0 || dy >= acc_map_info_.ny)
            return -1;

        int rx = dx;
        if (rx >= acc_map_info_.nx)
            rx -= acc_map_info_.nx;
        else if (rx < 0)
            rx += acc_map_info_.nx;

        int ry = dy;
        if (ry >= acc_map_info_.ny)
            ry -= acc_map_info_.ny;
        else if (ry < 0)
            ry += acc_map_info_.ny;

        return rx + ry * acc_map_info_.nx;
    }

    inline VoxelKey2D index2DToKey2D(int idx) const {
        const int ox = acc_map_info_.origin_key.x;
        const int oy = acc_map_info_.origin_key.y;
        const int half_x = acc_map_info_.nx >> 1;
        const int half_y = acc_map_info_.ny >> 1;

        int ry = idx / acc_map_info_.nx;
        int rx = idx - ry * acc_map_info_.nx;

        int dx = rx;
        if (dx < 0)
            dx += acc_map_info_.nx;
        else if (dx >= acc_map_info_.nx)
            dx -= acc_map_info_.nx;

        int dy = ry;
        if (dy < 0)
            dy += acc_map_info_.ny;
        else if (dy >= acc_map_info_.ny)
            dy -= acc_map_info_.ny;

        return { ox + dx - half_x, oy + dy - half_y };
    }

    inline VoxelKey2D worldToKey2D(const Eigen::Vector2f& p) const {
        Eigen::Vector2f q = p / acc_map_info_.voxel_size;
        return { static_cast<int>(std::floor(q.x())), static_cast<int>(std::floor(q.y())) };
    }

    inline Eigen::Vector3f key2DToWorld(const VoxelKey2D& k) const {
        Eigen::Vector3f p(static_cast<float>(k.x), static_cast<float>(k.y), 0.0f);
        p *= acc_map_info_.voxel_size;
        p.z() = getRoboBase().z();
        return p;
    }

    inline bool isPassableCached(int idx, const Eigen::Vector3f& robo_base) const {
        const Cell& c = occ_map_info_.grid[idx];
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
        if (static_map_info_.has_static_map) {
            int ix, iy;
            if (worldToImage(world_xy, ix, iy)) {
                // 在图像范围内 → 直接查 mask
                if (iy >= 0 && iy < static_map_info_.size.y() && ix >= 0 && ix < static_map_info_.size.x()) {
                    return static_map_info_.mask[iy * static_map_info_.size.x() + ix] == 0;
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
        Eigen::Vector2f p = pw - static_map_info_.origin;

        ix = static_cast<int>(std::floor(p.x() / static_map_info_.resolution));
        iy = static_cast<int>(std::floor(p.y() / static_map_info_.resolution));

        // ROS map: (0,0) 在左下 → OpenCV 在左上
        iy = static_map_info_.size.y() - 1 - iy;

        if (ix < 0 || iy < 0 || ix >= static_map_info_.size.x() || iy >= static_map_info_.size.y())
            return false;

        return true;
    }
    struct AccMapInfo {
        float voxel_size;
        Eigen::Vector2f origin, size;
        Eigen::Vector2f tmp_origin;
        int nx, ny;

        VoxelKey2D origin_key;
        VoxelKey2D min_key, max_key;
    } acc_map_info_;
    struct StaticMapInfo {
        bool has_static_map = false;
        float resolution = 0.0f;
        Eigen::Vector2f origin; // 左下角 (x,y)
        Eigen::Vector2i size;
        std::vector<uint8_t> mask; // 0 blocked, 1 free
    } static_map_info_;

    cv::Mat buf0_;
    cv::Mat buf1_;
    cv::Mat* curr_ { nullptr };
    Eigen::Vector3f robo_base_;
    
    
};

} // namespace rose_map
