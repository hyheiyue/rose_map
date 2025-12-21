#pragma once

#include "occ_map.hpp"

#include <vector>
#include <cstdint>
#include <algorithm>
#include <cmath>

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
        // Convert world-aligned key to local grid coordinates
        int dx = k.x - origin_key_.x + nx_ / 2;
        int dy = k.y - origin_key_.y + ny_ / 2;

        // Boundary check
        if ((unsigned)dx >= (unsigned)nx_ || (unsigned)dy >= (unsigned)ny_) {
            return -1;
        }

        // Ring buffer remapping
        int rx = (dx + ox_) % nx_;
        int ry = (dy + oy_) % ny_;

        // Row-major layout
        return rx + ry * nx_;
    }

    inline VoxelKey2D index2DToKey2D(int idx) const {
        int ry = idx / nx_;
        int rx = idx % nx_;

        int dx = (rx - ox_ + nx_) % nx_;
        int dy = (ry - oy_ + ny_) % ny_;

        return { origin_key_.x + dx - nx_ / 2, origin_key_.y + dy - ny_ / 2 };
    }

    // World (x,y) -> 2D voxel key
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

    void trackPassable(bool was_passable, bool now_passable, int idx) {
        if (was_passable && !now_passable) {
            upper_idx_.push_back(idx);
        }
        if (!was_passable && now_passable) {
            lower_idx_.push_back(idx);
        }
    }


    void update(Clock now) {
        OccMap::update(now);

        const int size2d = nx_ * ny_;


        std::vector<uint8_t>* other = (curr_ == &buf0_) ? &buf1_ : &buf0_;

        std::fill(other->begin(), other->end(), static_cast<uint8_t>(1));


        const Eigen::Vector3f robo_base = getRoboBase();
        const float occ_th = OccMap::params_.occ_th;

        for (int idx3d : active_idx_) {
            VoxelKey3D k3 = index3DToKey3D(idx3d);
            VoxelKey2D k2{ k3.x, k3.y };
            int idx2d = key2DToIndex2D(k2);
            if (idx2d < 0) continue;

            (*other)[idx2d] = isPassableCached(idx3d, robo_base, occ_th) ? 1 : 0;
        }


        upper_idx_.clear();
        lower_idx_.clear();
        upper_idx_.reserve(upper_idx_.capacity());
        lower_idx_.reserve(lower_idx_.capacity());

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

    inline bool isPassableCached(int idx, const Eigen::Vector3f& robo_base, float occ_th) const {
        const Cell& c = grid_[idx];

        if (!c.active) return true;
        if (c.log_odds <= occ_th) return true;


        Eigen::Vector3f p = key3DToWorld(index3DToKey3D(idx));
        Eigen::Vector3f diff = p - robo_base;

        if (diff.z() < params_.min_diff_z) return true;

        return false;
    }

    std::vector<Eigen::Vector4f> getOccupiedPoints() const {
        std::vector<Eigen::Vector4f> pts;
        pts.reserve(nx_ * ny_ / 8 + 16);

        // 遍历 in index order
        for (int i = 0; i < static_cast<int>(curr_->size()); ++i) {
            if ((*curr_)[i] == 0) { // 0 => blocked / occupied
                VoxelKey2D key2d = index2DToKey2D(i);
                Eigen::Vector3f p = key2DToWorld(key2d);
                pts.emplace_back(p.x(), p.y(), p.z(), 0.0f);
            }
        }

        Eigen::Vector3f robo_base = getRoboBase();
        pts.emplace_back(robo_base.x(), robo_base.y(), robo_base.z(), 0.0f);

        return pts;
    }

    void updateEnd() {
        OccMap::updateEnd();
        upper_idx_.clear();
        lower_idx_.clear();
    }

    struct Params {
        float origin2base { 0.0f };
        float min_diff_z { 0.0f };

        void load(const YAML::Node& config) {
            origin2base = config["origin2base"].as<float>();
            min_diff_z = config["min_diff_z"].as<float>();
        }
    } params_;


    const std::vector<uint8_t>& acc_grid_view() const { return *curr_; }

    const std::vector<uint8_t>& last_acc_grid_view() const {
        return (curr_ == &buf0_) ? buf1_ : buf0_;
    }

private:
    // 双缓冲真实数据存储
    std::vector<uint8_t> buf0_;
    std::vector<uint8_t> buf1_;
    // 指向当前帧有效缓冲区
    std::vector<uint8_t>* curr_ { nullptr };

    // 为兼容原接口，保留变化索引
    std::vector<int> upper_idx_;
    std::vector<int> lower_idx_;
};

} // namespace rose_map
