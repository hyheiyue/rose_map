#include "acc_map.hpp"
namespace rose_map {
AccMap::AccMap(rclcpp::Node& node): OccMap(node) {
    const int size2d = nx_ * ny_;
    buf0_.assign(size2d, 1);
    buf1_.assign(size2d, 1);
    curr_ = &buf0_;

    upper_idx_.reserve(1024);
    lower_idx_.reserve(1024);
}
void AccMap::update(Clock now) {
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

        if (block_cnt[i] >= params_.acc_map_params.min_block_count)
            blocked = true;
        if (params_.acc_map_params.block_ratio > 0.0f) {
            float ratio = static_cast<float>(block_cnt[i]) / static_cast<float>(nz_);
            if (ratio >= params_.acc_map_params.block_ratio) {
                blocked = true;
            } else {
                blocked = false;
            }
        }

        (*other)[i] = blocked ? 0 : 1;
    }

    for (int i = 0; i < size2d; ++i) {
        bool was = ((*curr_)[i] != 0);
        bool nowp = ((*other)[i] != 0);

        if (was && !nowp)
            upper_idx_.push_back(i);
        else if (!was && nowp)
            lower_idx_.push_back(i);
    }

    curr_ = other;
}
std::vector<Eigen::Vector4f> AccMap::getOccupiedPoints() const {
    std::vector<Eigen::Vector4f> pts;
    pts.reserve(nx_ * ny_ / 8 + 16);

    for (int i = 0; i < static_cast<int>(curr_->size()); ++i) {
        if ((*curr_)[i] == 0) {
            VoxelKey2D key2d = index2DToKey2D(i);
            Eigen::Vector3f p = key2DToWorld(key2d);
            pts.emplace_back(p.x(), p.y(), p.z() + 1.0, 0.0f);
        }
    }

    Eigen::Vector3f robo_base = getRoboBase();
    pts.emplace_back(robo_base.x(), robo_base.y(), robo_base.z() + 1.0, 0.0f);

    return pts;
}
} // namespace rose_map
