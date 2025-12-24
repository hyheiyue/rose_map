#pragma once
#include "acc_map.hpp"

#include <cmath>

namespace rose_map {

class ESDF: public AccMap {
public:
    using Ptr = std::shared_ptr<ESDF>;

    explicit ESDF(rclcpp::Node& node);

    static Ptr create(rclcpp::Node& node) {
        return std::make_shared<ESDF>(node);
    }

    void update(Clock now);

    inline float getDistance(const VoxelKey2D& key) const {
        int idx = key2DToIndex2D(key);
        return (idx < 0) ? kInf : esdf_[idx];
    }
    std::vector<Eigen::Vector4f> getOccupiedPoints(float sample_resolution_m = 0.1f) const;

    static constexpr float kInf = 1e6f;

    struct QNode {
        VoxelKey2D key;
        int idx; // cached index for key
        float dist;
    };

    // comparator for min-heap (smaller dist -> higher priority)
    struct QComp {
        bool operator()(const QNode& a, const QNode& b) const {
            return a.dist > b.dist;
        }
    };

    std::vector<float> esdf_;
    std::vector<float> dist_to_occ_;
    std::vector<float> dist_to_free_;

    static constexpr int dx8_[8] = { 1, -1, 0, 0, 1, 1, -1, -1 };
    static constexpr int dy8_[8] = { 0, 0, 1, -1, 1, -1, 1, -1 };

    float step_cost_[2];

    inline bool isDiagonalIdx(int k) const {
        return k >= 4;
    }

    void propagateKeyDistanceFieldTwoPass(
        const std::vector<uint8_t>& acc,
        std::vector<float>& dist,
        bool source_is_occ
    );

    void rebuildSigned();
};

} // namespace rose_map
