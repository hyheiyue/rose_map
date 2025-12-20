#pragma once
#include "occ_map.hpp"
namespace rose_map {
class AccMap: public OccMap {
public:
    using Ptr = std::shared_ptr<AccMap>;
    explicit AccMap(const YAML::Node& config): OccMap(config["occ_map"]) {
        params_.load(config["acc_map"]);
    }
    static Ptr create(const YAML::Node& config) {
        return std::make_shared<AccMap>(config);
    }
    Eigen::Vector3f getRoboBase() const {
        return OccMap::origin() - Eigen::Vector3f(0, 0, params_.origin2base);
    }
    std::vector<Eigen::Vector4f> getOccupiedPoints() const {
        std::vector<Eigen::Vector4f> pts;
        pts.reserve(active_idx_.size());
        auto robo_base = getRoboBase();
        for (int idx: active_idx_) {
            if(isPassable(idx))
                continue;
            auto p = keyToWorld(indexToKey(idx));
            pts.emplace_back(p.x(), p.y(), p.z(), p.z() - robo_base.z());
        }
        pts.emplace_back(robo_base.x(), robo_base.y(), robo_base.z(), 0);
        return pts;
    }
    void update(Clock now) {
        OccMap::update(now);
    }
    bool isPassable(int idx) const{
        auto robo_base = getRoboBase();
        const Cell& c = grid_[idx];
        if (!c.active)
            return true;
        if (c.log_odds <= OccMap::params_.occ_th)
            return true;
        auto p = keyToWorld(indexToKey(idx));
        auto diff = p - robo_base;
        if (diff.z() < params_.min_diff_z)
            return true;
        return false;
    }
    void updateEnd() {
        OccMap::updateEnd();
    }
    struct Params {
        float origin2base = 0.0;
        float min_diff_z = 0.0;
        void load(const YAML::Node& config) {
            origin2base = config["origin2base"].as<float>();
            min_diff_z = config["min_diff_z"].as<float>();
        }
    } params_;
};
} // namespace rose_map