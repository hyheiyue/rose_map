#pragma once

#include <Eigen/Core>
#include <yaml-cpp/yaml.h>

namespace YAML {

template<>
struct convert<Eigen::Vector3f> {
    static Node encode(const Eigen::Vector3f& rhs) {
        Node node;
        node.push_back(rhs.x());
        node.push_back(rhs.y());
        node.push_back(rhs.z());
        return node;
    }

    static bool decode(const Node& node, Eigen::Vector3f& rhs) {
        if (!node.IsSequence() || node.size() != 3)
            return false;

        rhs.x() = node[0].as<float>();
        rhs.y() = node[1].as<float>();
        rhs.z() = node[2].as<float>();
        return true;
    }
};

} // namespace YAML
