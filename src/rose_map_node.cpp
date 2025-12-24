#include "rclcpp/rclcpp.hpp"
#include "rose_map/rose_map.hpp"
namespace rose_map {
class RoseMapNode: public rclcpp::Node {
public:
    RoseMapNode(const rclcpp::NodeOptions& options): Node("rose_map_node", options) {
        rose_map_ = std::make_shared<RoseMap>(*this);
        std::string odom_topic = this->declare_parameter<std::string>("odom_topic", "");
        odometry_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            odom_topic,
            rclcpp::SensorDataQoS(),
            std::bind(&RoseMapNode::odomCallback, this, std::placeholders::_1)
        );
    }
    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        auto odom = *msg;
        rose_map_->setOrigin(Eigen::Vector3f(
            odom.pose.pose.position.x,
            odom.pose.pose.position.y,
            odom.pose.pose.position.z
        ));
    }
    RoseMap::Ptr rose_map_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odometry_sub_;
};

}; // namespace rose_map

#include "rclcpp_components/register_node_macro.hpp"

// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable when its library
// is being loaded into a running process.
RCLCPP_COMPONENTS_REGISTER_NODE(rose_map::RoseMapNode)
