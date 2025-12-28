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
        const auto& odom = *msg;

        Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
        try {
            auto tf = rose_map_->tf_buffer_.lookupTransform(
                rose_map_->target_frame_,
                odom.header.frame_id,
                odom.header.stamp
            );
            T = rose_map_->tf2ToEigen(tf);
        } catch (...) {
            RCLCPP_WARN(this->get_logger(), "[OccMap] Odom TF transform failed → using identity");
        }

        Eigen::Vector4f p(
            odom.pose.pose.position.x,
            odom.pose.pose.position.y,
            odom.pose.pose.position.z,
            1.0f
        );
        p = T * p;

        rose_map_->setOrigin(Eigen::Vector3f(p.x(), p.y(), p.z()));
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
