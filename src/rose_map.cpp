#include "acc_map.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
namespace rose_map {
class RoseMapNode: public rclcpp::Node {
public:
    RoseMapNode(const rclcpp::NodeOptions& options):
        Node("rose_map_node", options),
        tf_buffer_(this->get_clock()) {
        RCLCPP_INFO(this->get_logger(), "RoseMapNode has been started.");
        YAML::Node config = YAML::LoadFile("/home/hy/wust_nav2/src/rose_map/config/rose_map.yaml");
        acc_map_ = AccMap::create(config);

        pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/cloud_registered",
            rclcpp::SensorDataQoS(),
            std::bind(&RoseMapNode::pointCloudCallback, this, std::placeholders::_1)
        );
        odometry_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/Odometry",
            rclcpp::SensorDataQoS(),
            std::bind(&RoseMapNode::odomCallback, this, std::placeholders::_1)
        );
        occ_map_pub_ =
            this->create_publisher<sensor_msgs::msg::PointCloud2>("occ_map_out", rclcpp::QoS(10));
    }
    double occ_cost_accum_ = 0.0;
    size_t occ_call_count_ = 0;
    double last_report_time_ = 0.0;

    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        const double time = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;

        current_time_ = time;

        sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
        sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
        sensor_msgs::PointCloud2ConstIterator<float> iter_z(*msg, "z");

        const size_t size = msg->width * msg->height;
        std::vector<Eigen::Vector3f> pts;
        pts.reserve(size);

        for (size_t i = 0; i < size; ++i) {
            pts.emplace_back(*iter_x, *iter_y, *iter_z);
            ++iter_x;
            ++iter_y;
            ++iter_z;
        }

        const auto t0 = std::chrono::steady_clock::now();

        Eigen::Vector3f sensor_origin;
        try {
            sensor_origin = lookupTF(msg->header.frame_id, "front_mid360", msg->header.stamp);
        } catch (...) {
            RCLCPP_WARN(
                this->get_logger(),
                "[OccMap] TF lookup failed (%s)",
                msg->header.frame_id.c_str()
            );
            sensor_origin = acc_map_->origin();
        }

        acc_map_->insertPointCloud(pts, sensor_origin, current_time_);
        acc_map_->update(current_time_);
        acc_map_->updateEnd();
        sensor_msgs::msg::PointCloud2 occ_msg;
        occ_msg.header = msg->header;
        pubOccPointcloud(current_time_, occ_msg);
        const auto t1 = std::chrono::steady_clock::now();
        const double cost_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        occ_cost_accum_ += cost_ms;
        occ_call_count_++;

        if (last_report_time_ == 0.0)
            last_report_time_ = current_time_;

        if (current_time_ - last_report_time_ >= 1.0) {
            RCLCPP_INFO(
                this->get_logger(),
                "[OccMap] %.2f ms/s, avg %.3f ms (%zu calls)",
                occ_cost_accum_,
                occ_cost_accum_ / std::max<size_t>(1, occ_call_count_),
                occ_call_count_
            );

            occ_cost_accum_ = 0.0;
            occ_call_count_ = 0;
            last_report_time_ = current_time_;
        }
    }

    Eigen::Vector3f
    lookupTF(const std::string& parent, const std::string& child, const rclcpp::Time& stamp) {
        geometry_msgs::msg::TransformStamped tf;

        try {
            tf = tf_buffer_.lookupTransform(parent, child, stamp, tf2::durationFromSec(0.05));
        } catch (const tf2::TransformException& ex) {
            throw std::runtime_error(ex.what());
        }

        return Eigen::Vector3f(
            tf.transform.translation.x,
            tf.transform.translation.y,
            tf.transform.translation.z
        );
    }

    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        auto odom = *msg;
        acc_map_->setOrigin(Eigen::Vector3f(
            odom.pose.pose.position.x,
            odom.pose.pose.position.y,
            odom.pose.pose.position.z
        ));
    }
    void pubOccPointcloud(Clock time, sensor_msgs::msg::PointCloud2& occ_msg) {
        if (occ_map_pub_->get_subscription_count() < 1) {
            return;
        }
        auto all = acc_map_->getOccupiedPoints();
        occ_msg.height = 1;
        occ_msg.width = all.size();
        occ_msg.fields.resize(4);
        occ_msg.fields[0].name = "x";
        occ_msg.fields[0].offset = 0;
        occ_msg.fields[0].datatype = sensor_msgs::msg::PointField::FLOAT32;
        occ_msg.fields[0].count = 1;
        occ_msg.fields[1].name = "y";
        occ_msg.fields[1].offset = 4;
        occ_msg.fields[1].datatype = sensor_msgs::msg::PointField::FLOAT32;
        occ_msg.fields[1].count = 1;
        occ_msg.fields[2].name = "z";
        occ_msg.fields[2].offset = 8;
        occ_msg.fields[2].datatype = sensor_msgs::msg::PointField::FLOAT32;
        occ_msg.fields[2].count = 1;
        occ_msg.fields[3].name = "intensity";
        occ_msg.fields[3].offset = 12;
        occ_msg.fields[3].datatype = sensor_msgs::msg::PointField::FLOAT32;
        occ_msg.fields[3].count = 1;
        occ_msg.is_bigendian = false;
        occ_msg.point_step = 16;
        occ_msg.row_step = occ_msg.point_step * occ_msg.width;
        occ_msg.data.resize(occ_msg.row_step * occ_msg.height);
        for (size_t i = 0; i < all.size(); ++i) {
            float* data_ptr = reinterpret_cast<float*>(&occ_msg.data[i * occ_msg.point_step]);
            data_ptr[0] = all[i][0];
            data_ptr[1] = all[i][1];
            data_ptr[2] = all[i][2];
            data_ptr[3] = all[i][3];
        }
        occ_map_pub_->publish(occ_msg);
    }
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odometry_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr occ_map_pub_;
    AccMap::Ptr acc_map_;
    Clock current_time_ = 0.0;
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_ { tf_buffer_ };
};

}; // namespace rose_map

#include "rclcpp_components/register_node_macro.hpp"

// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable when its library
// is being loaded into a running process.
RCLCPP_COMPONENTS_REGISTER_NODE(rose_map::RoseMapNode)
