#pragma once
#include "esdf.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/node.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
namespace rose_map {

class RoseMap: public ESDF {
public:
    using Ptr = std::shared_ptr<RoseMap>;
    explicit RoseMap(rclcpp::Node& node): tf_buffer_(node.get_clock()), ESDF(node) {
        node_ = &node;
        RCLCPP_INFO_STREAM(node.get_logger(), "[RoseMap] Initializing...");
        sensor_frame_ = node.declare_parameter<std::string>("rose_map.sensor_frame", "");
        log_time_ = node.declare_parameter<bool>("rose_map.log_time", false);
        int max_update_rate = node.declare_parameter<int>("rose_map.max_update_rate", 10);
        max_update_dt_ = 1.0 / max_update_rate;
        std::string pointcloud_topic =
            node.declare_parameter<std::string>("rose_map.pointcloud_topic", "");
        pointcloud_sub_ = node.create_subscription<sensor_msgs::msg::PointCloud2>(
            pointcloud_topic,
            rclcpp::SensorDataQoS(),
            std::bind(&RoseMap::pointCloudCallback, this, std::placeholders::_1)
        );

        occ_map_pub_ =
            node.create_publisher<sensor_msgs::msg::PointCloud2>("occ_map_out", rclcpp::QoS(10));
        acc_map_pub_ =
            node.create_publisher<sensor_msgs::msg::PointCloud2>("acc_map_out", rclcpp::QoS(10));
        esdf_map_pub_ =
            node.create_publisher<sensor_msgs::msg::PointCloud2>("esdf_out", rclcpp::QoS(10));
    }
    static Ptr create(rclcpp::Node& node) {
        return std::make_shared<RoseMap>(node);
    }
    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        const double ros_time = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;

        static double t_init = -1.0;
        if (t_init < 0.0)
            t_init = ros_time;

        current_time_ = static_cast<Clock>(ros_time - t_init);
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
            sensor_origin = lookupTF(msg->header.frame_id, sensor_frame_, msg->header.stamp);
        } catch (...) {
            RCLCPP_WARN(
                node_->get_logger(),
                "[OccMap] TF lookup failed (%s)",
                sensor_frame_.c_str()
            );
            sensor_origin = origin();
        }

        insertPointCloud(pts, sensor_origin, current_time_);

        static auto last_map_update_tp = std::chrono::steady_clock::now();

        const auto now_sys = std::chrono::steady_clock::now();

        if (std::chrono::duration<double>(now_sys - last_map_update_tp).count() >= max_update_dt_) {
            update(current_time_);
            last_map_update_tp = now_sys;
            if (publisherSubscribed(occ_map_pub_)) {
                sensor_msgs::msg::PointCloud2 occ_msg;
                occ_msg.header = msg->header;
                auto cloud = OccMap::getOccupiedPoints();
                pubPointcloud(cloud, occ_msg, occ_map_pub_);
            }

            if (publisherSubscribed(acc_map_pub_)) {
                sensor_msgs::msg::PointCloud2 acc_msg;
                acc_msg.header = msg->header;
                auto cloud = AccMap::getOccupiedPoints();
                pubPointcloud(cloud, acc_msg, acc_map_pub_);
            }

            if (publisherSubscribed(esdf_map_pub_)) {
                sensor_msgs::msg::PointCloud2 map_msg;
                map_msg.header = msg->header;
                auto cloud = ESDF::getOccupiedPoints();
                pubPointcloud(cloud, map_msg, esdf_map_pub_);
            }
        }

        const auto t1 = std::chrono::steady_clock::now();
        const double cost_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        callback_cost_accum_ms_ += cost_ms;
        callback_count_++;

        if (last_report_tp_.time_since_epoch().count() == 0) {
            last_report_tp_ = now_sys;
        }

        if (std::chrono::duration<double>(now_sys - last_report_tp_).count() >= 1.0 && log_time_) {
            RCLCPP_INFO(
                node_->get_logger(),
                "[RoseMap] %.2f ms/s, avg %.3f ms (%zu calls)",
                callback_cost_accum_ms_,
                callback_cost_accum_ms_ / std::max<size_t>(1, callback_count_),
                callback_count_
            );
            callback_cost_accum_ms_ = 0.0;
            callback_count_ = 0;
            last_report_tp_ = now_sys;
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

    bool publisherSubscribed(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher
    ) {
        return publisher->get_subscription_count() > 0;
    }
    void pubPointcloud(
        const std::vector<Eigen::Vector4f> all,
        sensor_msgs::msg::PointCloud2& msg,
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher
    ) {
        if (!publisherSubscribed(publisher)) {
            return;
        }
        msg.height = 1;
        msg.width = all.size();
        msg.fields.resize(4);
        msg.fields[0].name = "x";
        msg.fields[0].offset = 0;
        msg.fields[0].datatype = sensor_msgs::msg::PointField::FLOAT32;
        msg.fields[0].count = 1;
        msg.fields[1].name = "y";
        msg.fields[1].offset = 4;
        msg.fields[1].datatype = sensor_msgs::msg::PointField::FLOAT32;
        msg.fields[1].count = 1;
        msg.fields[2].name = "z";
        msg.fields[2].offset = 8;
        msg.fields[2].datatype = sensor_msgs::msg::PointField::FLOAT32;
        msg.fields[2].count = 1;
        msg.fields[3].name = "intensity";
        msg.fields[3].offset = 12;
        msg.fields[3].datatype = sensor_msgs::msg::PointField::FLOAT32;
        msg.fields[3].count = 1;
        msg.is_bigendian = false;
        msg.point_step = 16;
        msg.row_step = msg.point_step * msg.width;
        msg.data.resize(msg.row_step * msg.height);
        for (size_t i = 0; i < all.size(); ++i) {
            float* data_ptr = reinterpret_cast<float*>(&msg.data[i * msg.point_step]);
            data_ptr[0] = all[i][0];
            data_ptr[1] = all[i][1];
            data_ptr[2] = all[i][2];
            data_ptr[3] = all[i][3];
        }
        publisher->publish(msg);
    }
    double callback_cost_accum_ms_ = 0.0;
    size_t callback_count_ = 0;
    double max_update_dt_ = 0.1;
    bool log_time_ = false;
    std::string sensor_frame_;
    std::chrono::steady_clock::time_point last_report_tp_;
    rclcpp::Node* node_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr occ_map_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr acc_map_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr esdf_map_pub_;
    Clock current_time_ = 0.0;
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_ { tf_buffer_ };
};

} // namespace rose_map
