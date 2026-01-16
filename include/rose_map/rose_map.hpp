#pragma once
#include "esdf.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/node.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
namespace rose_map {

class RoseMap: public ESDF {
public:
    using Ptr = std::shared_ptr<RoseMap>;
    RoseMap(rclcpp::Node& node);
    ~RoseMap();
    static Ptr create(rclcpp::Node& node) {
        return std::make_shared<RoseMap>(node);
    }
    struct Frame {
        double time;
        std::vector<Eigen::Vector3f> pts;
        Eigen::Vector3f sensor_origin;
    };
    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
    Eigen::Matrix4f tf2ToEigen(const geometry_msgs::msg::TransformStamped& tf) {
        Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
        const auto& t = tf.transform.translation;
        const auto& q = tf.transform.rotation;

        Eigen::Quaternionf Q(q.w, q.x, q.y, q.z);
        T.block<3, 3>(0, 0) = Q.toRotationMatrix();
        T(0, 3) = t.x;
        T(1, 3) = t.y;
        T(2, 3) = t.z;
        return T;
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
    template<typename T>
    bool publisherSubscribed(rclcpp::Publisher<T>::SharedPtr publisher) {
        return publisher->get_subscription_count() > 0;
    }
    void pubPointcloud(
        const std::vector<Eigen::Vector4f> all,
        sensor_msgs::msg::PointCloud2& msg,
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher
    );
    void handleinsertPointCloud(const Frame& frame);
    void handleUpdate();
    void updateThread();
    void insertPointCloudThread();
    void printStats();

    int insert_count_ = 0;
    int update_count_ = 0;
    double insert_cost_ms_ = 0.0;
    double update_cost_ms_ = 0.0;
    int processed_pts_ = 0;

    bool run_flag_ = true;
    std::thread update_thread;
    std::thread insert_thread;
    std::deque<Frame> frames_;
    std::mutex frames_mutex_;
    std::condition_variable frames_cv_;
    std::mutex map_mutex_;
    double callback_cost_accum_ms_ = 0.0;
    size_t callback_count_ = 0;
    double max_update_dt_ = 0.1;
    bool log_time_ = false;
    std::string sensor_frame_;
    std_msgs::msg::Header last_header_;
    rclcpp::Node* node_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;
    std::string target_frame_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr occ_map_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr acc_map_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr esdf_map_pub_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr grid_map_pub_;
    Clock current_time_ = 0.0;
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_ { tf_buffer_ };
};

} // namespace rose_map