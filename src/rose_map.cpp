#include "rose_map/rose_map.hpp"
namespace rose_map {
RoseMap::RoseMap(rclcpp::Node& node): tf_buffer_(node.get_clock()), ESDF(node) {
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
    target_frame_ = node.declare_parameter<std::string>("rose_map.target_frame", "odom");
    occ_map_pub_ =
        node.create_publisher<sensor_msgs::msg::PointCloud2>("occ_map_out", rclcpp::QoS(10));
    acc_map_pub_ =
        node.create_publisher<sensor_msgs::msg::PointCloud2>("acc_map_out", rclcpp::QoS(10));
    esdf_map_pub_ =
        node.create_publisher<sensor_msgs::msg::PointCloud2>("esdf_out", rclcpp::QoS(10));
    grid_map_pub_ = node.create_publisher<nav_msgs::msg::OccupancyGrid>("acc_grid", 10);
}
void RoseMap::pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    // 计算相对 ROS 起始时间
    const double ros_time = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
    static double t_init = -1.0;
    if (t_init < 0.0)
        t_init = ros_time;
    current_time_ = static_cast<Clock>(ros_time - t_init);
    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
    try {
        auto tf =
            tf_buffer_.lookupTransform(target_frame_, msg->header.frame_id, msg->header.stamp);
        T = tf2ToEigen(tf);
    } catch (...) {
        RCLCPP_WARN(node_->get_logger(), "[OccMap] TF transform failed → using identity");
    }
    const size_t size = msg->width * msg->height;
    std::vector<Eigen::Vector3f> pts;
    pts.reserve(size);

    sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
    sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
    sensor_msgs::PointCloud2ConstIterator<float> iter_z(*msg, "z");

    for (size_t i = 0; i < size; ++i) {
        Eigen::Vector4f p(*iter_x, *iter_y, *iter_z, 1.0f);
        p = T * p;

        pts.emplace_back(p.x(), p.y(), p.z());

        ++iter_x;
        ++iter_y;
        ++iter_z;
    }

    const auto t0 = std::chrono::steady_clock::now();

    Eigen::Vector3f sensor_origin;
    try {
        sensor_origin = lookupTF(target_frame_, sensor_frame_, msg->header.stamp);
    } catch (...) {
        RCLCPP_WARN(node_->get_logger(), "[OccMap] TF lookup failed (%s)", sensor_frame_.c_str());
        sensor_origin = origin();
    }
    insertPointCloud(pts, sensor_origin, current_time_);

    // 地图更新与发布逻辑
    static auto last_map_update_tp = std::chrono::steady_clock::now();
    const auto now_sys = std::chrono::steady_clock::now();

    if (std::chrono::duration<double>(now_sys - last_map_update_tp).count() >= max_update_dt_) {
        update(current_time_);
        last_map_update_tp = now_sys;

        if (publisherSubscribed<sensor_msgs::msg::PointCloud2>(occ_map_pub_)) {
            sensor_msgs::msg::PointCloud2 occ_msg;
            occ_msg.header.stamp = msg->header.stamp;
            occ_msg.header.frame_id = target_frame_;
            auto cloud = OccMap::getOccupiedPoints();
            pubPointcloud(cloud, occ_msg, occ_map_pub_);
        }

        if (publisherSubscribed<sensor_msgs::msg::PointCloud2>(acc_map_pub_)) {
            sensor_msgs::msg::PointCloud2 acc_msg;
            acc_msg.header.stamp = msg->header.stamp;
            acc_msg.header.frame_id = target_frame_;
            auto cloud = AccMap::getOccupiedPoints();
            pubPointcloud(cloud, acc_msg, acc_map_pub_);
        }

        if (publisherSubscribed<sensor_msgs::msg::PointCloud2>(esdf_map_pub_)) {
            sensor_msgs::msg::PointCloud2 esdf_msg;
            esdf_msg.header.stamp = msg->header.stamp;
            esdf_msg.header.frame_id = target_frame_;
            auto cloud = ESDF::getOccupiedPoints();
            pubPointcloud(cloud, esdf_msg, esdf_map_pub_);
        }
        if (publisherSubscribed<nav_msgs::msg::OccupancyGrid>(grid_map_pub_)) {
            const auto& grid = AccMap::acc_grid_view(); // cv::Mat (uint8 0/1)

            if (!grid.empty()) {
                nav_msgs::msg::OccupancyGrid msg;
                msg.header.stamp = node_->now();
                msg.header.frame_id = target_frame_;
                msg.info.width = acc_map_info_.nx_;
                msg.info.height = acc_map_info_.ny_;
                msg.info.resolution = acc_map_info_.voxel_size_;

                msg.info.origin.position.x =
                    acc_map_info_.origin_.x() - acc_map_info_.size_.x() / 2.0;
                msg.info.origin.position.y =
                    acc_map_info_.origin_.y() - acc_map_info_.size_.y() / 2.0;
                msg.info.origin.orientation.w = 1.0;

                const int size2d = acc_map_info_.nx_ * acc_map_info_.ny_;
                msg.data.assign(size2d, 0);
                std::vector<uint8_t> acc(size2d);
                std::memcpy(acc.data(), grid.data, size2d);

                for (int i = 0; i < size2d; ++i) {
                    bool is_occ = (acc[i] == 1);
                    msg.data[i] = is_occ ? 100 : 0;
                }

                grid_map_pub_->publish(msg);
            }
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
void RoseMap::pubPointcloud(
    const std::vector<Eigen::Vector4f> all,
    sensor_msgs::msg::PointCloud2& msg,
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher
) {
    if (!publisherSubscribed<sensor_msgs::msg::PointCloud2>(publisher)) {
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
} // namespace rose_map