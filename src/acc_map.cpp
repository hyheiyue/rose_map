#include "rose_map/acc_map.hpp"
#include "opencv2/opencv.hpp"
namespace rose_map {
AccMap::AccMap(rclcpp::Node& node): OccMap(node) {
    const int size2d = nx_ * ny_;
    buf0_ = cv::Mat(1, size2d, CV_8U, cv::Scalar(0));
    buf1_ = cv::Mat(1, size2d, CV_8U, cv::Scalar(0));
    curr_ = &buf0_;

    if (params_.acc_map_params.use_static_map) {
        loadRosMapYaml(params_.acc_map_params.static_map_path);
    }
}
// update 逻辑不变，仅存储改为 Mat
void AccMap::update(Clock now) {
    OccMap::update(now);

    const int size2d = nx_ * ny_;
    cv::Mat* other = (curr_ == &buf0_) ? &buf1_ : &buf0_;

    other->setTo(1);

    std::vector<int> block_cnt(size2d, 0);
    const Eigen::Vector3f robo_base = getRoboBase();

    for (int idx3d: occupied_buffer_idx_) {
        if (!isOccupied(idx3d, now))
            continue;

        VoxelKey3D k3 = index3DToKey3D(idx3d);
        int idx2d = key2DToIndex2D({ k3.x, k3.y });
        if (idx2d >= 0 && !isPassableCached(idx3d, robo_base)) {
            block_cnt[idx2d]++;
        }
    }

    for (int i = 0; i < size2d; ++i) {
        bool blocked = false;

        if (block_cnt[i] >= params_.acc_map_params.min_block_count)
            blocked = true;

        if (params_.acc_map_params.block_ratio > 0.0f) {
            float ratio = float(block_cnt[i]) / float(nz_);
            blocked = (ratio >= params_.acc_map_params.block_ratio);
        }

        if (!blocked && has_image_map_) {
            auto key2d = index2DToKey2D(i);
            Eigen::Vector3f pw3 = key2DToWorld(key2d);
            Eigen::Vector2f pw(pw3.x(), pw3.y());

            int ix, iy;
            if (worldToImage(pw, ix, iy)) {
                if (image_mask_[iy * image_width_ + ix] == 0)
                    blocked = true;
            }
        }

        (*other).at<uint8_t>(0, i) = blocked ? 1 : 0;
    }

    cv::Mat map2d(ny_, nx_, CV_8UC1, other->data);

    cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_RECT,
        cv::Size(params_.acc_map_params.kernel_size, params_.acc_map_params.kernel_size)
    );

    cv::morphologyEx(map2d, map2d, cv::MORPH_CLOSE, kernel);

    cv::dilate(map2d, map2d, kernel, cv::Point(-1, -1), params_.acc_map_params.dilate_iter);

    cv::threshold(map2d, map2d, 0, 1, cv::THRESH_BINARY);

    // cv::imshow("mask", map2d * 255);
    // cv::waitKey(1);

    curr_ = other;
}

bool AccMap::loadRosMapYaml(const std::string& yaml_path) {
    YAML::Node yaml = YAML::LoadFile(yaml_path);
    std::string image_path = yaml["image"].as<std::string>();
    float resolution = yaml["resolution"].as<float>();

    auto origin = yaml["origin"];
    Eigen::Vector2f origin_xy(origin[0].as<float>(), origin[1].as<float>());

    std::string base_dir = yaml_path.substr(0, yaml_path.find_last_of("/"));
    if (image_path[0] != '/')
        image_path = base_dir + "/" + image_path;

    cv::Mat img = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    if (img.empty())
        return false;

    image_width_ = img.cols;
    image_height_ = img.rows;
    image_resolution_ = resolution;
    image_origin_ = origin_xy;

    image_mask_.resize(image_width_ * image_height_);

    for (int y = 0; y < image_height_; ++y) {
        for (int x = 0; x < image_width_; ++x) {
            uint8_t v = img.at<uint8_t>(y, x);
            if (yaml["negate"].as<int>(0))
                v = 255 - v;

            float occ = (255 - v) / 255.0f;
            bool free = occ <= yaml["free_thresh"].as<float>(0.196f);
            bool occu = occ >= yaml["occupied_thresh"].as<float>(0.65f);

            image_mask_[y * image_width_ + x] = free ? 1 : 0;
        }
    }

    has_image_map_ = true;
    return true;
}

std::vector<Eigen::Vector4f> AccMap::getOccupiedPoints() const {
    std::vector<Eigen::Vector4f> pts;
    pts.reserve(nx_ * ny_ / 8 + 16);

    const int N = curr_->total(); // 取 Mat 元素总数

    for (int i = 0; i < N; ++i) {
        if (curr_->at<uint8_t>(0, i) == 1) { // blocked cell
            VoxelKey2D key2d = index2DToKey2D(i);
            Eigen::Vector3f p = key2DToWorld(key2d);
            pts.emplace_back(p.x(), p.y(), p.z() + 1.0f, 0.0f);
        }
    }

    Eigen::Vector3f robo_base = getRoboBase();
    pts.emplace_back(robo_base.x(), robo_base.y(), robo_base.z(), 0.0f);

    return pts;
}

} // namespace rose_map
