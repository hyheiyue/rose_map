#include "rose_map/acc_map.hpp"
#include "opencv2/opencv.hpp"
namespace rose_map {
AccMap::AccMap(rclcpp::Node& node): OccMap(node) {
    acc_map_info_.voxel_size_ = params_.acc_map_params.voxel_size;

    acc_map_info_.size_ = params_.acc_map_params.size;
    acc_map_info_.origin_ = params_.acc_map_params.origin;

    Eigen::Vector2f half = acc_map_info_.size_ * 0.5f;
    acc_map_info_.min_key_ = worldToKey2D(acc_map_info_.origin_ - half);
    acc_map_info_.max_key_ = worldToKey2D(acc_map_info_.origin_ + half);

    acc_map_info_.nx_ = acc_map_info_.max_key_.x - acc_map_info_.min_key_.x + 1;
    acc_map_info_.ny_ = acc_map_info_.max_key_.y - acc_map_info_.min_key_.y + 1;
    if (params_.acc_map_params.use_static_map) {
        loadRosMapYaml(params_.acc_map_params.static_map_path);
    }
    if (has_image_map_) {
        Eigen::Vector2f center_world;
        center_world.x() = image_origin_.x() + image_width_ * image_resolution_ * 0.5f;
        center_world.y() = image_origin_.y() + image_height_ * image_resolution_ * 0.5f;

        acc_map_info_.origin_ = center_world;

        Eigen::Vector2f world_size_img;
        world_size_img.x() = image_width_ * image_resolution_ + 1;
        world_size_img.y() = image_height_ * image_resolution_ + 1;
        acc_map_info_.size_ = world_size_img;
        Eigen::Vector2f half_img = world_size_img * 0.5f;
        acc_map_info_.min_key_ = worldToKey2D(acc_map_info_.origin_ - half_img);
        acc_map_info_.max_key_ = worldToKey2D(acc_map_info_.origin_ + half_img);

        acc_map_info_.nx_ = acc_map_info_.max_key_.x - acc_map_info_.min_key_.x + 1;
        acc_map_info_.ny_ = acc_map_info_.max_key_.y - acc_map_info_.min_key_.y + 1;
        acc_map_info_.origin_key_ = worldToKey2D(acc_map_info_.origin_);
    }

    const int size2d = acc_map_info_.nx_ * acc_map_info_.ny_;
    buf0_ = cv::Mat(1, size2d, CV_8U, cv::Scalar(0));
    buf1_ = cv::Mat(1, size2d, CV_8U, cv::Scalar(0));
    curr_ = &buf0_;
}

void AccMap::update(Clock now) {
    OccMap::update(now);
    updateRoboBase();
    if (!has_image_map_) {
        VoxelKey2D new_origin = worldToKey2D(acc_map_info_.tmp_origin_);
        acc_map_info_.origin_ = acc_map_info_.tmp_origin_;
        acc_map_info_.origin_key_ = new_origin;
        Eigen::Vector2f half = acc_map_info_.size_ * 0.5f;
        acc_map_info_.min_key_ = worldToKey2D(acc_map_info_.origin_ - half);
        acc_map_info_.max_key_ = worldToKey2D(acc_map_info_.origin_ + half);
    }

    const int size2d = acc_map_info_.nx_ * acc_map_info_.ny_;
    cv::Mat* other = (curr_ == &buf0_) ? &buf1_ : &buf0_;
    other->setTo(1);

    std::vector<int> block_cnt(size2d, 0);
    const Eigen::Vector3f robo_base = getRoboBase();

    for (int idx3d: occupied_buffer_idx_) {
        if (!isOccupied(idx3d, now))
            continue;
        if (!isPassableCached(idx3d, robo_base)) {
            auto w3d = key3DToWorld(index3DToKey3D(idx3d));
            int idx2d = key2DToIndex2D(worldToKey2D({ w3d.x(), w3d.y() }));
            if (idx2d >= 0)
                block_cnt[idx2d]++;
        }
    }

    for (int i = 0; i < size2d; ++i) {
        bool blocked = false;
        if (block_cnt[i] >= params_.acc_map_params.min_block_count)
            blocked = true;
        if (params_.acc_map_params.block_ratio > 0.0f) {
            float ratio = float(block_cnt[i])
                / float(occ_map_info_.nz_ * (acc_map_info_.voxel_size_ / occ_map_info_.voxel_size_)
                );
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
        other->at<uint8_t>(0, i) = blocked ? 1 : 0;
    }

    cv::Mat map2d(acc_map_info_.ny_, acc_map_info_.nx_, CV_8UC1, other->data);

    applyMorphology(map2d);

    // cv::imshow("mask", map2d * 255);
    // cv::waitKey(1);

    curr_ = other;
}

void AccMap::applyMorphology(cv::Mat& img) {
    static cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_RECT,
        cv::Size(params_.acc_map_params.kernel_size, params_.acc_map_params.kernel_size)
    );

    cv::medianBlur(img, img, 1);

    cv::morphologyEx(img, img, cv::MORPH_CLOSE, kernel);
    cv::dilate(img, img, kernel, cv::Point(-1, -1), params_.acc_map_params.dilate_iter);
    cv::threshold(img, img, 0, 1, cv::THRESH_BINARY);
}

bool AccMap::loadRosMapYaml(const std::string& yaml_path) {
    YAML::Node yaml;

    try {
        yaml = YAML::LoadFile(yaml_path);
    } catch (const std::exception& e) {
        std::cerr << "[AccMap] Failed to load yaml: " << yaml_path << " , reason: " << e.what()
                  << std::endl;
        return false;
    }

    if (!yaml || !yaml.IsMap()) {
        std::cerr << "[AccMap] Invalid yaml format: " << yaml_path << std::endl;
        return false;
    }

    if (!yaml["image"] || !yaml["resolution"] || !yaml["origin"]) {
        std::cerr << "[AccMap] Missing required fields in yaml: " << yaml_path << std::endl;
        return false;
    }

    std::string image_path;
    float resolution = 0.f;
    Eigen::Vector2f origin_xy;

    try {
        image_path = yaml["image"].as<std::string>();
        resolution = yaml["resolution"].as<float>();

        const auto& origin = yaml["origin"];
        if (!origin.IsSequence() || origin.size() < 2) {
            std::cerr << "[AccMap] Invalid origin field in yaml: " << yaml_path << std::endl;
            return false;
        }
        origin_xy << origin[0].as<float>(), origin[1].as<float>();
    } catch (const std::exception& e) {
        std::cerr << "[AccMap] YAML field parse error: " << e.what() << std::endl;
        return false;
    }

    if (image_path.empty()) {
        std::cerr << "[AccMap] Empty image path in yaml: " << yaml_path << std::endl;
        return false;
    }

    if (image_path[0] != '/') {
        const auto pos = yaml_path.find_last_of("/\\");
        if (pos == std::string::npos) {
            image_path = image_path;
        } else {
            image_path = yaml_path.substr(0, pos + 1) + image_path;
        }
    }

    cv::Mat img = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "[AccMap] Failed to load map image: " << image_path << std::endl;
        return false;
    }

    if (resolution <= 0.f) {
        std::cerr << "[AccMap] Invalid resolution: " << resolution << std::endl;
        return false;
    }

    image_width_ = img.cols;
    image_height_ = img.rows;
    image_resolution_ = resolution;
    image_origin_ = origin_xy;

    image_mask_.assign(image_width_ * image_height_, 0);

    const bool negate = yaml["negate"].as<int>(0) != 0;
    const float free_thresh = yaml["free_thresh"].as<float>(0.196f);
    const float occ_thresh = yaml["occupied_thresh"].as<float>(0.65f);

    for (int y = 0; y < image_height_; ++y) {
        const uint8_t* row_ptr = img.ptr<uint8_t>(y);
        for (int x = 0; x < image_width_; ++x) {
            uint8_t v = row_ptr[x];
            if (negate)
                v = static_cast<uint8_t>(255 - v);

            const float occ = (255.f - v) / 255.f;
            const bool free = occ <= free_thresh;

            image_mask_[y * image_width_ + x] = free ? 1 : 0;
        }
    }

    has_image_map_ = true;
    return true;
}

std::vector<Eigen::Vector4f> AccMap::getOccupiedPoints() const {
    std::vector<Eigen::Vector4f> pts;
    pts.reserve(acc_map_info_.nx_ * acc_map_info_.ny_ / 8 + 16);

    const int N = curr_->total(); // 取 Mat 元素总数

    for (int i = 0; i < N; ++i) {
        if (curr_->at<uint8_t>(0, i) == 1) { // blocked cell
            VoxelKey2D key2d = index2DToKey2D(i);
            Eigen::Vector3f p = key2DToWorld(key2d);
            pts.emplace_back(p.x(), p.y(), p.z() + 1.0f, 0.0f);
        }
    }

    Eigen::Vector3f robo_base = getRoboBase();
    pts.emplace_back(robo_base.x(), robo_base.y(), robo_base.z(), 1.0f);
    return pts;
}

} // namespace rose_map
