#pragma once

#include <rclcpp/node.hpp>
namespace rose_map {
class Parameters {
public:
    void load(rclcpp::Node& node) {
        occ_map_params.load(node);
        acc_map_params.load(node);
        esdf_params.load(node);
    }
    struct OccMapParams {
        float log_hit = 0.85f;
        float log_free = -0.4f;
        float log_min = -5.f;
        float log_max = 5.f;
        float occ_th = 1.2f;
        int min_shift = 1;
        double timeout = 0.8;
        float max_ray_range = 20.0;
        bool unknown_is_occupied = false;
        bool use_ray = true;
        void load(rclcpp::Node& node) {
            log_hit = node.declare_parameter<float>("rose_map.occ_map.log_hit", log_hit);
            log_free = node.declare_parameter<float>("rose_map.occ_map.log_free", log_free);
            log_min = node.declare_parameter<float>("rose_map.occ_map.log_min", log_min);
            log_max = node.declare_parameter<float>("rose_map.occ_map.log_max", log_max);
            occ_th = node.declare_parameter<float>("rose_map.occ_map.occ_th", occ_th);
            min_shift = node.declare_parameter<int>("rose_map.occ_map.min_shift", min_shift);
            timeout = node.declare_parameter<double>("rose_map.occ_map.timeout", timeout);
            max_ray_range =
                node.declare_parameter<float>("rose_map.occ_map.max_ray_range", max_ray_range);
            unknown_is_occupied = node.declare_parameter<bool>(
                "rose_map.occ_map.unknown_is_occupied",
                unknown_is_occupied
            );
            use_ray = node.declare_parameter<bool>("rose_map.occ_map.use_ray", use_ray);
        }
    } occ_map_params;
    struct AccMapParams {
        float origin2base { 0.0f };
        float min_diff_z { 0.0f };

        int min_block_count { 1 };
        float block_ratio { 0.4f };
        void load(rclcpp::Node& node) {
            origin2base =
                node.declare_parameter<float>("rose_map.acc_map.origin2base", origin2base);
            min_diff_z = node.declare_parameter<float>("rose_map.acc_map.min_diff_z", min_diff_z);
            min_block_count =
                node.declare_parameter<int>("rose_map.acc_map.min_block_count", min_block_count);
            block_ratio =
                node.declare_parameter<float>("rose_map.acc_map.block_ratio", block_ratio);
        }
    } acc_map_params;

    struct EsdfParams {
        bool use_diagonals { false };
        void load(rclcpp::Node& node) {
            use_diagonals =
                node.declare_parameter<bool>("rose_map.esdf.use_diagonals", use_diagonals);
        }
    } esdf_params;
};
} // namespace rose_map