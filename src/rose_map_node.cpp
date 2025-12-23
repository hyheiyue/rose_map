#include "rclcpp/rclcpp.hpp"
#include "rose_map.hpp"
namespace rose_map {
class RoseMapNode: public rclcpp::Node {
public:
    RoseMapNode(const rclcpp::NodeOptions& options): Node("rose_map_node", options) {
        rose_map_ = std::make_shared<RoseMap>(*this);
    }
    RoseMap::Ptr rose_map_;
};

}; // namespace rose_map

#include "rclcpp_components/register_node_macro.hpp"

// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable when its library
// is being loaded into a running process.
RCLCPP_COMPONENTS_REGISTER_NODE(rose_map::RoseMapNode)
