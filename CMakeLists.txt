cmake_minimum_required(VERSION 3.8)
project(rose_map)

## use c++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# enable color output
set(CMAKE_COLOR_DIAGNOSTICS ON)

## export compile commands for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
file(GLOB_RECURSE SOURCE src/*.cpp 3rdparty/*.c 3rdparty/*.cpp)
# find dependencies
find_package(ament_cmake_auto REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(yaml-cpp REQUIRED)
ament_auto_add_library(${PROJECT_NAME} SHARED ${SOURCE})
target_include_directories(${PROJECT_NAME} PUBLIC
    include
    3rdparty
    src
    ${CMAKE_CURRENT_BINARY_DIR}/include
)

target_link_libraries(${PROJECT_NAME}
    Eigen3::Eigen
    yaml-cpp
)
ament_target_dependencies(${PROJECT_NAME}
    rclcpp
    rclcpp_components
    sensor_msgs
    geometry_msgs
    nav_msgs
    tf2_ros
    tf2_geometry_msgs
)
rclcpp_components_register_node(${PROJECT_NAME}
    PLUGIN rose_map::RoseMapNode
    EXECUTABLE ${PROJECT_NAME}_node
)
ament_auto_package(
    INSTALL_TO_SHARE
    config
    launch
)
