cmake_minimum_required(VERSION 3.8)
project(rose_map)


set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_COLOR_DIAGNOSTICS ON)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON) 
add_compile_options(
    -O3              
    -march=native     
    -mtune=native
    -ffast-math       
    -funroll-loops  
)
add_link_options(-O3 -flto)  
set(CMAKE_BUILD_TYPE "Release")

find_package(ament_cmake_auto REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(TBB REQUIRED)


file(GLOB_RECURSE ROSE_MAP_SOURCES
  src/*.cpp
  3rdparty/*.c
  3rdparty/*.cpp
)


ament_auto_add_library(${PROJECT_NAME} SHARED
  ${ROSE_MAP_SOURCES}
)


target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(${PROJECT_NAME}
  Eigen3::Eigen
  yaml-cpp
  TBB::tbb
  ${OpenCV_LIBS}
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


install(
  DIRECTORY include/
  DESTINATION include
)


ament_export_include_directories(include

)

ament_export_dependencies(
  rclcpp
  rclcpp_components
  sensor_msgs
  geometry_msgs
  nav_msgs
  tf2_ros
  tf2_geometry_msgs
  Eigen3
  yaml-cpp
  TBB
)


ament_auto_package(
  INSTALL_TO_SHARE
  config
  launch
)
