from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    param_file = PathJoinSubstitution(
        [FindPackageShare("rose_map"), "config", "rose_map.yaml"]
    )

    rose_map_node = Node(
        package="rose_map",
        executable="rose_map_node",
        name="rose_map",
        output="screen",
        arguments=["--ros-args", "--params-file", param_file],
        # 直接用 prefix 里执行 ros2 pkg prefix 并传入 gdb
        prefix='gnome-terminal -- bash -c "gdb -ex run --args $(ros2 pkg prefix rose_map)/lib/rose_map/rose_map_node --ros-args --params-file $(ros2 pkg prefix rose_map)/share/rose_map/config/rose_map.yaml; exec bash"',
    )

    return LaunchDescription([rose_map_node])
