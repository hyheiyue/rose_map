from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    rose_map_node = Node(
        package="rose_map",
        executable="rose_map_node",
        name="rose_map",
        output="screen",
        parameters=[
            PathJoinSubstitution(
                [
                    FindPackageShare("rose_map"),
                    "config",
                    "rose_map.yaml",
                ]
            )
        ],
    )

    return LaunchDescription([rose_map_node])
