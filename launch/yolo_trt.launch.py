from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.substitutions import LaunchConfiguration
import os
from ament_index_python import get_package_share_directory
from launch.conditions import IfCondition

def generate_launch_description():
    default_rviz = os.path.join(get_package_share_directory('yolo_py'),
                                'config','rviz', 'rviz.rviz')
    
    # Declare rviz_config argument
    rviz_config_argument = DeclareLaunchArgument("rviz_config", default_value = default_rviz,
                                          description="Full path to the RVIZ config file to use")

    # Declare debug_mode argument
    rviz_en_argument = DeclareLaunchArgument("rviz_en", default_value="true",
                                                description="Enable or disable rviz GUI")
    
    # Declare input_topic argument
    input_topic_argument = DeclareLaunchArgument("input_topic", default_value="/camera/image",
                                                 description="Topic for YOLO input images")

     # Declare model_name argument
    model_name_argument = DeclareLaunchArgument("model_name", default_value="yolo_nas_m.trt",
                                                description="Name of the YOLO TRT model to use")

    ld = LaunchDescription()
    
    yolo_node = Node(
                    package='yolo_py',
                    executable='yolo_trt',
                    name='yolo_trt',
                    output='log',
                    remappings=[
                        ('/image', LaunchConfiguration('input_topic')), 
                        ],
                    parameters=[{'model_name': LaunchConfiguration('model_name')}],
                )
    
    rviz_node = Node(
                    package="rviz2",
                    executable="rviz2",
                    name="rviz2",
                    output="log",
                    arguments=["-d", LaunchConfiguration("rviz_config")],
                    )
    
    # Arguments
    ld.add_action(rviz_config_argument)
    ld.add_action(rviz_en_argument)
    ld.add_action(input_topic_argument)
    ld.add_action(model_name_argument)

    # Nodes
    ld.add_action(yolo_node)

    #Conditionally add the rviz_node
    rviz_group = GroupAction(
        actions=[rviz_node],
        condition=IfCondition(LaunchConfiguration("rviz_en"))
    )
    ld.add_action(rviz_group)

    return ld
