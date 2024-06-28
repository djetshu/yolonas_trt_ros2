import rclpy
from yolo_py.yolo_trt_helper import YoloTRTNode

def main(args=None):
    rclpy.init(args=args)

    yolo_trt_node = YoloTRTNode()

    rclpy.spin(yolo_trt_node)

    yolo_trt_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
