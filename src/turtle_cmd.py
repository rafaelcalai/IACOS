#!/usr/bin/env python
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
PI = 3.1415926535897

class TurtleCmdPublisher(Node):

    def __init__(self):
        super().__init__('turtle_cmd_publisher')
        self.publisher_ = self.create_publisher(Twist, '/turtle1/cmd_vel', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        vel_msg = Twist()

        # Receiveing the user's input
        print("Let's control our turtle")
        print("Type L to turn left")
        print("Type R to turn right")
        print("Type F to go foward")
        cmd = input("command: ")

        #Converting from angles to radians
        relative_angle = 90*2*PI/360 # 90 degrees
    
        #Initialize variables
        vel_msg.linear.x=0.0
        vel_msg.linear.y=0.0
        vel_msg.linear.z=0.0
        vel_msg.angular.x=0.0
        vel_msg.angular.y=0.0
        vel_msg.angular.z=0.0
    
        if cmd == "R":
            vel_msg.angular.z = -abs(relative_angle)
        elif cmd == "L": 
            vel_msg.angular.z = abs(relative_angle)
        elif cmd == "F":
            vel_msg.linear.x=1.0

        self.publisher_.publish(vel_msg)
        self.get_logger().info('Publishing: "%s"' % vel_msg)
        self.i += 1


def cmd(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node('turtle_commander')
    node.get_logger().info('Created node')

    turtle_cmd_publisher = TurtleCmdPublisher()
    
    rclpy.spin(turtle_cmd_publisher)

    # Destroy the node explicitly
    turtle_cmd_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
      cmd()

