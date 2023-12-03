#!/usr/bin/env python
import rclpy
from rclpy.node import Node
import math

from geometry_msgs.msg import Twist
PI = math.pi

class TurtleCmdPublisher(Node):

    def __init__(self, queue=None):
        super().__init__('turtle_cmd_publisher')
        self.publisher_ = self.create_publisher(Twist, '/turtle1/cmd_vel', 10)
        timer_period = 1  # seconds
        self.cmd = 'S'
        if queue:
            self.queue = queue
            self.timer = self.create_timer(timer_period, self.timer_callback2)
        else:
            self.timer = self.create_timer(timer_period, self.timer_callback)  

    def timer_callback(self):
        vel_msg = Twist()

        # Receiveing the user's input
        print("Let's control our turtle")
        print("Type L to turn left")
        print("Type R to turn right")
        print("Type F to go foward")
        print("Type S to stop")
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
    
        if cmd == "R" or cmd == "r":
            vel_msg.angular.z = -abs(relative_angle)
        elif cmd == "L" or cmd == "l":
            vel_msg.angular.z = abs(relative_angle)
        elif cmd == "S" or cmd == "s":
            vel_msg.linear.x=0.0
        else:    
            vel_msg.linear.x=0.5

        self.publisher_.publish(vel_msg)
        self.get_logger().info('Publishing: "%s"' % vel_msg)
        self.i += 1    

    def timer_callback2(self):
        vel_msg = Twist()


        if not self.queue.empty():
            self.cmd = str(self.queue.get())
            self.get_logger().info("Command received: %s" % self.cmd)
            

        #Converting from angles to radians
        relative_angle = 90*2*PI/360 # 90 degrees
    
        #Initialize variables
        vel_msg.linear.x=0.0
        vel_msg.linear.y=0.0
        vel_msg.linear.z=0.0
        vel_msg.angular.x=0.0
        vel_msg.angular.y=0.0
        vel_msg.angular.z=0.0


        if self.cmd == 'R' or self.cmd == 'r':
            vel_msg.angular.z = -abs(relative_angle)
            self.cmd = 'F'
        elif self.cmd == 'L' or self.cmd == 'l':
            vel_msg.angular.z = abs(relative_angle)
            self.cmd = 'F'
        elif self.cmd == 'S' or self.cmd == 's':
            vel_msg.linear.x=0.0
        else:    
            vel_msg.linear.x=0.5


        self.publisher_.publish(vel_msg)
        self.get_logger().info('Publishing command:"%s" "%s"' % (self.cmd, vel_msg))


def cmd(queue=None,args=None):
    rclpy.init(args=args)
    node = rclpy.create_node('turtle_commander')
    node.get_logger().info('Created node')

    turtle_cmd_publisher = TurtleCmdPublisher(queue)
    
    rclpy.spin(turtle_cmd_publisher)

    # Destroy the node explicitly
    turtle_cmd_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
      cmd()

