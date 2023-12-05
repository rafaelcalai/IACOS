from threading import *
import time
import queue
from turtle_cmd import TurtleCmdPublisher, cmd
from action_recognition import GestureRecognition

# ros2 run turtlesim turtlesim_node
def produce(queue):
    gesture = GestureRecognition(queue)
    gesture.detect_video_device()
    gesture.detec_gesture()

def consume(queue, args=None):
    cmd(queue)

def main():    
    q=queue.Queue()
    t1=Thread(target=consume, args=(q,))
    t2=Thread(target=produce, args=(q,))
    t1.start()
    t2.start()

if __name__ == '__main__':
      main()
