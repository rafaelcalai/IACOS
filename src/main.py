from threading import *
import time
import queue
from turtle_cmd import TurtleCmdPublisher, cmd

# ros2 run turtlesim turtlesim_node

items=[]
def produce(c):
   while True:
       item='S' #Step 1.2
       print("Producer Producing Item:", item)
       c.put(item)
       print("Producer giving Notification")
       time.sleep(2)

       item='F' #Step 1.2
       c.put(item)
       print("Producer giving Notification")
       time.sleep(2)
       item='L' #Step 1.2
       c.put(item)
       print("Producer giving Notification")
       time.sleep(2)
       item='L' #Step 1.2
       c.put(item)
       print("Producer giving Notification")
       time.sleep(2)
       item='R' #Step 1.2
       c.put(item)
       print("Producer giving Notification")
       time.sleep(2)

       item='S' #Step 1.2
       c.put(item)
       print("Producer giving Notification")


def consume(c, args=None):
    cmd(c)

def main():    
    q=queue.Queue()
    t1=Thread(target=consume, args=(q,))
    t2=Thread(target=produce, args=(q,))
    t1.start()
    t2.start()

if __name__ == '__main__':
      main()
