#! /usr/bin/env python
import time
import rospy
from std_msgs.msg import String, Float64



rospy.init_node('gripper_action_command_sender', anonymous=True)
pub = rospy.Publisher('/gripper_controller/command', Float64,queue_size=1)


def width(set_width):
    pub.publish(set_width)


width(30)
rospy.sleep(3)
width(100)
rospy.sleep(3)
width(10)
