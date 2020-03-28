#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
import cv2
import numpy as np
import os
import sys
from cv_bridge import CvBridge, CvBridgeError

bridge = CvBridge()

cv_image = None

def callback(data):
	global cv_image

	cv_image = bridge.imgmsg_to_cv2(data, data.encoding)

if __name__ == '__main__':
	try: 
   		os.remove("background_front_no_shadow.jpg")
		print("Deleted old background_front_no_shadow.jpg")
	except:
		print("Old Image not found")
		pass
	rospy.init_node('camera_node')
	rospy.Subscriber('/rrbot/camera1/image_raw', Image, callback)
	print("Subscribed to image_raw...")
	rate = 100
	r = rospy.Rate(rate)
	print("Waiting for image_raw...")
	while cv_image is None:
		r.sleep()
	cv2.imwrite('background_front_no_shadow.jpg',cv_image)
	print("Image Updated at background_front_no_shadow.jpg")
