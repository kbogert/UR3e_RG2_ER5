from collections import Counter
import rospy
import time
from sensor_msgs.msg import Image
import cv2
import numpy as np
import sys
from cv_bridge import CvBridge, CvBridgeError
bridge = CvBridge()
Depth_image = None
RGB_image = None

def callbackDepth(data):
	global Depth_image
	Depth_image = bridge.imgmsg_to_cv2(data, data.encoding)

def callbackRGB(data):
	global RGB_image
	RGB_image = bridge.imgmsg_to_cv2(data, data.encoding)

# print(cv_image)
# print(image.encoding)
# (rows, cols, channels) = cv_image.shape
# new_image = cv_image
# return cv_image
# cv2.waitKey(3	while(1):)
#cv2.meanStdDev

if __name__ == '__main__':

	rospy.init_node('camera_node')
	rospy.Subscriber('/kinect2/hd/image_color', Image, callbackRGB)
	#rospy.Subscriber('/kinect2/qhd/image_color', Image, callbackRGB)
	rospy.Subscriber('/kinect2/sd/image_depth', Image, callbackDepth)
	rate = 100
	r = rospy.Rate(rate)
	while RGB_image is None:
		r.sleep()
	while Depth_image is None:
		r.sleep()
	#first_frame = cv2.imread('background_front.jpg',0)
	while True:
		cv2.imshow("Color Kinect", RGB_image)
		cv2.imshow("Depth Kinect", Depth_image)
		if cv2.waitKey(1) == 27:
			break  # esc to quit
	cv2.destroyAllWindows()
