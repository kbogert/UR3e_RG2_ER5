#!/usr/bin/env python

from sklearn.cluster import KMeans
from collections import Counter
import rospy
import time
from sensor_msgs.msg import Image
import cv2
import numpy as np
import sys
from cv_bridge import CvBridge, CvBridgeError
new_image = np. ones(shape=[250, 250, 3], dtype=np. uint8)
bridge = CvBridge()

cv_image = None


def get_average_color(image,image_processing_size=None):
	if image_processing_size is not None:
		image = cv2.resize(image, image_processing_size, interpolation=cv2.INTER_AREA)
	rows,cols,ch = image.shape
	pixlist = []
	for y in range(0,rows,4):
		for x in range(0,cols,4):
			if (np.any(image[x, y] != 0)):
				pixlist.append(image[x,y])
	npa = np.asarray(pixlist, dtype=np.float32)
	return cv2.mean(npa)

def get_dominant_color(image, k=10, image_processing_size=None):

	if image_processing_size is not None:
		image = cv2.resize(image, image_processing_size, interpolation=cv2.INTER_AREA)
	rows,cols,ch = image.shape
	pixlist = []
	count = 0
	
	for y in range(0,rows,1):
		for x in range(0,cols,1):
			if (np.any(image[x, y] != 0)):
				count = count + 1
				pixlist.append(image[x,y])
	if(count == 0):
		return list([0,0,0])
	# reshape the image to be a list of pixels
	#image = image.reshape((image.shape[0] * image.shape[1], 3))
	# cluster and assign labels to the pixels
	clt = KMeans(n_clusters=k)
	labels = clt.fit_predict(pixlist)

	# count labels to find most popular
	label_counts = Counter(labels)
	# subset out most popular centroid
	dominant_color = clt.cluster_centers_[label_counts.most_common(1)[0][0]]
	print(len(pixlist))
	return list(dominant_color)


def callback(data):
	global cv_image

	cv_image = bridge.imgmsg_to_cv2(data, data.encoding)

# print(cv_image)
# print(image.encoding)
# (rows, cols, channels) = cv_image.shape
# new_image = cv_image
# return cv_image
# cv2.waitKey(3	while(1):)
#cv2.meanStdDev

if __name__ == '__main__':

	rospy.init_node('camera_node')
	rospy.Subscriber('/rrbot/camera1/image_raw', Image, callback)
	rate = 100
	r = rospy.Rate(rate)
	while cv_image is None:
		r.sleep()
	#first_frame = cv2.imread('background_front.jpg',0)
	first_frame = cv2.imread('background_front_no_shadow.jpg',0)
	first_gray = first_frame
	first_gray = cv2.GaussianBlur(first_gray, (5, 5), 0)
	kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(5,5))
	while True:
		start_time = time.time()
		gray_frame = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
		gray_frame = cv2.GaussianBlur(gray_frame, (5,5),0)
		diff = cv2.absdiff(first_gray, gray_frame)
		_, diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
		diff = cv2.morphologyEx(diff,cv2.MORPH_CLOSE,kernel,iterations = 2)
		end_time = time.time()
		print("Seconds for image processing " + str(end_time - start_time))

		bgr_image = cv2.bitwise_and(cv_image, cv_image, mask = diff)
		hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
		dom_color = get_dominant_color(hsv_image, k=1)

		end_time = time.time()
		print("Seconds for dominant color " + str(end_time - start_time))
		dom_color_hsv = np.full(bgr_image.shape, dom_color, dtype='uint8')
		dom_color_bgr = cv2.cvtColor(dom_color_hsv, cv2.COLOR_HSV2RGB)
		output_image = np.hstack((bgr_image, dom_color_bgr))

		avg = get_average_color(bgr_image)
		print(avg)
		avg = np.array([(avg[2], avg[1], avg[0])])
		avg_color_rgb = np.full(bgr_image.shape, avg, dtype='uint8')
		
		end_time = time.time()
		print("Seconds overall " + str(end_time - start_time))
		#print(end_time - start_time)
		cv2.imshow('Image Dominant Color',output_image)
		cv2.imshow('Image Average Color', avg_color_rgb)
		cv2.imshow("image window", cv_image)
		cv2.imshow("background", diff)
		if cv2.waitKey(1) == 27:
			break  # esc to quit
	cv2.destroyAllWindows()

