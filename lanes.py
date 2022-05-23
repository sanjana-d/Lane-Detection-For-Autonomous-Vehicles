import cv2
import numpy as np
import matplotlib.pyplot as plt

# using opencv library functions to open and show an image

def make_coordinates(image, line_paramters):
	try:
		slope, intercept = line_paramters
	except TypeError:
		slope, intercept = 0.0001, 0
	y1 = image.shape[0]
	y2 = int(y1*(3/5)) # line starts at 700 and goes up to 420
	x1 = int((y1 - intercept)/slope)
	x2 = int((y2 - intercept)/slope)
	return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
	left_fit = []
	right_fit = []
	for line in lines:
		x1, y1, x2, y2 = line.reshape(4)
		parameters = np.polyfit((x1, x2), (y1, y2), 1) # fit a first degree polynomial y=mx+b to our x,y points and return vector of coefs which describe m,b
		slope = parameters[0]
		intercept = parameters[1]
		if slope < 0:
			left_fit.append((slope, intercept))
		else:
			right_fit.append((slope, intercept))
	left_fit_average = np.average(left_fit, axis=0)
	right_fit_average = np.average(right_fit, axis=0)
	left_line = make_coordinates(image, left_fit_average)
	right_line = make_coordinates(image, right_fit_average)
	return np.array([left_line, right_line])

def canny(image):
	gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) # converts img from one color space to another
	blur = cv2.GaussianBlur(gray, (5, 5), 0) # reduce noise
	canny = cv2.Canny(blur, 50, 150) # compute gradients
	return canny

def display_lines(image, lines):
	line_image = np.zeros_like(image) # black img
	if lines is not None:
		for x1, y1, x2, y2 in lines:
			# x1, y1, x2, y2 = line.reshape(4)
			cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10) # draws a line seg connecting two points onto line_image black picture
	return line_image

# returns region of interst masked onto a black image
def region_of_interest(image):
	height = image.shape[0] # array is mxn where m is # of rows will be close to 700
	polygons = np.array([
	[(200, height), (1110, height), (550, 250)]
	]) # region of interest coordinates
	mask = np.zeros_like(image) # creates array of 0 with shape of image- all black pixels 
	cv2.fillPoly(mask, polygons, 255) # map triangle onto black img

	masked_image = cv2.bitwise_and(image, mask)
	return masked_image

# image = cv2.imread('test_image.jpeg') # reads img from file and returns it as a multi-dim numpy array- img data
# lane_image = np.copy(image)
# canny_image = canny(lane_image)
# cropped_image = region_of_interest(canny_image)
# # 1 rad is pi/180 , 4 arg is threshold, # of votes in bin of hough space to be considered line
# lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5) 
# averaged_lines = average_slope_intercept(lane_image, lines)
# line_image = display_lines(lane_image, averaged_lines)
# final_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1) # pixelwise addition
# cv2.imshow('result', final_image) # show img
# cv2.waitKey(0) # display img for infinite time until key is pressed

# video capture
cap = cv2.VideoCapture("test2.mp4")
while(cap.isOpened()):
	_, frame = cap.read() # decode every video frame. returns boolean and the current frame/snapshot
	canny_image = canny(frame)
	cropped_image = region_of_interest(canny_image)
	# 1 rad is pi/180 , 4 arg is threshold, # of votes in bin of hough space to be considered line
	lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=7) 
	averaged_lines = average_slope_intercept(frame, lines)
	line_image = display_lines(frame, averaged_lines)
	final_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1) # pixelwise addition
	cv2.imshow('result', final_image) # show img
	# when we press q, exit
	if cv2.waitKey(1) == ord('q'):
		break;
cap.release()
cv2.destroyAllWindows()








# Edge detection algorithm
# purpose: identify regions in img when sharp changes in intensity of pixels- color

# background:
# img can be read as a matrix which is an array of pixels
# each pixels intensity is denoted by a numeric value [0,255] (0-no intensity black, 255-max intensity white)
# gradient- measure of change in brightness over adjecent pixels

# step 1: convert img to grayscale- processing single channel is faster than 3 channel color image (done above)

# step 2: reduce noise and smoothen img- Gaussian filtering (done above)
# Gaussian filtering- set each pixel value to avg of neighbouring pixels. Use 5x5 kernel

# step 3: finding edges- use canny - perform a derivative/gradient of our function in both x,y dim 
# small derivative- small change in intensity
# large derivative- big change in intensity

# step 4: specify a region of interest

# step 5: bit wise AND between mask and canny img pixels to show only regions belonging in triangle

# step 6: Hough Transform y=mx+b ((x,y)->(m,b)) where m,b is hough space
# for vertical lines rho = xcos(theta) + ysin(theta) equation (polar coords)
# theta- angle from normal measured cw