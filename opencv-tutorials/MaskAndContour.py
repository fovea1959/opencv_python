'''
Created on Oct 18, 2017

@author: wegscd
'''

import cv2
import numpy as np

kernel = np.ones((5,5), np.uint8)

# https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
colors = (
    (230, 25, 75),
    (60, 180, 75),
    (255, 225, 25),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
    (250, 190, 190)
)

cap = cv2.VideoCapture(0)

while cap.isOpened():

    # Take each frame
    _, frame = cap.read()

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame, frame, mask= mask)

    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    
    image, contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # image = cv2.drawContours(image, contours, -1, (0,255,0), 3)
    cv2.imshow('image',image)
    
    if len(contours) > 0:
        contours.sort(key = lambda c: - cv2.contourArea(c))
        
        cnt = contours[0]
        print cnt
        image2 = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        x,y,w,h = cv2.boundingRect(cnt)
        image2 = cv2.rectangle(image2,(x,y),(x+w,y+h),(0,255,0),1)
        
        xc = x + (w/2)
        yc = y + (h/2)
        
        image2 = cv2.line(image2, (xc - 20, yc), (xc + 20, yc), (255, 0, 0), 1)
        image2 = cv2.line(image2, (xc, yc - 20), (xc, yc + 20), (255, 0, 0), 1)
        # image2 = cv2.drawContours(image2, [cnt], 0, (0,255,0), 3)
        
        for i, contour in enumerate(contours):
            cv2.drawContours(image2, [contour], -1, colors[i % len(colors)], 1)
        
        cv2.imshow('image2',image2)
    
    #print contours
    #print hierarchy
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cv2.destroyAllWindows()