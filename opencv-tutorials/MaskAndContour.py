'''
Created on Oct 18, 2017

@author: wegscd
'''

import cv2
import numpy as np

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

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)

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
        
        image2 = cv2.line(image2,(xc - 20, yc),(xc + 20, yc), (255,0,0), 1)
        image2 = cv2.line(image2,(xc, yc - 20),(xc, yc + 20), (255,0,0), 1)
        # image2 = cv2.drawContours(image2, [cnt], 0, (0,255,0), 3)
        # cv2.rectangle(image2,(384,0),(510,128),(0,255,0),3)
        cv2.imshow('image2',image2)
    
    #print contours
    #print hierarchy
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cv2.destroyAllWindows()