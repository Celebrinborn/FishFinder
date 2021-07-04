#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('jupyter nbconvert --to script main.ipynb')
#!jupyter nbconvert --to script main.ipynb


# In[2]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

ap = argparse.ArgumentParser()
# ap.add_argument('-i', '--image', required=True,
#                 help = 'path to input image')
# ap.add_argument('-c', '--config', required=True,
#                 help = 'path to yolo config file')
# ap.add_argument('-w', '--weights', required=True,
#                 help = 'path to yolo pre-trained weights')
# ap.add_argument('-cl', '--classes', required=True,
#                 help = 'path to text file containing class names')
# args = ap.parse_args()


# In[3]:


image = cv2.imread(os.path.join('data', 'test', 'portrait.jpg'))


def show(img, convert=True):
    if convert == True:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(1, figsize=(12,8))
    ax.axis('off')   
    plt.imshow(img, cmap='Greys')

    #example of how to convert to grey scale
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

show(image)


# In[4]:


#examples
#split image into 3 channels. delete blue and red channels
#B, G, R = cv2.split(img) 
#img = cv2.merge([B*0, G, R*0])



# In[18]:


video = cv2.VideoCapture(os.path.join('data', 'test', 'fish2.mp4'))

counter = 0

ret, frame1 = video.read()
ret, frame2 = video.read()
while(video.isOpened()):
    #ret, frame = video.read()

    diff = cv2.absdiff(frame1, frame2)
    grey = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (5,5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #cv2.drawContours(frame1, [contour for contour in contours if cv2.contourArea(contour) > 400], -1, (0,255,0), thickness=2)

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        
        if cv2.contourArea(contour) < 700:
            continue
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0,255,0), thickness=2)
        cv2.putText(frame1, f'movement: ', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,255), thickness=3)
        img = frame1[y:y+h, x:x+w]
        cv2.imwrite(os.path.join('data', 'output', f'movement_{counter}.jpg'), img)
        counter += 1
    
    
    


    cv2.imshow('feed',frame1)

    frame1 = frame2

    ret, frame2 = video.read()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

