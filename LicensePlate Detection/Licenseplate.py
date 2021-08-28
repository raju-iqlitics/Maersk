# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 17:06:49 2021

@author: A1019089
"""
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
from tkinter import PhotoImage
import numpy as np
import cv2
import pytesseract as tess
import os 

path = os.getcwd()

print(path)

nPlateCascade = cv2.CascadeClassifier(path +'\cascade.xml')

framewidth = 480
frameheight = 480

minArea = 200
color = (255,0,255)

cap = cv2.VideoCapture(path+'\car3.jpeg')

cap.set(3, framewidth)
cap.set(4, frameheight)
cap.set(10, 150)


while True:
    success, img = cap.read()
    
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    numberPlates = nPlateCascade.detectMultiScale(imgGray, scaleFactor = 1.025, minNeighbors = 5)
    print(numberPlates)
    
    if len(numberPlates) > 0:
        for (x,y,w,h) in numberPlates:
            area = w * h
            
            
            if area > minArea:
                
                cv2.rectangle(img, (x,y),(x + w, y + h), (255,0,255), 2)                
                imgRoi = img[y:y + h, x: x + w]                
                cv2.imshow("ROI ", imgRoi)                
                
                cv2.blur(img,(9,9))
                cv2.imshow('Result', img) 
                count = 0
                if cv2.waitKey(0) & 0xff == ord('s'):
                    count +=1
                    cv2.imwrite(path + str(count)+".png", imgRoi)
                    cv2.rectangle(img, (0,200),(640,300),(0,255,0),cv2.FILLED)
                    cv2.putText(img, "Scanned Saved", (150, 265), cv2.FONT_HERSHEY_DUPLEX,2,(0,0,255),2)
                    cv2.imshow("Result", img)
                    cv2.waitKey(5)
                    
                    cv2.destroyAllWindows()
                    cap.release()
                    break
                
    else:
        print("could not detect the scale")
                    
                    
        