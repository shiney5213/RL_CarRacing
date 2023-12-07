import numpy as np
import os
import cv2
import pygame




def image_save(img, path, name):

    print(img.shape)

    if img.shape[0] < 96:
    
    
        # 3배로 증가
        scale_percent = 300

        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        img = cv2.resize(img, (width, height))
    # print(img_resize)

    img_path = os.path.join(path, name)
    cv2.imwrite(img_path, img)


def detect_car(img):
    """
    obserbation에서 빨간 차 찾기 찾기
    이미지가 너무 작아서 못찾나? 찾을 수 없네
    """

    img = cv2.resize(img, dsize = (300, 300))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    lower_red = (0, 0, 50)
    upper_red = (50, 50, 255)  

    mask = cv2.inRange(img, lower_red, upper_red)

    result = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow("mask", mask)

    cv2.imshow("Result", result)

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
         
    # dst1 = cv2.inRange(img, (0, 0, 0), (0, 0, 255)) # B : 0 ~ 150, G : 0 ~ 150, R : 100 ~ 255
    
    lower_hsv = (0, 200, 70)
    upper_hsv = (0, 250, 80)
    dst2 = cv2.inRange(img_hsv, lower_hsv, upper_hsv ) # H(색상) : 160 ~ 179, S(채도) : 100 ~ 255, V(진하기) : 0 ~ 255

    # cv2.imshow('src1', img)
    # cv2.imshow('dst1', dst1)
    cv2.imshow('dst2', dst2)


    cv2.waitKey(0)
    cv2.destroyAllWindows()
