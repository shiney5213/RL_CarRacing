import cv2
import numpy as np
from  utils import image_save

def basicpreprocess(img, is_save = False):
    # image size : 84 * 84
    croped = img[:84, 6:90]
    gray  = gray_scale(croped)
    return gray


def preprocess(img ,is_save = False):
    # preprocessing in DQN paper
    # img -> grascale -> [1, 84, 84] -> normalization

    # image size : 84 * 84
    croped = img[:84, 6:90]

    # 배경만 남기기
    green = green_mask(croped)

    # grayscale
    grey  = gray_scale(green)

    # 노이즈 없애기
    blur  = blur_image(grey)

    # 가장자리 감지
    canny = canny_edge_detector(blur)
    if is_save:
        image_save(img, './result/preprocess', '0.img_resize.png')
        image_save(croped, './result/preprocess', '1.croped.png')
        image_save(green, './result/preprocess', '2.green.png')
        image_save(grey, './result/preprocess', '3.gray.png')
        image_save(blur, './result/preprocess', '4.blur.png')
        image_save(canny, './result/preprocess', '5.canny.png')

    #  normalization
    img = canny/ 255.0

    
    return img

def preprocess_224(img, is_save = False):
    # preprocessing in DQN paper
    # img -> grascale -> [1, 84, 84] -> normalization
    image_save(img, './result/preprocess', '0.img_244.png')

    # image size 증가
    scale_percent = 300
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    img_resize = cv2.resize(img, (width, height))
    # image size : 84 * 84
    croped = img_resize[:224, 32:256]

    # 배경만 남기기
    green = green_mask(croped)

    # grayscale
    grey  = gray_scale(green)

    # 노이즈 없애기
    blur  = blur_image(grey)

    # 가장자리 감지
    canny = canny_edge_detector(blur)

    if is_save:
        image_save(img, './result/preprocess', '0.img_resize_244.png')
        image_save(croped, './result/preprocess', '1.croped_244.png')
        image_save(green, './result/preprocess', '2.green_244.png')
        image_save(grey, './result/preprocess', '3.gray_244.png')
        image_save(blur, './result/preprocess', '4.blur_244.png')
        image_save(canny, './result/preprocess', '5.canny_244.png')

    #  normalization
    img = canny/ 255.0

    img =img

    return img

def green_mask(observation):
    
    #convert to hsv
    hsv = cv2.cvtColor(observation, cv2.COLOR_BGR2HSV)
    
    # seek green part
    mask_green = cv2.inRange(hsv, (36, 25, 25), (70, 255,255))

    #slice the green
    imask_green = mask_green>0
    green = np.zeros_like(observation, np.uint8)
    green[imask_green] = observation[imask_green]
    
    return(green)

def gray_scale(observation):
    gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
    return gray


def blur_image(observation):
    blur = cv2.GaussianBlur(observation, (5, 5), 0)
    return blur


def canny_edge_detector(observation):
    canny = cv2.Canny(observation, 50, 150)
    return canny


