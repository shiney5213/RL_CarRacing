import numpy as np
import os
import cv2
import pygame
import matplotlib.pyplot as plt
import torch

def check_dir(dir_path):
    if not os.path.isdir(dir_path):
            os.makedirs(dir_path)

def save_model(model, n_epi, t, score, optimizer, dir_path, filename, model_type):

    score = round(score, 3)
    checkpoint_path = os.path.join(dir_path, f'{filename}_{model_type}_{n_epi}_{t}_{score}.pt')
    state =  {
        'episode': n_epi,
        'score': score,
        'model': model.state_dict(),               # model weight 저
        'step': t, 
        'optimizer': optimizer.state_dict()
        }

    
    torch.save(state, checkpoint_path)

def save_returngraph(return_list, dir_path, filename, n_episode, learning_rate):
    plt.plot(return_list)
    plt.xlabel('Iteration')
    plt.ylabel('Return')

    plt.savefig(os.path.join(dir_path, f'{filename}_{n_episode}.png'), format='png', dpi=300)

def plot_durations(episode_durations, dir_path, filename, n_episode, show_result=True):
    """
    지난 100개 에피소드의 평균(공식 평가에서 사용 된 수치)에 따른 에피소드의 지속을 도표화
    """
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # 100개의 에피소드 평균을 가져 와서 도표 그리기
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.savefig(os.path.join(dir_path, f'{filename}_{n_episode}_durations.png'), format='png', dpi=300)
    


def image_save(img, dir_path, name):

    check_dir(dir_path)

    if img.shape[0] < 96:
    
    
        # 3배로 증가
        scale_percent = 300

        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        img = cv2.resize(img, (width, height))
    # print(img_resize)

    img_path = os.path.join(dir_path, name)
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
