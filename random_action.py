import gymnasium as gym
import numpy as np
from PIL import Image
import os

from utils import image_save, detect_car


env_spec = gym.envs.registry['CarRacing-v2'].to_json()
print(env_spec)

DIR_PATH = 'result/1.random'
N_TIME = 1
env = gym.make('CarRacing-v2', render_mode = 'human', continuous=False)
print(env.action_space)


# state 초기 상태(env 초기화)
observation, info = env.reset()
image_save(observation, DIR_PATH, f'opservation{N_TIME}_0.jpg')


#env에 대해 알기 위해 random action 설정
step_n = 500
for i, _ in enumerate(range(step_n)):

    ### 1.action_sapce 
    # print('action_space', type(env.action_space), env.action_space)
    # print('type', type(env.action_space))    
    # print('low', env.action_space.low)  # [-1.  0.  0.]
    # print('high', env.action_space.high)  # [1. 1. 1.]
    # print('sample', env.action_space.sample())  # [-0.76988626  0.88109034  0.9193942 ]

    # random하게 action 선택
    action = env.action_space.sample()


    #### 2. Observation Space
    # print('Observation Space:', env.observation_space)
    # print('shape', env.observation_space.shape)
    # print('low', env.observation_space.low)
    # print('high', env.observation_space.high)

    ### step 실행
    obsetvation, reword, terminated, truncated, info = env.step(action)

    if i % 100 == 0:
        print(i, reword, terminated, truncated, info)
        image_save(observation, DIR_PATH, f'opservation{N_TIME}_{i}.jpg')

    
    # episode 종료
    if terminated or truncated:
        observation, info = env.reset()

env.close()

