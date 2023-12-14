import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2

from preprocessing import preprocess, basicpreprocess
from utils import image_save


    
    

class DescreteEnv(gym.Wrapper):
    """
    - decrete한 Env
    - stacked_frames: 4개 stack으로 쌓기
    - skip_frames : frame skipping technuque
    """
    def __init__(self, env, stack_frames = 4, is_preprocess = True, skip_frames = 4,  initial_no_op= 50,  **kwargs):
        super(DescreteEnv, self).__init__(env, **kwargs)
        self.inital_no_op = initial_no_op
        self.skip_frames = skip_frames
        self.stack_frames= stack_frames
        self.is_preprocess = is_preprocess
    
    def reset(self, seed = 42, options = None):
        s,  info = self.env.reset()

        # 초반 50번의 step:  pygame window에서 zoom하는 과정 -> 아무것도 하지 않도록 action = 0
        init_action = 0
        for i in range(self.inital_no_op):
            s, r, terminated, truncated, info = self.env.step(init_action)
        
        if self.is_preprocess:
            s = preprocess(s)
        else:
            s = basicpreprocess(s)


        
        # 초기 observation : 같은 s를 4번 쌓기 -> [4, 84, 84]
        self.stacked_frame = np.tile(A = s,
                                     reps = (self.stack_frames, 1, 1))


        return self.stacked_frame, info

    def step(self, action):
        
        # 각 state마다 frame 1개 제공 -> 자동차의 이동 방향 알 수 없음.
        # history frame 4개 쌓아서 제공
        reward = 0
        for _ in range(self.skip_frames):
            # print('action', action)
            # print('self.env', self.env())
            s, r, terminated, truncated, info = self.env.step(action)
            reward += r
            if terminated or truncated:
                break

        if self.is_preprocess:
            s = preprocess(s)   # [84, 84]
        else:
            s = basicpreprocess(s)

        # self.stacked_frame의 끝에 s넣기
        self.stacked_frame = np.concatenate( (self.stacked_frame[1:], s[np.newaxis]), axis = 0)  # (4, 84, 84)
        return self.stacked_frame, reward, terminated, truncated, info



class ContinuousEnv(gym.Wrapper):
    """
    - 원래 action space가 continuous한 상태의 env
    - but continuous한 action space를 descrete한 상태 5가지 변경한 env
    """
    mapping = {
                    0: (0, 0, 0),  # 정지
                    1: (1, 0, 0),  # 왼쪽으로  steer
                    2: (-1, 0, 0),  # 오른쪽으로 steer
                    3: (0, 1, 0),    # 가속
                    4:  (0, 0, 1),  # 감속
                }
    def __init__(self, env, stack_frames = 4, is_preprocess = True, skip_frames = 4,  initial_no_op= 50,  **kwargs):
        super(ContinuousEnv, self).__init__(env, **kwargs)
        self.inital_no_op = initial_no_op
        self.skip_frames = skip_frames
        self.stack_frames= stack_frames
        self.action_space = spaces.Discrete(5)
        self.is_preprocess = is_preprocess


    def _action(self, action):
        return self.mapping.get(action)
        
    def _reverse_action(self, action):
        for k in self.mapping.keys():
            if (self.mapping[k] == action) :
                return self.mapping[k]
        return 0

    
    def reset(self, seed = 42, options = None):
        s,  info = self.env.reset()

        # 초반 50번의 step:  pygame window에서 zoom하는 과정 -> 아무것도 하지 않도록 action = 0
        init_action = (0, 0, 0)
        for i in range(self.inital_no_op):
            s, r, terminated, truncated, info = self.env.step(init_action)
            
        if self.is_preprocess:
            s = preprocess(s)
        else :
            s = basicpreprocess(s)

            

        # 초기 observation : 같은 s를 4번 쌓기 -> [4, 84, 84]
        self.stacked_frame = np.tile(A = s,
                                     reps = (self.stack_frames, 1, 1))


        return self.stacked_frame, info

    def step(self, action):
        
        # 각 state마다 frame 1개 제공 -> 자동차의 이동 방향 알 수 없음.
        # history frame 4개 쌓아서 제공
        reward = 0
        for _ in range(self.skip_frames):
            # print('action', action)
            # print('self.env', self.env())
            s, r, terminated, truncated, info = self.env.step(action)
            reward += r
            if terminated or truncated:
                break

        if self.is_preprocess:
            s = preprocess(s )# [84, 84]
        else:
            s = basicpreprocess(s)


        # self.stacked_frame의 끝에 s넣기
        self.stacked_frame = np.concatenate( (self.stacked_frame[1:], s[np.newaxis]), axis = 0)  # (4, 84, 84)
        return self.stacked_frame, reward, terminated, truncated, info

    

    


class CarRacingActionSpace(gym.ActionWrapper):
    # https://brunch.co.kr/@kakao-it/144
    mapping = {
                    0: (0, 0, 0),  # 정지
                    1: (1, 0, 0),  # 왼쪽으로  steer
                    2: (-1, 0, 0),  # 오른쪽으로 steer
                    3: (0, 1, 0),    # 가속
                    4:  (0, 0, 1),  # 감속
                }
    def __init__(self, env):
        super(CarRacingActionSpace, self).__init__(env)
        # self.action_space = spaces.Discrete(11)
        self.action_space = spaces.Discrete(5)


    def _action(self, action):
        return self.mapping.get(action)
        
    def _reverse_action(self, action):
        for k in self.mapping.keys():
            if (self.mapping[k] == action) :
                return self.mapping[k]
        return 0

