import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2

from preprocessing import green_mask, gray_scale, blur_image, canny_edge_detector
from utils import image_save

def preprocess(img):
    # preprocessing in DQN paper
    # img -> grascale -> [1, 84, 84] -> normalization
    # image_save(img, './result/preprocess', '0.img.png')

    # image size : 84 * 84
    croped = img[:84, 6:90]
    # image_save(croped, './result/preprocess', '1.croped.png')

    # 배경만 남기기
    green = green_mask(croped)
    # image_save(green, './result/preprocess', '2.green.png')

    # grayscale
    grey  = gray_scale(green)
    # image_save(grey, './result/preprocess', '3.gray.png')

    # 노이즈 없애기
    blur  = blur_image(grey)
    # image_save(blur, './result/preprocess', '4.blur.png')

    # 가장자리 감지
    canny = canny_edge_detector(blur)
    # image_save(canny, './result/preprocess', '5.canny.png')

    #  normalization
    img = canny/ 255.0

    return img


class CarRacingActionSpace(gym.ActionWrapper):
    # https://brunch.co.kr/@kakao-it/144
    # mapping = {
    #                 0: (0, 0, 0),  # 정지
    #                 1: (1, 1, 0),  # 왼쪽으로 많이 틀면서 이동
    #                 2: (0.5, 1, 0),  # 왼쪽으로 조금 틀면서 이동
    #                 3: (-1, 1, 0),    # 오른쪽으로 많이 틀면서 이동
    #                 4:  (0.5, 1, 0),  # 오른쪽으로 조금 틀면서 이동
    #             }
    mapping = {
            0: (0, 0,0),
            1: (1, 1, 0),
            2: (1, -1, 0),
            3:  (0.5, 1, 0),
            4: (0.5, -1, 0),
            5: (0, 1, 0),
            6: (0, -1, 0),
            7:  (-0.5, 1, 0),
            8:  (-0.5, -1, 0),
            9:  (-1, 1, 0),
            10:  (-1, -1, 0),
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

    
    

class SetEnv(gym.Wrapper):
    """
    env를 학습할 수 있도록 초기화

    - stacked_frames: 4개 stack으로 쌓기
    - skip_frames : frame skipping technuque
    -
    """
    def __init__(self, env, is_continuous, skip_frames = 4, stack_frames = 4, initial_no_op= 50,  **kwargs):
        super(SetEnv, self).__init__(env, **kwargs)
        self.inital_no_op = initial_no_op
        self.skip_frames = skip_frames
        self.stack_frames= stack_frames
        self.is_continuous = is_continuous
    
    def reset(self, seed = 42, options = None):
        s,  info = self.env.reset()

        # 초반 50번의 step:  pygame window에서 zoom하는 과정 -> 아무것도 하지 않도록 action = 0
        # error 발생
        if self.is_continuous:
            init_action = (0, 0, 0)
        else:
            init_action = 0
        for i in range(self.inital_no_op):
            s, r, terminated, truncated, info = self.env.step(init_action)
            
        s = preprocess(s)

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
        s = preprocess(s)    # [84, 84]

        # self.stacked_frame의 끝에 s넣기
        self.stacked_frame = np.concatenate( (self.stacked_frame[1:], s[np.newaxis]), axis = 0)  # (4, 84, 84)
        return self.stacked_frame, reward, terminated, truncated, info



class ActionSpaceEnv(gym.Wrapper):
    """
    env를 학습할 수 있도록 초기화

    - stacked_frames: 4개 stack으로 쌓기
    - skip_frames : frame skipping technuque
    -
    """
    # mapping = {
    #         0: (0, 0,0),
    #         1: (1, 1, 0),
    #         2: (1, -1, 0),
    #         3:  (0.5, 1, 0),
    #         4: (0.5, -1, 0),
    #         5: (0, 1, 0),
    #         6: (0, -1, 0),
    #         7:  (-0.5, 1, 0),
    #         8:  (-0.5, -1, 0),
    #         9:  (-1, 1, 0),
    #         10:  (-1, -1, 0),
    #     }
    mapping = {
                        0: (0, 0, 0),  # 정지
                        1: (1, 1, 0),  # 왼쪽으로 많이 틀면서 이동
                        2: (0.5, 1, 0),  # 왼쪽으로 조금 틀면서 이동
                        3: (-1, 1, 0),    # 오른쪽으로 많이 틀면서 이동
                        4:  (0.5, 1, 0),  # 오른쪽으로 조금 틀면서 이동
                    }
    def __init__(self, env, is_continuous, skip_frames = 4, stack_frames = 4, initial_no_op= 50,  **kwargs):
        super(ActionSpaceEnv, self).__init__(env, **kwargs)
        self.inital_no_op = initial_no_op
        self.skip_frames = skip_frames
        self.stack_frames= stack_frames
        self.is_continuous = is_continuous
        self.action_space = spaces.Discrete(5)


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
        # error 발생
        if self.is_continuous:
            init_action = (0, 0, 0)
        else:
            init_action = 0
        for i in range(self.inital_no_op):
            s, r, terminated, truncated, info = self.env.step(init_action)
            
        s = preprocess(s)

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
        s = preprocess(s)    # [84, 84]

        # self.stacked_frame의 끝에 s넣기
        self.stacked_frame = np.concatenate( (self.stacked_frame[1:], s[np.newaxis]), axis = 0)  # (4, 84, 84)
        return self.stacked_frame, reward, terminated, truncated, info

    

    

