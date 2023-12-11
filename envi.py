import gymnasium as gym
import numpy as np
import cv2


def preprocess(img):
    # preprocessing in DQN paper
    # img -> grascale -> [1, 84, 84] -> normalization
    img = img[:84, 6:90]

    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)/ 255.0

    return img





class SetEnv(gym.Wrapper):
    """
    env를 학습할 수 있도록 초기화

    - stacked_frames: 4개 stack으로 쌓기
    - skip_frames : frame skipping technuque
    -

    """

    def __init__(self, env, skip_frames = 4, stack_frames = 4, initial_no_op= 50, **kwargs):
        super(SetEnv, self).__init__(env, **kwargs)
        self.inital_no_op = initial_no_op
        self.skip_frames = skip_frames
        self.stack_frames= stack_frames
        # self.stacked_frame 
    
    def reset(self):

        s,  info = self.env.reset()


        # 초반 50번의 step:  pygame window에서 zoom하는 과정 -> 아무것도 하지 않도록 action = 0
        # error 발생
        for i in range(self.inital_no_op):
            s, r, terminated, truncated, info = self.env.step(0)
            
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
            s, r, terminated, truncated, info = self.env.step(action)
            reward += r
            if terminated or truncated:
                break
        s = preprocess(s)    # [84, 84]

        # self.stacked_frame의 끝에 s넣기
        self.stacked_frame = np.concatenate( (self.stacked_frame[1:], s[np.newaxis]), axis = 0)  # (4, 84, 84)
        return self.stacked_frame, reward, terminated, truncated, info

    
