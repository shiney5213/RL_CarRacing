import collections
import random
import numpy as np
import torch


class ReplayBuffer():
    def __init__(self, state_dim, action_dim, max_size):
        self.buffer = collections.deque(maxlen = max_size)
        
    def size(self):
        return len(self.buffer)

    def put(self, transition):
        # transition(s, a, r, s')
        self.buffer.append(transition)

    def sample(self, n):
        # 50000개 중 minibatch size만큼 random으로 빼기
        mini_batch = random.sample(self.buffer, n)


        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition

            # s, s_prime : [4, 84, 84] array/ a, r: scala value
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        # dtype : 숫자형 변환
        return torch.tensor(s_lst, dtype = torch.float), \
                torch.tensor(a_lst),\
                torch.tensor(r_lst),\
                torch.tensor(s_prime_lst, dtype = torch.float),\
                torch.tensor(done_mask_lst)





