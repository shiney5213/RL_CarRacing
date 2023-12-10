import collections
import random
import numpy as np
import torch


class ReplayBuffer():
    def __init__(self, state_dim, action_dim, max_size, device):
        self.buffer = collections.deque(maxlen = max_size)
        self.device = device
        
    def size(self):
        return len(self.buffer)

    def put(self, transition):
        # transition(s, a, r, s')
        self.buffer.append(transition)



    def sample(self, batch_size):
        # 50000개 중 minibatch size만큼 random으로 빼기
        mini_batch = random.sample(self.buffer, batch_size)   
        # print('mini_batch', len(mini_batch))         # mini_batch
        # print(mini_batch[0][0].shape)                # [4, 84, 84]

        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition     # ndarray, int, float, ndarray, float
            
            # s, s_prime : [4, 84, 84] array/ a, r: scala value
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        s_lst = np.array(s_lst)
        s_prime_lst = np.array(s_prime_lst)
        # dtype : 숫자형 변환
        return torch.tensor(s_lst, dtype=torch.float).to(self.device),\
               torch.tensor(a_lst).to(self.device), \
               torch.tensor(r_lst).to(self.device), \
               torch.tensor(s_prime_lst, dtype=torch.float).to(self.device), \
               torch.tensor(done_mask_lst).to(self.device)
    
    

