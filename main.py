import gymnasium as gym
import torch
import torch.optim as optim
from torchsummary import summary
import numpy as np
import os
import matplotlib.pyplot as plt
from itertools import count
import math
import time
import datetime
import logging
import json



from envi import SetEnv
from models import DQN
from buffer import ReplayBuffer
from train import train
from utils import save_returngraph, plot_durations, save_model, check_dir


def main(device):
    
    
    ### hyperparameter in paper
    # batch_size = 32
    # learing_rate = 0.00025
    # buffer_size = 1,000,000 

    # model save
    root = './result'
    dir = f'2.DQN_e{n_episode}_s{1000}'
    dir_path = os.path.join(root, dir)
    check_dir(dir_path)
    filename = f'1. DQN_{n_episode}_{1000}'

    # hyperparameter and save setting
    test_mode = True
    learning_rate = 0.0005
    gamma = 0.98  
    batch_size = 32
    epsilon_init = 1.0
    epsilon_min = 0.1
  

    if test_mode:
        ### hyperparameter in test
        n_episode = 500
        buffer_limit = 100
        buffer_save = 50
        n_episode = 10
        print_interval = 1
        max_episode_steps = 100
    else:
        ### hyperparameter in class
        n_episode = 3000
        buffer_limit = 50000
        buffer_save = 2000
        buffer_limit = 50000        # size of replay buffer
        print_interval = 20
        max_episode_steps = 1000



    # random seed 설정
    random_seed = 42
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    # random.seed(random_seed)
    
    
    ## 1. env setting
    is_continuous = True
    # is_continuous = False
    env = gym.make("CarRacing-v2", render_mode = 'state_pixels', continuous = is_continuous)
    
    env_spec = gym.envs.registry['CarRacing-v2'].to_json()
    # max_episode_steps = json.loads(env_spec)['max_episode_steps']   # 1000

    # env setting                 
    env = SetEnv(env, is_continuous)

    # env reset
    s, _  = env.reset()        # s.shape = (4, 84, 84)
    state_dim = s.shape
    if is_continuous:
        action_dim  = env.action_space.shape[0]  # Box([-1.  0.  0.], 1.0, (3,), float32)
    else:
        action_dim = env.action_space.n          # Discrete(5)
    print("The shape of an observation: ", s.shape)
    print('the shape of action_space:', action_dim)  
    

    ## 2. model
    q = DQN(state_dim[0], action_dim, random_seed).to(device)
    # print(summary(q, (4, 84, 84)))
    q_target = DQN(state_dim[0], action_dim, random_seed).to(device)
    q_target.load_state_dict(q.state_dict())

    ## 3. buffer
    memory = ReplayBuffer(state_dim, (1, ), buffer_limit, device, random_seed)

    ## 4. optimizer
    optimizer = optim.RMSprop(q.parameters(), learning_rate)

    ## 5. episode 반복
    score = 0.0
    epsilon = epsilon_init
    return_list = []
    episode_durations = []
    for i, n_epi in enumerate(range(n_episode)):
        print('episode:', i)

        time_start = time.time()
        # 1) epsilon decay
        epsilon_decay = (epsilon_init - epsilon_min) / 1e6
        epsilon -= epsilon_decay

        # 2) env 초기화
        s, _  = env.reset()        # s.shape = (4, 84, 84)

        
        # 3) transition buffer에 put
        done = False
        score = 0.0
        t = 0
        while not done:        
        # for t in range(0, max_episode_steps) :
            t += 1
            
            # transition 만들기: (s, a, r, s', done_mask)
            a = q.sample_action(torch.from_numpy(s).float().to(device), epsilon, is_continuous)
            s_prime, r, terminated, truncated, info = env.step(a)

            # r: 작은 수로 만들기
            r = r/100.0      
            
            # done : 0 -> terminal
            done = (terminated or truncated) 
            # done = terminated  
            done_mask = 0.0 if done else 1.0   

            transition = (s, a, r, s_prime, done_mask)
          
            memory.put(tuple(transition))

            # 다음 state로 가기
            s = s_prime

            score  += r
            
            # print("t =>", t, "terminated", terminated, "truncated", truncated)

            if terminated:
                save_model(q, n_epi, t, score, optimizer, dir_path, filename, 'q')
                
            if done:
                time_end = time.time()
                sec = time_end - time_start
                times = str(datetime.timedelta(seconds=sec)).split(".")[0]
                return_list.append(score)
                episode_durations.append(t + 1)
                # save_model(q, n_epi, t, score, optimizer, dir_path, f'{filename}_terminated', 'q')
                break
                
 
        # time_end = time.time()
        # sec = time_end - time_start
        # times = str(datetime.timedelta(seconds=sec)).split(".")[0]
        # return_list.append(score)
        # episode_durations.append(t + 1)
        
            
        # 4) transition 쌓이면 학습
        if memory.size() > buffer_save:
            train(q, q_target, memory, optimizer, gamma, batch_size, is_continuous, device)

        # 5) q_target model update
        if n_epi % print_interval==0 and n_epi!=0:
            q_target.load_state_dict(q.state_dict())

        # 6) log print
        if n_epi % print_interval==0 and n_epi!=0:
            print("n_episode :{}, score : {:.3f}, n_buffer : {}, eps : {:.1f}%, step : {}, time : {}".format(
                                                            n_epi, score/print_interval, memory.size(), epsilon*100, t , times)) 
            
        # 7) model save
        if score >= 0:
            if not test_mode:
                save_model(q, n_epi, t, score, optimizer, dir_path, filename, 'q')
            
        score = 0.0

    if not test_mode:
        save_returngraph(return_list, dir_path, filename, n_episode, learning_rate)
        plot_durations(episode_durations, dir_path, filename, n_episode, show_result=True)
    print('RL end')
    env.close()



if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'{device} is available')
    # device = 'cpu'
    main(device)