import gymnasium as gym
import torch
import torch.optim as optim
from torchsummary import summary
import numpy as np
import os
import matplotlib.pyplot as plt
from itertools import count



from envi import SetEnv
from models import DQN
from buffer import ReplayBuffer
from train import train
from utils import save_returngraph, plot_durations, save_model


def main(device):
    
    # hyperparameter in class
    buffer_limit = 50000
    buffer_save = 2000
    learning_rate = 0.0005
    gamma = 0.98
    buffer_limit = 50000        # size of replay buffer
    batch_size = 32
    n_episode = 3000
    print_interval = 20

    # hyperparameter in paper
    epsilon_init = 1.0
    epsilon_min = 0.1
    batch_size = 32
    learing_rate = 0.00025
    buffer_size = 1,000,000 

    # hyperparameter in test
    buffer_limit = 100
    buffer_save = 50
    n_episode = 3
    print_interval = 1

    # model save
    dir_path = './result/DQN'
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    
    filename = '1. DQN'

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
    # env = gym.make("CarRacing-v2", render_mode = 'human', continuous = False)
    env = gym.make("CarRacing-v2", render_mode = 'state_pixels', continuous = False)

    # env setting                 
    env = SetEnv(env)

    # env reset
    s, _  = env.reset()        # s.shape = (4, 84, 84)
    print("The shape of an observation: ", s.shape)
    state_dim = s.shape
    action_dim = env.action_space.n

    ## 2. model
    q = DQN(state_dim[0], action_dim).to(device)
    # print(summary(q, (4, 84, 84)))
    q_target = DQN(state_dim[0], action_dim).to(device)
    q_target.load_state_dict(q.state_dict())

    ## 3. buffer
    memory = ReplayBuffer(state_dim, (1, ), buffer_limit, device)

    ## 4. optimizer
    optimizer = optim.RMSprop(q.parameters(), learning_rate)

    ## 5. episode 반복

    score = 0.0
    epsilon = epsilon_init
    return_list = []
    episode_durations = []
    for n_epi in range(n_episode):
        # 1) epsilon decay
        epsilon_decay = (epsilon_init - epsilon_min) / 1e6
        epsilon -= epsilon_decay

        # 2) env 초기화
        s, _  = env.reset()        # s.shape = (4, 84, 84)

        
        # 3) transition buffer에 put
        done = False
        score = 0.0
        # while not done:        
        for t in count() :
            # transition 만들기: (s, a, r, s', done_mask)
            a = q.sample_action(torch.from_numpy(s).float().to(device), epsilon)
            s_prime, r, terminated, truncated, info = env.step(a)

            # r: 작은 수로 만들기
            r = r/100.0      
            
            # done : 0 -> terminal
            done = (terminated or truncated)   
            done_mask = 0.0 if done else 1.0   

            transition = (s, a, r, s_prime, done_mask)
          
            memory.put(tuple(transition))

            # 다음 state로 가기
            s = s_prime

            score  += r

            if terminated:
                save_model(q, n_epi, t, score, optimizer, dir_path, filename, 'q')

            if done:
                return_list.append(score)
                episode_durations.append(t + 1)
                # save_model(q, n_epi, t, score, optimizer, dir_path, filename, 'q')
                break

        # 4) transition 쌓이면 학습
        if memory.size() > buffer_save:
            train(q, q_target, memory, optimizer, gamma, batch_size)

        # 5) q_target model update
        if n_epi%print_interval==0 and n_epi!=0:
            q_target.load_state_dict(q.state_dict())

            # 6) log print
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                                                            n_epi, score/print_interval, memory.size(), epsilon*100))
            score = 0.0

    save_returngraph(return_list, dir_path, filename, n_episode, learning_rate)
    plot_durations(episode_durations, dir_path, filename, n_episode, show_result=True)
    env.close()



if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'{device} is available')
    # device = 'cpu'
    main(device)