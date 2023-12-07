import gymnasium as gym
import torch
import torch.optim as optim
from torchsummary import summary
import numpy as np


from envi import SetEnv
from models import DQN
from buffer import ReplayBuffer


def main(device):
    BUFFER_SIZE = 50000
    EPSILON = 1.0
    EPSILON_MIN = 0.1
    LERNING_RATE = 0.00025
    # N_EPISODE = 3000  
    N_EPISODE = 1





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

    q_target = DQN(state_dim[0], action_dim).to(device)
    q_target.load_state_dict(q.state_dict())



    ## 3. buffer
    memory = ReplayBuffer(state_dim, (1, ), BUFFER_SIZE)


    ## 4. optimizer
    optimizer = optim.RMSprop(q.parameters(), LERNING_RATE)

    ## 5. episode 반복
    
    epsilon = EPSILON
    for n_epi in range(N_EPISODE):
        # epsilon decay
        epsilon_decay = (EPSILON - EPSILON_MIN) / 1e6
        epsilon -= epsilon_decay

        # env 초기화
        s, _  = env.reset()        # s.shape = (4, 84, 84)

        # transition buffer에 put
        done = False

        score = 0.0
        # while not done:
        # while(1):  
        for i in range(0, 2) :
            print('step:', i + 1)
            # transition 만들기: (s, a, r, s', done_mask)
            a = q.sample_action(torch.from_numpy(s).float().to(device), epsilon)
            s_prime, r, terminated, truncated, info = env.step(a)
            r = r/100.0      
            
            # done : 0 -> terminal
            done = (terminated or truncated)   
            done_mask = 0.0 if done else 1.0   

            transition = (s, a, r, s_prime, done_mask)
            transition_tensor= []
            for i in transition:
                transition_tensor.append(torch.from_numpy(i).to(device) if isinstance(i, np.ndarray) else i)

            memory.put(tuple(transition_tensor))

            # 다음 state로 가기
            s = s_prime

            score  += r

            if done:
                break









if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'{device} is available')
    main(device)