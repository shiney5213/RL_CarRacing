import gymnasium as gym
import torch.nn  as nn
import torch.nn.functional as F
import random
import collections
import torch
import torch.optim as optim


class DQN(nn.Module):
    def __init__(self):
        # nn.Module에서 상속받을 때, 원래 있던 init 함수가 있는 것을 먼저 부름
        # nn.Module의 class 상속 ->부모 class의 init 함수를 불러라
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(4, 128)  # input: 4개
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)  # output: action(left, right)의 Q값


    def forward(self, x):
        """
        input(x)가 들어오면 model을 통과해 output을 산출하는 과정을 정의
        """

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)   # 마지막 layer: activation 함수 안넣음 -> 관계


        return x

    def sample_action(self, obs, epsilon):
        # optional function
        # action을 고르는 funcgion
        # ops: state -> 4개짜리 vector
        out = self.forward(obs)
        random_value = random.random()

        if random_value < epsilon:
            return random.randint(0, 1)  # random으로 0, 1 중 선택
        else:
            return out.argmax().item()  # o ut의 큰 값의 index 반환 : 0 or 1


class ReplayBuffer():
    def __init__(self):
        # 실제 메모리를 저장할 저장소 : double ended queue
        # 50000개 이상되면, 이전 데이터 지우고, 새 데이터 추가
        # 
        self.buffer = collections.deque(maxlen = 50000)
    
    def size(self):
        return len(self.buffer)

    def put(self, transition):
        # transition(s, a, r, s')
        self.buffer.append(transition)

    def sample(self, n):
        # 50000개 중 minibatch size만큼 random으로 빼기
        mini_batch = random.sample(self.buffer, n)

        # modelf에 넣을 수 있는 tensor 형으로 만들기 
        #  why: nn 모델에 넣을 때 내가 한 operation(곱하기, 더하기)을 다 기억하고 있음. -> 오차역전파할 때 w, b에 대한 편미붑할 때 유용
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition


            # s, s_prime : 4개짜리 vector/ a, r: scala value
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

def train(q, q_target, memory, optimizer):
    """
    nn update 함수(파라미터 update)
    optimizer: train함수를 여러번 부르기 때문에, 밖에서 지정한 후, 인자로 넘기는 것이 효율적
    """

    gamma = 0.98

    for i in range(10):
        # 1. batch size 만큼의 batch size 가져오기 
        # s, s_prime: (32, 4)
        # a, r, done_mask : (32, 1)
        s, a, r, s_prime, done_mask = memory.sample(32)
        
        # 2. DQN class안의 forware()함수 실행
        q_out = q(s)


        # 3. loss 구하기
        # 3.1 prediction 값
        # 실제로 취한 action의 Q값 구하기
        # mini_batch: (32, 2)의 vector -> 1차원: 2shape에 해당하는 값 => a: 0 or 1값 -> q_out에서 a에 해당하는 index의 값을 가져오기
        # ex) q_out : [[0.4, 0.6], [0.3, 0.7], [0.9, 0.1]]
        # a = [ 0, 1, 0]
        # q_a = [0.4, 0.7. 0.9] 
        q_a = q_out.gather(1, a)  # action에 해당하는 q값을 가져와서 loss function을 계산하기 위함


        # 3.2. target 구하기  
        # DQN 슬라이드 15p
        # DQN의 loss function 계산 : prediction(q_out), target(gamma * max Q(s', a') : 다음 state에서 취한 action 중 Q이 가장 큰 값 ) 비교
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)  # target network에 s_prime을 넣어서 나오는 두개의 output값 중 max값
        target = r + gamma * max_q_prime * done_mask    # dome_mask: 0이면 terminal state

        # 3.3. loss 구하기
        loss = F.mse_loss(q_a, target)

        # 4. train
        # 4.1. optimizer의 gradient 없애기, 이번에 update한 내용이 다음에 남아있지 않도록 하기
        optimizer.zero_grad()
        # 4.2. loss값 이용해서 오차역전파 
        loss.backward()
        # 4.3. 파라미터 update
        optimizer.step()


        # double DQN 코드
        # argmax_Q = q(s_prime).max(1)[1].unsqueeze(1)
        # max_q_prime = q_target(s_prime).gathrer(1, argmax_Q)

        


def  main():
    env = gym.make("CartPole-v1",
                     render_mode = 'human',
                    #  render_mode = 
                     )

    # model
    q = DQN()
    q_target = DQN()   # 처음에 q와 q target 같아야함. q의 파라미터가 random으로 지정하므로, 똑같이 만들기 위해
    q_target.load_state_dict(q.state_dict())  # 하나의 모델을 쓰는 것보다 두개의 모델로 update를 천천히 진행하는 것이 결과가 더 좋다!


    # buffer
    memory = ReplayBuffer()

    # optimizer
    # training할 때 파라미터를 어떻게 바꿀 것인가? 설정-> 어떻게 최적화를 시킬 것인가
    optimizer = optim.Adam(q.parameters(), lr = 0.0005)

    # episode를 돌리면서 모델 update
    episode_n = 3000
    for n_epi in range(episode_n):
        # 1. epsilon dacay
        # 0.08 -> 0.01까지 계속 줄어듦.(episode가 200이 될때마다 0.01씩 줄어듦.)
        # epsilon: 초기: 탐색 많이 -> 점차 탐색을 적게 하도록
        epsilon = max(0.01, 0.08-0.01 * (n_epi/200))

        # 2. env 초기화
        s, _ = env.reset()


        # 3. transition 만들고 buffer에 넣기
        # terminal state가 되었는지 여부
        done = False
        
        score = 0.0
        while not done:
            # transition 만들기: (s, a, r, s', done_mask)
            a = q.sample_action(torch.from_numpy(s).float(), epsilon)   # s : obs(state)
            s_prime, r, terminated, truncated, info = env.step(a)
            r = r/100.0       # 값을 작은 수로 만들기 -> nn의 모델에 값: 0~1사이로 넣기 -> training할 때 어려움
            
            done = (terminated or truncated)   # 다음 state가 terminal인가 아닌가
            done_mask = 0.0 if done else 1.0   # 0: 끝/ 1: 계속

            memory.put((s, a, r, s_prime, done_mask))

            # 다음 state로 가기
            s = s_prime

            score  += r

            if done:
                break

        # 4. nn update
        if memory.size() > 2000:
            # momory가 2000이 되면 train 시작
            train(q, q_target, memory, optimizer)

        # 5. 20번 마다, q의 파라미터를 q_target의 파라미터로 복사
        if (n_epi != 0) and (n_epi % 20):
            q_target.load_state_dict(q.state_dict())

            # 이번 episode의 reware 출력
            print('n_epi:{}, mean_score: {:.1f}, n_buffer: {}, eps: {:.1f}%'.format(n_epi, score/20, memory.size(), epsilon * 100))
            score = 0.0

    env.close()























# entry pint 만들기
# main 함수를 부르기 전에 gpu 설정, 파라미터 받기 등 설정 가능
if __name__ == "__main__":
    main()