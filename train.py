import torch.nn.functional as F
import torch
import numpy as np


def train(q, q_target, memory, optimizer, gamma, batch_size, is_continuous, device):
    """
    nn update 함수(파라미터 update)
    optimizer: train함수를 여러번 부르기 때문에, 밖에서 지정한 후, 인자로 넘기는 것이 효율적
    """

    for i in range(10):
        # 1. batch size 만큼의 batch size 가져오기 
        # s, s_prime: (32, 4, 84, 84)
        # a, r, done_mask : (32, 1)
        s, a, r, s_prime, done_mask = memory.sample(batch_size)
        
        # 2. DQN class안의 forware()함수 실행
        q_out = q(s)          # [32, 5]

        # 3. loss 구하기
        if not is_continuous:
            # 3.1 prediction 값
            # 실제로 취한 action의 Q값 구하기 -> loss function 계산하기 위해
            # descreate : a.shape = [32, 1 ]  , a[0] = tensor([2], device='cuda:0')
            q_a = q_out.gather(1, a)  

            # 3.2. target 구하기  
            # DQN 슬라이드 15p
            # DQN의 loss function 계산 : prediction(q_out), target(gamma * max Q(s', a') : 다음 state에서 취한 action 중 Q이 가장 큰 값 ) 비교
            max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)  # target network에 s_prime을 넣어서 나오는 두개의 output값 중 max값
            target = r + gamma * max_q_prime * done_mask    # dome_mask: 0이면 terminal state

            # 3.3. loss 구하기
            loss = F.mse_loss(q_a, target)

        else:
            # continuout : a.shape = [32, 1, 3]  a[0] = tensor([[-0.6242,  0.1169,  0.9398]], device='cuda:0', dtype=torch.float64)

            # 3.1 prediction 값
            # action space 값 중 어떤 값을  actiond으로 취하는지 모르겠음.
            # descrete 처럼 action space 3개 중 제일 큰 값인 action을 1개만 취했다고 가정
            new_a = []
            for i in a:
                new_a.append(i.argmax().item())
            new_a = np.array(new_a).reshape(batch_size, 1)
            q_a = q_out.gather(1, torch.tensor(new_a, dtype = int).to(device))

            # 3.2. target 구하기
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

        