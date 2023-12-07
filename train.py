def train(q, q_target, memory, optimizer):
    """
    nn update 함수(파라미터 update)
    optimizer: train함수를 여러번 부르기 때문에, 밖에서 지정한 후, 인자로 넘기는 것이 효율적
    """

    GAMMA = 0.98
    MINI_BATCH = 32

    

    for i in range(10):
        # 1. batch size 만큼의 batch size 가져오기 
        # s, s_prime: (32, 4, 84, 84)
        # a, r, done_mask : (32, 1)
        s, a, r, s_prime, done_mask = memory.sample(MINI_BATCH)
        
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

        