import gymnasium as gym

env = gym.make('CarRacing-v2', render_mode = 'human')

# state 초기 상태(env 초기화)
observation, info = env.reset()

#env에 대해 알기 위해 random action 설정
step_n = 500
for _ in range(step_n):
    action = env.action_space.sample() 

    # step 실행
    obsetvation, reword, terminated, truncated, info = env.step(action)

    # episode 종료
    if terminated or truncated:
        observation, info = env.reset()

env.close()

