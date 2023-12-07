# RL_CarRacing
Reinforcement Learning으로 CarRacing을 진행합니다

---

## 0. CarRacing info
### 1) env_spec
- reward_threshold: 900        -> 작업이 해결된 것으로 보는 보상 임계값
- nondeterministic: false      -> deterministic 환경(non slippery)
- max_episode_steps: 1000      -> max_step: 1000
- order_enforce: true
- autoreset: false              -> episode가 끝날 때마다 env.reset() 해야 함.
- disable_env_checker: false 
- apply_api_compatibility: false
-  vector_entry_point: null

### 2) action_space
- action_space: Discrete(5)
- [do nothing, steer left, steer right, gas, brake]

### 3) observation Space
- Box(0, 255, (96, 96, 3), uint8)
- 96x96 RGB image
- low: 모두 0으로 가득찬 img
- high: 모두 255로 가득찬 img
- preprocess : img -> grascale -> [1, 84, 84] -> normalization

### 4) opservation state 초기화 -> 새로운 episode 시작 
- env: 길은 항상 바뀜 -> 고정할 때 env.seed(42)불가
- agemt: 항상 같은 위치에 있음.-> 빨간색의 좌표값 확인하자



## 추가로 하고 싶은 것
1. 내 action(방향, gas, breaking)과 rewoar 값이 pygame에 나타나면 좋겠다
2. 일단 처음에는 env를 고정하고 시작하자 -> 추후에 변경


## 1. DQN
### 1) input: stacked_frames
- preprocessing : img -> grayscale -> [1, 84, 84] -> [4, 84, 84]
- 기본 env에서 제공하는 state(frame)는 1개 -> 이동 방향 알 수 없음.
- state : history frame 4개 쌓아서 사용 
- reward:  4개 state의 합 사용
### 2) output:Q값
- Q learning : 특정 s에서 특정 a를 취했을 때 가치 추정
- output : 각 action에 대한 Q값 
### 3) model : Deep Q-Networks(DQN)
- input : [84, 84, 4]
- 1st layer: Convolution layer(filter: 8 * 8, stride: 4)
- 2nt layer : Convolution layer(filter: 4 * 4, stride: 2)
- 3th layer : fully connected layer ( unit: 256)
#### target
#### predict
#### loss function

### 4) ect
#### skip frame technique
- 모든 게임의 4번째 frame을 건너 뛰기 
- 1. 학습 속도 개선: 초당 처리하는 frame 수 감소
- 2. 중요한 정보에 초점 : frame 간의 상관관계가 높고, 연속적인 프레임에 중복 요소가 많을 때 사용
- 3. 학습 안정성 향상 : 잡음이 많거나 관련없는 프레임의 영향 감소