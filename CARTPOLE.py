import gymnasium as gym
import numpy as np
import math
import time

# 환경 생성
env = gym.make('CartPole-v1', render_mode='human')

# 각 상태 변수를 몇 개의 구간으로 나눌지 정의
# (카트 위치, 카트 속도, 막대기 각도, 막대기 각속도) 순서
num_buckets = (10, 10, 10, 10) 

# 행동은 2개 (왼쪽으로 밀기, 오른쪽으로 밀기)
num_actions = env.action_space.n

# 이산화된 상태 공간의 범위
# 각 상태 변수의 최소/최대값 (CartPole 환경에서 제공)
# 카트 속도와 막대기 각속도는 무한대(-inf, inf)까지 가능하므로 적절한 값으로 제한
state_value_bounds = [
    [-4.8, 4.8],
    [-4, 4],
    [-0.418, 0.418], # 약 24도
    [-4, 4]
]

# Q-테이블 초기화
# (10, 10, 10, 10, 2) 크기
q_table = np.zeros(num_buckets + (num_actions,))

# 연속적인 상태를 이산적인 상태로 변환해주는 함수
def discretize_state(state):
    discrete_state = []
    for i in range(len(state)):
        # 상태 값이 범위 내에 있도록 조정
        scaled_value = np.clip(state[i], state_value_bounds[i][0], state_value_bounds[i][1])
        
        # 값이 어느 버킷에 속하는지 계산
        # [수정된 부분] state_value_bounds[i][2] -> state_value_bounds[i][1]
        bucket_index = int(((scaled_value - state_value_bounds[i][0]) / 
                            (state_value_bounds[i][1] - state_value_bounds[i][0])) * (num_buckets[i] - 1))
        discrete_state.append(bucket_index)
    return tuple(discrete_state)

# 2. 하이퍼파라미터 설정
alpha = 0.1             # 학습률
gamma = 0.99            # 할인율
epsilon = 1.0           # 탐험 확률
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.0001     # CartPole은 더 천천히 epsilon을 줄이는 것이 좋음
episodes = 40000        # 더 많은 학습 필요

# 3. Q-러닝 알고리즘
for episode in range(episodes):
    state, info = env.reset() # 환경 초기화
    discrete_state = discretize_state(state) # 연속된 상채를 bucket index로 변환
    done = False
    
    # ε-탐욕(ε-greedy) 정책
    while not done:
        if np.random.random() < epsilon:
            action = env.action_space.sample() # 탐험(Exploration)
        else:
            action = np.argmax(q_table[discrete_state]) # 활용(Exploitation)
            
        new_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        new_discrete_state = discretize_state(new_state)
        
        # Q-Value 업데이트(벨만 방정식)
        old_value = q_table[discrete_state + (action,)]
        next_max = np.max(q_table[new_discrete_state])
        
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[discrete_state + (action,)] = new_value
        
        discrete_state = new_discrete_state

    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode) # 반 탐험 ↑, 후반 활용 ↑
    
    if (episode + 1) % 10000 == 0:
        print(f"Episode: {episode + 1} | Epsilon: {epsilon:.4f}")

print("\n--- 학습 완료 ---")

# 4. 학습된 정책으로 테스트
state, info = env.reset()
discrete_state = discretize_state(state)
done = False
total_reward = 0

for _ in range(500): # 최대 500 스텝 동안 테스트
    env.render() # 화면 렌더링
    time.sleep(0.02)
    
    # 이제 탐험 없이 Q-테이블이 시키는대로만 행동
    action = np.argmax(q_table[discrete_state])
    new_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    
    discrete_state = discretize_state(new_state)
    total_reward += reward
    
    if done:
        break

print(f"\n테스트 완료! 총 점수: {total_reward}")
env.close()