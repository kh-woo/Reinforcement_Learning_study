import gymnasium as gym
import numpy as np
import time
import random

# 1. 환경 설정
env = gym.make('FrozenLake-v1', is_slippery=True, render_mode='human')

# 2. Q-테이블 초기화
action_space_size = env.action_space.n
state_space_size = env.observation_space.n
q_table = np.zeros((state_space_size, action_space_size))

# 3. 하이퍼파라미터 설정
episodes = 20000        # 총 학습 시킬 게임 횟수
alpha = 0.1             # 학습률 (Learning Rate): 얼마나 현재 Q-table을 업데이트할지
gamma = 0.99            # 할인율 (Discount Factor): 미래의 보상을 얼마나 가치있게 여길지

# 탐험(Exploration) 관련 파라미터
epsilon = 1.0           # 탐험 확률. 처음엔 100% 무작위 행동.
max_epsilon = 1.0       # 탐험 확률의 최대값
min_epsilon = 0.01      # 탐험 확률의 최소값
decay_rate = 0.0005     # 탐험 확률을 점차 줄여나갈 비율

# 4. Q-러닝 알고리즘 (학습 루프)
for episode in range(episodes):
    # 각 게임(에피소드) 시작 시 환경과 상태 초기화
    state, info = env.reset()
    done = False
    
    while not done:
        # Exploration vs Exploitation (탐험 vs 활용)
        if random.uniform(0, 1) < epsilon:
            # 탐험: 무작위 행동 선택
            action = env.action_space.sample()
        else:
            # 활용: 현재 상태에서 Q-Value가 가장 높은 행동 선택
            action = np.argmax(q_table[state, :])
            
        # 선택한 행동으로 환경과 상호작용
        new_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Q-Value 업데이트 (벨만 방정식)
        # Q(s,a) <- Q(s,a) + α * [R + γ * max(Q(s',a')) - Q(s,a)]
        old_value = q_table[state, action]
        next_max = np.max(q_table[new_state, :])
        
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value
        
        # 다음 상태로 이동
        state = new_state

    # 게임이 끝난 후 epsilon 값을 점차 줄여나감 (학습이 진행될수록 탐험보다 활용을 더 많이 하도록)
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
    
    if (episode + 1) % 5000 == 0:
        print(f"Episode: {episode + 1}")

print("\n--- 학습 완료 ---")
print("최종 Q-Table:")
print(q_table)


# 5. 학습된 에이전트 성능 테스트
print("\n--- 학습된 에이전트 테스트 ---")
state, info = env.reset()
done = False
time.sleep(1)

for step in range(100):
    # 'human' 모드에서는 이 함수가 그래픽 창을 업데이트함. print 불필요.
    env.render()
    time.sleep(0.3)
    
    # 이제 탐험 없이, 오직 Q-table에서 가장 좋다고 생각하는 행동만 수행
    action = np.argmax(q_table[state, :])
    new_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    
    state = new_state
    
    if done:
        # 마지막 프레임 렌더링
        env.render()
        if reward == 1:
            print("성공적으로 도착점에 도달했습니다!")
        else:
            print("구멍에 빠졌습니다...")
        time.sleep(1)
        break

env.close()