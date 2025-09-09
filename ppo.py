import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import gymnasium as gym
import numpy as np
import time

# --- 하이퍼파라미터 설정 ---
learning_rate = 0.0005
gamma = 0.98         # 보상 할인율
lmbda = 0.95         # GAE 람다 값 [cite: 326]
eps_clip = 0.2       # L_CLIP의 클리핑 파라미터 epsilon
K_epochs = 4         # 한 번의 데이터 수집 후 업데이트 반복 횟수 
T_horizon = 20       # 데이터 수집을 위한 타임스텝 수 

# --- 신경망 모델 정의 (Actor-Critic) ---
class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        
        # Actor: 정책을 출력 (어떤 행동을 할지 결정)
        self.actor_head = nn.Linear(256, action_dim)
        
        # Critic: 상태의 가치를 출력 (현재 상태가 얼마나 좋은지 평가)
        self.critic_head = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        
        # 행동 확률 분포 (Policy)
        action_probs = F.softmax(self.actor_head(x), dim=-1)
        
        # 상태 가치 (Value)
        state_value = self.critic_head(x)
        
        return action_probs, state_value

# --- PPO 에이전트 클래스 ---
class PPOAgent:
    def __init__(self, input_dim, action_dim):
        self.policy = ActorCritic(input_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        # 이전 정책과 현재 정책의 KL Divergence를 제한하기 위한 객체 L_CLIP [cite: 66, 127]
        self.objective_function = nn.SmoothL1Loss() # Value Loss 계산용

    def get_action(self, state):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        probs, value = self.policy(state_tensor)
        
        # 행동 확률 분포로부터 행동 샘플링
        m = Categorical(probs)
        action = m.sample()
        
        # 행동, 행동의 로그 확률, 가치 반환
        return action.item(), m.log_prob(action), value
    
    def get_action_deterministic(self, state):
        """테스트용: 확률적이 아닌 결정적 행동 선택"""
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        probs, _ = self.policy(state_tensor)
        return torch.argmax(probs).item()

    def compute_gae(self, next_value, rewards, dones, values):
        # Generalized Advantage Estimation (GAE) 계산
        # Source: Proximal Policy Optimization Algorithms, Page 4, Equation (11) 
        advantages = torch.zeros_like(rewards)
        last_gae_lam = 0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                mask = 0 # 에피소드가 끝났으면 다음 상태 가치는 0
            else:
                mask = 1
            
            # TD-error (delta) 계산
            # delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
            delta = rewards[t] + gamma * next_value * mask - values[t]
            next_value = values[t] # 다음 루프를 위해 현재 value 저장
            
            # GAE 계산
            advantages[t] = last_gae_lam = delta + gamma * lmbda * mask * last_gae_lam
            
        # 가치 함수의 타겟 값 (Returns) 계산
        # 텐서 크기 맞춤 (1차원으로 통일)
        values = values.squeeze() if values.dim() > 1 else values
        returns = advantages + values
        return advantages, returns

    def update(self, storage):
        # --- 학습 단계 ---
        # 1. Rollout Buffer에서 데이터 가져오기
        states, actions, rewards, dones, old_log_probs, values = storage

        # 2. GAE 및 가치 타겟 계산
        # 마지막 스텝의 가치를 예측하여 GAE 계산에 사용
        with torch.no_grad():
            _, last_value = self.policy(states[-1].unsqueeze(0))
        advantages, returns = self.compute_gae(last_value, rewards, dones, values)
        
        # 텐서 정규화 (학습 안정성 향상)
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        else:
            advantages = advantages - advantages.mean()

        # 3. K번의 epoch 동안 학습 진행 
        for _ in range(K_epochs):
            # 현재 정책으로 새로운 로그 확률 및 가치 계산
            new_probs, new_values = self.policy(states)
            new_values = new_values.squeeze(-1) if new_values.dim() > 1 else new_values
            
            m = Categorical(new_probs)
            new_log_probs = m.log_prob(actions)
            
            # 4. PPO의 Clipped Surrogate Objective 계산
            # ratio = pi_theta(a|s) / pi_theta_old(a|s)
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # L_CPI(theta) * A_t
            surr1 = ratio * advantages
            # clip(ratio, 1-eps, 1+eps) * A_t
            surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantages
            
            # Actor Loss (Clipped Objective)
            # 수식: L_CLIP(theta) = E[min(surr1, surr2)]
            # 경사 상승(Ascent)을 해야 하므로, 손실 함수에는 음수를 붙여 경사 하강(Descent)으로 변환
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Critic Loss (Value Function Loss)
            # 수식: L_VF(theta) = (V_theta(s_t) - V_t^targ)^2 
            critic_loss = F.mse_loss(new_values, returns)
            
            # Entropy Bonus (탐험 장려) 
            entropy = m.entropy().mean()
            
            # 최종 Loss = Actor Loss + Critic Loss - Entropy Bonus
            # (엔트로피는 커져야 하므로 손실 함수에서는 빼준다)
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
            
            # 업데이트
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


# --- 메인 학습 루프 ---
def main():
    env = gym.make('CartPole-v1', render_mode='human')
    agent = PPOAgent(env.observation_space.shape[0], env.action_space.n)
    
    print_interval = 20
    score = 0.0
    
    for n_epi in range(10000):
        # 1. 데이터 수집 (Rollout) 
        storage = [], [], [], [], [], [] # s, a, r, done, log_prob, val
        state, _ = env.reset()
        
        for t in range(T_horizon):
            action, log_prob, value = agent.get_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            
            # 화면 렌더링 (학습 과정 시각화)
            env.render()
            time.sleep(0.02)  # 애니메이션 속도 조절
            
            # 데이터 저장
            for i, item in enumerate((torch.from_numpy(state).float(), action, reward/100.0, done, log_prob.detach(), value.detach())):
                storage[i].append(item)
            
            state = next_state
            score += reward
            
            if done or truncated:
                break

        # 2. 수집된 데이터로 정책 업데이트 
        # 리스트들을 텐서로 변환 (Float 타입으로 통일)
        s, a, r, d, lp, v = [torch.tensor(np.array(item), dtype=torch.float32) for item in storage]
        v = v.squeeze(-1) if v.dim() > 1 else v  # 마지막 차원만 제거
        storage_tensors = (s, a, r, d, lp, v)
        agent.update(storage_tensors)
        
        if n_epi % print_interval == 0 and n_epi != 0:
            print(f"#{n_epi} Episode, Avg Score: {score/print_interval:.1f}")
            score = 0.0

    print("\n--- 학습 완료 ---")
    
    # 4. 학습된 정책으로 테스트 (시각화)
    test_env = gym.make('CartPole-v1', render_mode='human')
    state, _ = test_env.reset()
    done = False
    total_reward = 0

    for step in range(500):  # 최대 500 스텝 동안 테스트
        test_env.render()  # 화면 렌더링
        time.sleep(0.02)
        
        # 학습된 정책으로 결정적 행동 선택
        action = agent.get_action_deterministic(state)
        next_state, reward, terminated, truncated, info = test_env.step(action)
        done = terminated or truncated
        
        state = next_state
        total_reward += reward
        
        if done:
            break

    print(f"\n테스트 완료! 총 점수: {total_reward}")
    test_env.close()
    env.close()

if __name__ == '__main__':
    main()