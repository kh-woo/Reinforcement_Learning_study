"""
MAPPO 에이전트 구현 (Multi-Agent 안정화 기법 포함)
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from config import *

class SharedActorCritic(nn.Module):
    """
    모든 에이전트가 공유하는 Actor-Critic 네트워크
    """
    def __init__(self, input_dim, action_dim, hidden_dim=HIDDEN_DIM):
        super(SharedActorCritic, self).__init__()
        
        # 공유된 특징 추출기
        self.shared_fc1 = nn.Linear(input_dim, hidden_dim)
        self.shared_fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Actor: 정책 출력
        self.actor_head = nn.Linear(hidden_dim, action_dim)
        
        # Critic: 가치 함수 출력
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # 공유된 특징 추출
        x = F.relu(self.shared_fc1(x))
        x = F.relu(self.shared_fc2(x))
        
        # 행동 확률 분포
        action_probs = F.softmax(self.actor_head(x), dim=-1)
        
        # 상태 가치
        state_value = self.critic_head(x)
        
        return action_probs, state_value

class MAPPOAgent:
    """
    MAPPO 에이전트 클래스 (Multi-Agent 안정화 기법 포함)
    """
    def __init__(self, input_dim, action_dim, num_agents):
        self.num_agents = num_agents
        self.device = torch.device(DEVICE)
        
        # 공유된 정책 네트워크
        self.policy = SharedActorCritic(input_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LEARNING_RATE)
        
        # 경험 저장소 (모든 에이전트 통합)
        self.storage = []
        
        # 학습 통계
        self.episode_rewards = [[] for _ in range(num_agents)]
        self.episode_lengths = [[] for _ in range(num_agents)]
        
        # Multi-Agent 안정화를 위한 변수들
        self.entropy_coef = ENTROPY_COEF
        self.value_loss_coef = VALUE_LOSS_COEF
        self.gradient_clip_norm = GRADIENT_CLIP_NORM
        self.entropy_decay = ENTROPY_DECAY
        self.update_count = 0

    def get_action(self, state, agent_id):
        """공유된 정책으로 행동 선택"""
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        probs, value = self.policy(state_tensor)
        
        # 행동 샘플링
        m = Categorical(probs)
        action = m.sample()
        
        return action.item(), m.log_prob(action), value
    
    def get_action_deterministic(self, state, agent_id):
        """테스트용 결정적 행동 선택"""
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        probs, _ = self.policy(state_tensor)
        return torch.argmax(probs).item()

    def store_experience(self, agent_id, state, action, reward, done, log_prob, value):
        """경험 저장"""
        self.storage.append({
            'agent_id': agent_id,
            'state': torch.from_numpy(state).float(),
            'action': action,
            'reward': reward,
            'done': done,
            'log_prob': log_prob.detach(),
            'value': value.detach()
        })

    def clear_storage(self):
        """경험 저장소 초기화"""
        self.storage = []

    def compute_gae(self, next_values, rewards, dones, values):
        """GAE (Generalized Advantage Estimation) 계산"""
        advantages = torch.zeros_like(rewards)
        last_gae_lam = 0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                mask = 0
            else:
                mask = 1
            
            delta = rewards[t] + GAMMA * next_values[t] * mask - values[t]
            advantages[t] = last_gae_lam = delta + GAMMA * LAMBDA * mask * last_gae_lam
        
        values = values.squeeze() if values.dim() > 1 else values
        returns = advantages + values
        return advantages, returns

    def update(self):
        """정책 업데이트 (Multi-Agent 안정화 기법 포함)"""
        if len(self.storage) == 0:
            return
        
        # 모든 에이전트의 경험을 통합
        states = torch.stack([exp['state'] for exp in self.storage]).to(self.device)
        actions = torch.tensor([exp['action'] for exp in self.storage], dtype=torch.long).to(self.device)
        rewards = torch.tensor([exp['reward'] for exp in self.storage], dtype=torch.float32).to(self.device)
        dones = torch.tensor([exp['done'] for exp in self.storage], dtype=torch.bool).to(self.device)
        old_log_probs = torch.stack([exp['log_prob'] for exp in self.storage]).to(self.device)
        values = torch.stack([exp['value'] for exp in self.storage]).to(self.device)
        
        # 마지막 상태의 가치 예측
        with torch.no_grad():
            _, last_value = self.policy(states[-1].unsqueeze(0))
        
        # GAE 계산
        values = values.squeeze(-1) if values.dim() > 1 else values
        last_value = last_value.squeeze(-1) if last_value.dim() > 1 else last_value
        
        next_values = torch.cat([values[1:], last_value.unsqueeze(0)])
        advantages, returns = self.compute_gae(next_values, rewards, dones, values)
        
        # Advantage 정규화 (Multi-Agent 안정화)
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        else:
            advantages = advantages - advantages.mean()

        # K번의 epoch 동안 학습
        for _ in range(K_EPOCHS):
            new_probs, new_values = self.policy(states)
            new_values = new_values.squeeze(-1) if new_values.dim() > 1 else new_values
            
            m = Categorical(new_probs)
            new_log_probs = m.log_prob(actions)
            
            # PPO Clipped Surrogate Objective
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - EPS_CLIP, 1 + EPS_CLIP) * advantages
            
            # 손실 계산
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # 차원 맞춤
            new_values = new_values.flatten()
            returns = returns.flatten()
            min_len = min(len(new_values), len(returns))
            new_values = new_values[:min_len]
            returns = returns[:min_len]
            
            critic_loss = F.mse_loss(new_values, returns)
            entropy = m.entropy().mean()
            
            # Multi-Agent 안정화를 위한 손실 계산
            # 1. 엔트로피 감소 (학습 진행에 따라 탐험 감소)
            current_entropy_coef = self.entropy_coef * (self.entropy_decay ** self.update_count)
            
            # 2. 가치 함수 손실 가중치 조정
            current_value_coef = self.value_loss_coef
            
            # 3. 최종 손실
            loss = actor_loss + current_value_coef * critic_loss - current_entropy_coef * entropy
            
            # 업데이트
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient Clipping (Multi-Agent 안정화)
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.gradient_clip_norm)
            
            self.optimizer.step()
        
        # 업데이트 카운트 증가
        self.update_count += 1
        
        # 엔트로피 계수 업데이트
        self.entropy_coef = max(0.001, self.entropy_coef * self.entropy_decay)

    def save_model(self, filepath):
        """모델 저장"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'update_count': self.update_count,
            'entropy_coef': self.entropy_coef,
        }, filepath)

    def load_model(self, filepath):
        """모델 로드"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.update_count = checkpoint.get('update_count', 0)
        self.entropy_coef = checkpoint.get('entropy_coef', ENTROPY_COEF)