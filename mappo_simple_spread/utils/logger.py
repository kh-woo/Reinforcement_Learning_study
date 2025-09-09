"""
로깅 및 통계 관리 유틸리티
"""
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import os

class TrainingLogger:
    """
    학습 과정 로깅 및 통계 관리
    """
    def __init__(self, num_agents, log_dir="logs"):
        self.num_agents = num_agents
        self.log_dir = log_dir
        
        # 로그 디렉토리 생성
        os.makedirs(log_dir, exist_ok=True)
        
        # 통계 저장소
        self.episode_rewards = [[] for _ in range(num_agents)]
        self.episode_lengths = [[] for _ in range(num_agents)]
        self.total_rewards = deque(maxlen=100)
        self.episode_count = 0
        
        # 최고 성능 추적
        self.best_reward = float('-inf')
        self.best_episode = 0
    
    def log_episode(self, rewards, episode_length, episode_num):
        """에피소드 결과 로깅"""
        self.episode_count = episode_num
        
        # 에이전트별 보상 저장
        for i, reward in enumerate(rewards):
            self.episode_rewards[i].append(reward)
        
        # 전체 보상
        total_reward = sum(rewards)
        self.total_rewards.append(total_reward)
        
        # 에피소드 길이
        for i in range(self.num_agents):
            self.episode_lengths[i].append(episode_length)
        
        # 최고 성능 업데이트
        if total_reward > self.best_reward:
            self.best_reward = total_reward
            self.best_episode = episode_num
    
    def get_statistics(self):
        """현재 통계 반환"""
        stats = {}
        
        # 전체 보상 통계
        if self.total_rewards:
            stats['avg_reward'] = np.mean(self.total_rewards)
            stats['std_reward'] = np.std(self.total_rewards)
            stats['max_reward'] = np.max(self.total_rewards)
            stats['min_reward'] = np.min(self.total_rewards)
        
        # 에이전트별 보상 통계
        for i in range(self.num_agents):
            if self.episode_rewards[i]:
                stats[f'agent_{i+1}_avg_reward'] = np.mean(self.episode_rewards[i][-100:])
                stats[f'agent_{i+1}_std_reward'] = np.std(self.episode_rewards[i][-100:])
        
        # 최고 성능
        stats['best_reward'] = self.best_reward
        stats['best_episode'] = self.best_episode
        
        return stats
    
    def print_statistics(self, episode_num, print_interval):
        """통계 출력"""
        if episode_num % print_interval == 0 and episode_num > 0:
            stats = self.get_statistics()
            
            print(f"\n=== Episode {episode_num} Statistics ===")
            print(f"Average Reward (last 100): {stats.get('avg_reward', 0):.2f}")
            print(f"Reward Std: {stats.get('std_reward', 0):.2f}")
            print(f"Best Reward: {stats.get('best_reward', 0):.2f} (Episode {stats.get('best_episode', 0)})")
            
            for i in range(self.num_agents):
                avg_reward = stats.get(f'agent_{i+1}_avg_reward', 0)
                print(f"Agent {i+1} Avg Reward: {avg_reward:.2f}")
            
            print("=" * 40)
    
    def plot_training_curves(self, save_path=None):
        """학습 곡선 플롯"""
        if not self.episode_rewards[0]:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('MAPPO Simple Spread Training Curves', fontsize=16)
        
        # 전체 보상 곡선
        axes[0, 0].plot(self.total_rewards)
        axes[0, 0].set_title('Total Reward per Episode')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 에이전트별 보상 곡선
        for i in range(self.num_agents):
            if self.episode_rewards[i]:
                axes[0, 1].plot(self.episode_rewards[i], label=f'Agent {i+1}', alpha=0.7)
        axes[0, 1].set_title('Individual Agent Rewards')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Reward')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 이동 평균 보상
        if len(self.total_rewards) > 10:
            window = min(50, len(self.total_rewards) // 2)
            moving_avg = np.convolve(self.total_rewards, np.ones(window)/window, mode='valid')
            axes[1, 0].plot(moving_avg)
            axes[1, 0].set_title(f'Moving Average Reward (window={window})')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Average Reward')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 에피소드 길이
        if self.episode_lengths[0]:
            axes[1, 1].plot(self.episode_lengths[0])
            axes[1, 1].set_title('Episode Length')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Steps')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def save_logs(self, filepath):
        """로그 저장"""
        logs = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'total_rewards': list(self.total_rewards),
            'best_reward': self.best_reward,
            'best_episode': self.best_episode,
            'episode_count': self.episode_count
        }
        
        np.savez(filepath, **logs)
        print(f"Logs saved to {filepath}")
