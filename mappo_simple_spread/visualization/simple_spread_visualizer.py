"""
Simple Spread 환경을 위한 시각화 클래스
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from config import *

class SimpleSpreadVisualizer:
    """
    Simple Spread 환경 시각화
    """
    def __init__(self, num_agents=NUM_AGENTS):
        self.num_agents = num_agents
        plt.style.use('dark_background')
        
        # 플롯 설정
        self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 8))
        self.ax.set_xlim(-2, 2)
        self.ax.set_ylim(-2, 2)
        self.ax.set_aspect('equal')
        self.ax.set_facecolor('#1a1a1a')
        self.ax.set_title('Simple Spread - MAPPO', fontsize=16, fontweight='bold', color='white')
        
        # 그리드 설정
        self.ax.grid(True, alpha=0.3, color='gray')
        self.ax.tick_params(colors='white')
        
        # 에이전트 색상
        self.agent_colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57']
        
        # 상태 정보 텍스트
        self.status_text = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes, 
                                       fontsize=10, verticalalignment='top',
                                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
                                       color='white')
        
        plt.ion()
        plt.show()
    
    def update(self, observations, rewards=None, dones=None):
        """시각화 업데이트"""
        try:
            # 기존 요소들 제거 (clear 대신)
            for artist in self.ax.lines + self.ax.collections + self.ax.patches:
                if artist != self.status_text:
                    artist.remove()
            
            # 에이전트 위치 및 상태 시각화
            agent_positions = []
            agent_velocities = []
            
            for i, obs in enumerate(observations):
                # 관찰값에서 위치 추출 (Simple Spread는 [x, y, vx, vy, ...] 형태)
                if len(obs) >= 4:
                    x, y = obs[0], obs[1]
                    vx, vy = obs[2], obs[3]
                else:
                    x, y = 0, 0
                    vx, vy = 0, 0
                
                agent_positions.append([x, y])
                agent_velocities.append([vx, vy])
                
                # 에이전트 그리기
                color = self.agent_colors[i % len(self.agent_colors)]
                
                # 에이전트 원
                agent_circle = plt.Circle((x, y), 0.1, color=color, alpha=0.8, 
                                        edgecolor='white', linewidth=2)
                self.ax.add_patch(agent_circle)
                
                # 속도 벡터
                if np.linalg.norm([vx, vy]) > 0.01:
                    self.ax.arrow(x, y, vx*0.5, vy*0.5, head_width=0.05, 
                                head_length=0.05, fc=color, ec=color, alpha=0.6)
                
                # 에이전트 번호
                self.ax.text(x, y+0.15, f'A{i+1}', ha='center', va='center', 
                           color='white', fontweight='bold', fontsize=8)
            
            # 랜드마크 (목표 지점) 그리기
            self._draw_landmarks(observations)
            
            # 상태 정보 업데이트
            if rewards is not None and dones is not None:
                self._update_status_text(agent_positions, agent_velocities, rewards, dones)
            
            plt.draw()
            # Ctrl+C 문제 해결을 위해 더 안전한 방법 사용
            try:
                plt.pause(0.01)  # 매우 짧은 시간으로 변경
            except KeyboardInterrupt:
                print("시각화 중단됨")
                return
            
        except Exception as e:
            print(f"Visualization error: {e}")
            pass
    
    def _draw_landmarks(self, observations):
        """랜드마크 (목표 지점) 그리기"""
        # Simple Spread에서는 에이전트가 랜드마크에 도달해야 함
        # 관찰값에서 랜드마크 정보 추출 (실제 환경에 따라 조정 필요)
        landmark_positions = [
            [0.5, 0.5],   # 랜드마크 1
            [-0.5, 0.5],  # 랜드마크 2
            [0, -0.5]     # 랜드마크 3
        ]
        
        for i, (lx, ly) in enumerate(landmark_positions):
            landmark = plt.Circle((lx, ly), 0.08, color='#ffd700', alpha=0.7,
                                edgecolor='white', linewidth=2)
            self.ax.add_patch(landmark)
            self.ax.text(lx, ly+0.12, f'L{i+1}', ha='center', va='center',
                        color='white', fontweight='bold', fontsize=8)
    
    def _update_status_text(self, positions, velocities, rewards, dones):
        """상태 정보 텍스트 업데이트"""
        status_lines = ["=== Simple Spread Status ==="]
        
        for i, (pos, vel, reward, done) in enumerate(zip(positions, velocities, rewards, dones)):
            x, y = pos
            vx, vy = vel
            speed = np.linalg.norm([vx, vy])
            
            status = "DONE" if done else "ACTIVE"
            status_lines.append(f"Agent {i+1}: Pos({x:.2f}, {y:.2f}) | "
                              f"Vel({vx:.2f}, {vy:.2f}) | Speed: {speed:.2f} | "
                              f"Reward: {reward:.2f} | {status}")
        
        # 전체 보상
        total_reward = sum(rewards)
        status_lines.append(f"Total Reward: {total_reward:.2f}")
        
        self.status_text.set_text('\n'.join(status_lines))
    
    def close(self):
        """시각화 종료"""
        plt.close('all')