"""
MAPPO Simple Spread 학습 메인 스크립트 (Multi-Agent 안정화 기법 포함)
"""
import sys
import os
import time
import numpy as np

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import *
from environments.simple_spread_env import SimpleSpreadMultiAgentEnv
from agents.mappo_agent import MAPPOAgent
from visualization.simple_spread_visualizer import SimpleSpreadVisualizer
from utils.logger import TrainingLogger

def main():
    """메인 학습 함수"""
    print("MAPPO Simple Spread 학습 시작!")
    print(f"에이전트 수: {NUM_AGENTS}")
    print(f"최대 에피소드: {MAX_EPISODES}")
    print(f"환경: {ENV_NAME}")
    print("=" * 50)
    
    # 환경 초기화
    try:
        env = SimpleSpreadMultiAgentEnv(num_agents=NUM_AGENTS)
        print("Simple Spread 환경 초기화 완료")
    except Exception as e:
        print(f"환경 초기화 실패: {e}")
        print("PettingZoo가 설치되어 있는지 확인하세요: pip install pettingzoo[mpe]")
        return
    
    # 에이전트 초기화
    agent = MAPPOAgent(
        input_dim=env.get_observation_space().shape[0],
        action_dim=env.get_action_space().n,
        num_agents=NUM_AGENTS
    )
    print("MAPPO 에이전트 초기화 완료")
    
    # 시각화 초기화
    visualizer = SimpleSpreadVisualizer(num_agents=NUM_AGENTS) if VISUALIZATION_ENABLED else None
    if visualizer:
        print("시각화 초기화 완료")
    
    # 로거 초기화
    logger = TrainingLogger(num_agents=NUM_AGENTS)
    print("로거 초기화 완료")
    
    # 학습 루프
    print("\n학습 시작!")
    print("=" * 50)
    
    # 안정화 모니터링 변수
    best_reward = float('-inf')
    patience = 0
    max_patience = 1000  # 1000 에피소드 동안 개선이 없으면 조기 종료
    
    for episode in range(MAX_EPISODES):
        # 에피소드 초기화
        agent.clear_storage()
        observations, infos = env.reset()
        
        episode_rewards = [0.0 for _ in range(NUM_AGENTS)]
        episode_length = 0
        
        # 에피소드 실행
        for step in range(T_HORIZON):
            # 모든 에이전트의 행동 계산
            actions = []
            log_probs = []
            values = []
            
            for agent_id in range(NUM_AGENTS):
                action, log_prob, value = agent.get_action(observations[agent_id], agent_id)
                actions.append(action)
                log_probs.append(log_prob)
                values.append(value)
            
            # 환경에서 한 스텝 실행
            next_observations, rewards, dones, truncations, infos = env.step(actions)
            
            # 시각화 업데이트
            if visualizer and episode % 10 == 0:  # 10 에피소드마다 시각화
                visualizer.update(observations, rewards, dones)
                time.sleep(ANIMATION_SPEED)
            
            # 경험 저장
            for agent_id in range(NUM_AGENTS):
                agent.store_experience(
                    agent_id, observations[agent_id], actions[agent_id],
                    rewards[agent_id], dones[agent_id],
                    log_probs[agent_id], values[agent_id]
                )
                episode_rewards[agent_id] += rewards[agent_id]
            
            observations = next_observations
            episode_length += 1
            
            # 에피소드 종료 확인
            if all(dones) or all(truncations):
                break
        
        # 정책 업데이트
        agent.update()
        
        # 로깅
        logger.log_episode(episode_rewards, episode_length, episode)
        
        # 안정화 모니터링
        total_reward = sum(episode_rewards)
        if total_reward > best_reward:
            best_reward = total_reward
            patience = 0
            # 최고 성능 모델 저장
            best_model_path = "models/mappo_simple_spread_best.pth"
            os.makedirs("models", exist_ok=True)
            agent.save_model(best_model_path)
        else:
            patience += 1
        
        # 통계 출력
        if episode % PRINT_INTERVAL == 0:
            stats = logger.get_statistics()
            print(f"\nEpisode {episode}")
            print(f"  Total Reward: {total_reward:.2f}")
            print(f"  Best Reward: {best_reward:.2f}")
            print(f"  Avg Reward (last 100): {stats.get('avg_reward', 0):.2f}")
            print(f"  Patience: {patience}/{max_patience}")
            print(f"  Entropy Coef: {agent.entropy_coef:.4f}")
            print("-" * 50)
        
        # 조기 종료 (성능이 개선되지 않으면)
        if patience >= max_patience:
            print(f"\n조기 종료: {max_patience} 에피소드 동안 개선 없음")
            break
        
        # 모델 저장 (주기적으로)
        if episode % 1000 == 0 and episode > 0:
            model_path = f"models/mappo_simple_spread_episode_{episode}.pth"
            os.makedirs("models", exist_ok=True)
            agent.save_model(model_path)
            print(f"모델 저장: {model_path}")
    
    # 학습 완료
    print("\n학습 완료!")
    
    # 최종 통계
    final_stats = logger.get_statistics()
    print(f"최고 보상: {final_stats.get('best_reward', 0):.2f} (에피소드 {final_stats.get('best_episode', 0)})")
    print(f"평균 보상: {final_stats.get('avg_reward', 0):.2f}")
    
    # 학습 곡선 플롯
    logger.plot_training_curves("training_curves.png")
    
    # 최종 모델 저장
    final_model_path = "models/mappo_simple_spread_final.pth"
    agent.save_model(final_model_path)
    print(f"최종 모델 저장: {final_model_path}")
    
    # 환경 정리
    env.close()
    if visualizer:
        visualizer.close()
    
    print("모든 작업 완료!")

def test_trained_model(model_path="models/mappo_simple_spread_final.pth"):
    """학습된 모델 테스트"""
    print(f"학습된 모델 테스트: {model_path}")
    
    # 환경 초기화
    env = SimpleSpreadMultiAgentEnv(num_agents=NUM_AGENTS)
    
    # 에이전트 초기화
    agent = MAPPOAgent(
        input_dim=env.get_observation_space().shape[0],
        action_dim=env.get_action_space().n,
        num_agents=NUM_AGENTS
    )
    
    # 모델 로드
    try:
        agent.load_model(model_path)
        print("모델 로드 완료")
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        return
    
    # 시각화
    visualizer = SimpleSpreadVisualizer(num_agents=NUM_AGENTS)
    
    # 테스트 실행
    for episode in range(5):  # 5 에피소드 테스트
        observations, _ = env.reset()
        total_reward = 0
        
        for step in range(100):  # 최대 100 스텝
            actions = []
            for agent_id in range(NUM_AGENTS):
                action = agent.get_action_deterministic(observations[agent_id], agent_id)
                actions.append(action)
            
            next_observations, rewards, dones, truncations, _ = env.step(actions)
            
            # 시각화
            visualizer.update(observations, rewards, dones)
            time.sleep(ANIMATION_SPEED)
            
            total_reward += sum(rewards)
            observations = next_observations
            
            if all(dones) or all(truncations):
                break
        
        print(f"테스트 에피소드 {episode + 1}: 총 보상 = {total_reward:.2f}")
    
    # 정리
    env.close()
    visualizer.close()
    print("테스트 완료!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MAPPO Simple Spread")
    parser.add_argument("--mode", choices=["train", "test"], default="train", help="실행 모드")
    parser.add_argument("--model", type=str, default="models/mappo_simple_spread_final.pth", help="테스트할 모델 경로")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        main()
    elif args.mode == "test":
        test_trained_model(args.model)