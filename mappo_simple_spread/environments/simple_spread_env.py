"""
Simple Spread 환경을 위한 Multi-Agent 래퍼
"""
import numpy as np
import gymnasium as gym
from pettingzoo.mpe import simple_spread_v3
from pettingzoo.utils import parallel_to_aec, aec_to_parallel
import torch
from config import MAX_CYCLES, REWARD_CLIP

class SimpleSpreadMultiAgentEnv:
    """
    Simple Spread 환경을 MAPPO에 맞게 래핑
    """
    def __init__(self, num_agents=3, max_cycles=MAX_CYCLES):
        self.num_agents = num_agents
        self.max_cycles = max_cycles
        
        # PettingZoo 환경 생성
        self.parallel_env = simple_spread_v3.parallel_env(
            N=num_agents, 
            local_ratio=0.5,  # penalty와 목표 도달 보상 모두 활성화
            max_cycles=max_cycles,
            continuous_actions=False
        )
        
        # 환경 정보
        self.observation_space = self.parallel_env.observation_space("agent_0")
        self.action_space = self.parallel_env.action_space("agent_0")
        
        # 에이전트 이름 리스트
        self.agent_names = [f"agent_{i}" for i in range(num_agents)]
        
    def reset(self):
        """환경 초기화"""
        observations, infos = self.parallel_env.reset()
        
        # 에이전트 순서대로 관찰값 정렬
        obs_list = []
        info_list = []
        for agent_name in self.agent_names:
            obs_list.append(observations[agent_name])
            info_list.append(infos[agent_name])
        
        return obs_list, info_list
    
    def step(self, actions):
        """환경에서 한 스텝 실행"""
        # 액션을 딕셔너리 형태로 변환
        action_dict = {}
        for i, action in enumerate(actions):
            action_dict[self.agent_names[i]] = action
        
        # 환경 실행
        observations, rewards, terminations, truncations, infos = self.parallel_env.step(action_dict)
        
        # 디버깅: 처음 10에피소드만 출력 (에피소드 번호는 외부에서 전달받지 못하므로 step별로 1000번까지만 출력)
        if hasattr(self, 'debug_count'):
            self.debug_count += 1
        else:
            self.debug_count = 1
        if self.debug_count <= 1000:
            print(f"[ENV DEBUG] rewards: {rewards}, terminations: {terminations}, truncations: {truncations}, infos: {infos}")

        # 결과를 리스트로 변환
        obs_list = []
        reward_list = []
        done_list = []
        truncation_list = []
        info_list = []
        
        for agent_name in self.agent_names:
            obs_list.append(observations[agent_name])
            r = rewards[agent_name]
            if REWARD_CLIP is not None:
                r = float(np.clip(r, -REWARD_CLIP, REWARD_CLIP))
            reward_list.append(r)
            done_list.append(terminations[agent_name])
            truncation_list.append(truncations[agent_name])
            info_list.append(infos[agent_name])
        
        return obs_list, reward_list, done_list, truncation_list, info_list
    
    def render(self):
        """환경 렌더링"""
        return self.parallel_env.render()
    
    def close(self):
        """환경 종료"""
        self.parallel_env.close()
    
    def get_agent_names(self):
        """에이전트 이름 리스트 반환"""
        return self.agent_names
    
    def get_observation_space(self):
        """관찰 공간 반환"""
        return self.observation_space
    
    def get_action_space(self):
        """액션 공간 반환"""
        return self.action_space
