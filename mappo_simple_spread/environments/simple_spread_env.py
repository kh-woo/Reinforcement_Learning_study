"""
Simple Spread 환경을 위한 Multi-Agent 래퍼
"""
import numpy as np
import gymnasium as gym
from pettingzoo.mpe import simple_spread_v3
from pettingzoo.utils import parallel_to_aec, aec_to_parallel
import torch

class SimpleSpreadMultiAgentEnv:
    """
    Simple Spread 환경을 MAPPO에 맞게 래핑
    """
    def __init__(self, num_agents=3, max_cycles=25):
        self.num_agents = num_agents
        self.max_cycles = max_cycles
        
        # PettingZoo 환경 생성
        self.parallel_env = simple_spread_v3.parallel_env(
            N=num_agents, 
            local_ratio=0.5, 
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
        
        # 결과를 리스트로 변환
        obs_list = []
        reward_list = []
        done_list = []
        truncation_list = []
        info_list = []
        
        for agent_name in self.agent_names:
            obs_list.append(observations[agent_name])
            reward_list.append(rewards[agent_name])
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
