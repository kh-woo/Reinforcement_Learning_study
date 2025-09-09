"""
MAPPO Simple Spread 환경 설정
"""
import torch

# 하이퍼파라미터
LEARNING_RATE = 0.0005
GAMMA = 0.98
LAMBDA = 0.95
EPS_CLIP = 0.2
K_EPOCHS = 2      # 4에서 2로 줄임 (GPU 사용량 감소)
T_HORIZON = 10    # 20에서 10으로 줄임 (메모리 사용량 감소)
NUM_AGENTS = 3
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5

# 환경 설정
ENV_NAME = "simple_spread_v3"
MAX_EPISODES = 10000
PRINT_INTERVAL = 20

# 시각화 설정
RENDER_MODE = "human"
VISUALIZATION_ENABLED = False # 시각화 없이 빠른 학습
# VISUALIZATION_ENABLED = True  # 학습 시에도 이미지 보기
ANIMATION_SPEED = 0.05

# 네트워크 설정
HIDDEN_DIM = 128  # 256에서 128로 줄임 (GPU 사용량 감소)
SHARED_LAYERS = 2

# 디바이스 설정
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"