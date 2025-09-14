"""
MAPPO Simple Spread 환경 설정
"""
import torch

# 하이퍼파라미터
LEARNING_RATE = 0.0003
GAMMA = 0.98
LAMBDA = 0.95
EPS_CLIP = 0.2
K_EPOCHS = 4      # 더 안정적인 업데이트
T_HORIZON = 10    # 20에서 10으로 줄임 (메모리 사용량 감소)
NUM_AGENTS = 3
ENTROPY_COEF = 0.02
VALUE_COEF = 0.5
VALUE_LOSS_COEF = VALUE_COEF  # alias for compatibility with agent code

# 환경 설정
ENV_NAME = "simple_spread_v3"
MAX_CYCLES = 100  # 에피소드 최대 스텝 수 축소
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
GRADIENT_CLIP_NORM = 0.5  # gradient clipping norm for stability
ENTROPY_DECAY = 0.995     # decay factor for entropy coefficient

# 디바이스 설정
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 보상 처리
REWARD_CLIP = 1   # 보상 클리핑 활성화 (-10~10)

# 모델 저장 설정
SAVE_FORMAT = "pth"  # 기본은 pth로 안정 저장. "csv"로 변경 시 CSV 저장
MODEL_SAVE_INTERVAL = 1000

# 학습 기록 CSV 저장
STATS_CSV_PATH = "logs/training_log.csv"

# 엔트로피 감쇠(느리게)
ENTROPY_DECAY = 0.998

# PPO Rollout 길이
T_HORIZON = 32

# 보상 정규화 사용 여부
REWARD_NORMALIZE = True