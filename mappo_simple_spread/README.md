# MAPPO Simple Spread

Multi-Agent Proximal Policy Optimization (MAPPO)을 사용한 Simple Spread 환경 구현

##  프로젝트 구조

```
mappo_simple_spread/
├── config.py                          # 설정 파일
├── train_mappo.py                     # 메인 학습 스크립트
├── requirements.txt                   # 의존성 패키지
├── README.md                          # 프로젝트 설명
├── environments/
│   └── simple_spread_env.py          # Simple Spread 환경 래퍼
├── agents/
│   └── mappo_agent.py                # MAPPO 에이전트 구현
├── visualization/
│   └── simple_spread_visualizer.py   # 시각화 클래스
└── utils/
    └── logger.py                      # 로깅 및 통계 관리
```

##  설치 및 실행

### 1. 의존성 설치
```bash
pip install -r requirements.txt
```

### 2. 학습 실행
```bash
python train_mappo.py --mode train
```

### 3. 학습된 모델 테스트
```bash
python train_mappo.py --mode test --model models/mappo_simple_spread_final.pth
```

##  주요 특징

### 진짜 MAPPO 구현
- **공유 정책**: 모든 에이전트가 같은 정책 네트워크 사용
- **상호작용**: 에이전트들이 같은 환경에서 협력/경쟁
- **통합 학습**: 모든 에이전트의 경험을 통합하여 학습

### Simple Spread 환경
- **목표**: 각 에이전트가 서로 다른 랜드마크에 도달
- **협력**: 에이전트들이 서로 방해하지 않도록 학습
- **상호작용**: 다른 에이전트의 행동이 보상에 영향

### 시각화
- **실시간 애니메이션**: 학습 과정을 실시간으로 시각화
- **상태 정보**: 각 에이전트의 위치, 속도, 보상 표시
- **학습 곡선**: 학습 진행 상황을 그래프로 표시

##  설정

`config.py`에서 하이퍼파라미터를 조정할 수 있습니다:

```python
# 학습 설정
LEARNING_RATE = 0.0005
GAMMA = 0.98
EPS_CLIP = 0.2
K_EPOCHS = 4
T_HORIZON = 20

# 환경 설정
NUM_AGENTS = 3
MAX_EPISODES = 10000
```

##  결과

학습이 완료되면 다음 파일들이 생성됩니다:
- `models/mappo_simple_spread_final.pth`: 최종 모델
- `training_curves.png`: 학습 곡선 그래프
- `logs/`: 학습 로그 파일들

## 🔧 문제 해결

### PettingZoo 설치 오류
```bash
pip install pettingzoo[mpe]
```

### CUDA 사용 (GPU 학습)
```python
# config.py에서
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

##  성능 모니터링

학습 중 다음 정보를 확인할 수 있습니다:
- 에피소드별 총 보상
- 에이전트별 개별 보상
- 최고 성능 기록
- 학습 곡선 (이동 평균)

##  환경 설명

Simple Spread는 Multi-Agent 환경으로:
- **에이전트**: 3개의 에이전트
- **목표**: 각 에이전트가 서로 다른 랜드마크에 도달
- **보상**: 랜드마크 도달 시 보상, 충돌 시 페널티
- **관찰**: 에이전트 위치, 속도, 다른 에이전트 위치, 랜드마크 위치

이 구현은 진짜 MAPPO의 특성을 보여주며, 에이전트들이 협력적으로 학습하는 과정을 시각화할 수 있습니다.
