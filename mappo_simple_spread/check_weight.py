import torch

# .pth 파일 불러오기
# 'cuda'에서 저장된 모델을 'cpu'에서 불러오기 위해 map_location='cpu' 옵션 사용
checkpoint = torch.load('./models/mappo_simple_spread_best.pth', map_location=torch.device('cpu'))

# 1. 파일에 어떤 종류의 데이터가 저장되어 있는지 최상위 키(key) 확인
print("--- 체크포인트의 최상위 키 ---")
print(checkpoint.keys())
# 출력 예시: dict_keys(['policy_state_dict', 'optimizer_state_dict', ...])
print("-" * 30)


# 2. 정책 네트워크(Actor-Critic)의 구조 확인
# 'policy_state_dict'에 저장된 모델의 파라미터를 가져옴
policy_state_dict = checkpoint['policy_state_dict']

print("\n--- 정책 네트워크의 레이어와 파라미터 크기 ---")
for layer_name, params in policy_state_dict.items():
    # 각 레이어의 이름과 파라미터 텐서의 크기(shape)를 출력
    print(f"레이어: {layer_name.ljust(25)} | 크기: {params.shape}")
print("-" * 30)


# 3. 특정 레이어의 실제 값 확인 (선택 사항)
# 너무 많은 숫자가 출력될 수 있으니 참고용으로만 보세요.
print("\n--- 'actor_head.weight' 레이어의 실제 파라미터 값 (일부) ---")
print(policy_state_dict['actor_head.weight'])
print("-" * 30)