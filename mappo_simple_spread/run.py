#!/usr/bin/env python3
"""
MAPPO Simple Spread 실행 스크립트
"""
import sys
import os

# 현재 디렉토리를 Python 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

if __name__ == "__main__":
    from train_mappo import main, test_trained_model
    import argparse
    
    parser = argparse.ArgumentParser(description="MAPPO Simple Spread 실행")
    parser.add_argument("--mode", choices=["train", "test"], default="train", 
                       help="실행 모드: train(학습) 또는 test(테스트)")
    parser.add_argument("--model", type=str, 
                       default="models/mappo_simple_spread_final.pth", 
                       help="테스트할 모델 경로")
    
    args = parser.parse_args()
    
    print("MAPPO Simple Spread")
    print("=" * 30)
    
    if args.mode == "train":
        print("학습 모드")
        main()
    elif args.mode == "test":
        print("테스트 모드")
        test_trained_model(args.model)