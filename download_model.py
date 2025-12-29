#!/usr/bin/env python3
"""
모델 다운로드 스크립트
로컬에 모델을 다운로드하여 Docker 이미지에 포함시킵니다.
"""
import os
import sys

# UnicodeEncodeError 방지
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

from huggingface_hub import hf_hub_download

MODEL_REPO_ID = "bartowski/Llama-3.2-3B-Instruct-GGUF"
MODEL_FILENAME = "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
MODELS_DIR = "./models"

def download_model():
    """모델을 로컬에 다운로드"""
    print(f"모델 다운로드 중: {MODEL_REPO_ID}/{MODEL_FILENAME}")
    print(f"저장 위치: {MODELS_DIR}")
    
    # models 디렉토리 생성
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    try:
        # local_dir을 사용하면 파일이 local_dir/filename 형태로 저장됨
        model_path = hf_hub_download(
            repo_id=MODEL_REPO_ID,
            filename=MODEL_FILENAME,
            local_dir=MODELS_DIR,
            local_dir_use_symlinks=False  # 실제 파일로 다운로드
        )
        
        # 실제 파일 경로 확인
        expected_path = os.path.join(MODELS_DIR, MODEL_FILENAME)
        if os.path.exists(expected_path):
            print(f"✓ 모델 다운로드 완료: {expected_path}")
            return expected_path
        elif os.path.exists(model_path):
            print(f"✓ 모델 다운로드 완료: {model_path}")
            return model_path
        else:
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
    except Exception as e:
        print(f"✗ 모델 다운로드 실패: {e}")
        raise

if __name__ == "__main__":
    download_model()

