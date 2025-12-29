#!/usr/bin/env python3
"""
LLaMA 3.2B 4Q 모델 벤치마크 스크립트
임베딩 활성화/비활성화 두 가지 모드로 성능을 측정합니다.
"""
import os
import sys
import time
import gc
from typing import Optional, Dict, List

# UnicodeEncodeError 방지를 위해 stdout 인코딩 설정
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

try:
    from llama_cpp import Llama
except ImportError:
    print("✗ llama-cpp-python is not installed. Install it with: pip install llama-cpp-python")
    sys.exit(1)

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("✗ huggingface-hub is not installed. Install it with: pip install huggingface-hub")
    sys.exit(1)

# 모델 설정
MODEL_REPO_ID = "bartowski/Llama-3.2-3B-Instruct-GGUF"
MODEL_FILENAME = "Llama-3.2-3B-Instruct-Q4_K_M.gguf"

# 테스트 질문 및 시스템 프롬프트
QUESTION = "who are you?"
SYSTEM_PROMPT = "Answer in about 10 words or less."
FULL_PROMPT = f"<|system|>\n{SYSTEM_PROMPT}<|end|>\n<|user|>\n{QUESTION}<|end|>\n<|assistant|>\n"

def download_model(repo_id: str, filename: str) -> str:
    """모델을 다운로드하고 경로를 반환"""
    print(f"  다운로드 중: {repo_id}/{filename}")
    cache_dir = os.getenv('HF_CACHE_DIR', os.path.expanduser('~/.cache/huggingface/hub'))
    try:
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir
        )
        print(f"  ✓ 다운로드 완료: {model_path}")
        return model_path
    except Exception as e:
        print(f"  ✗ 다운로드 실패: {e}")
        raise

def load_model(model_path: str, n_threads: int = 4, embedding: bool = False) -> Llama:
    """모델을 로드하고 반환"""
    embedding_str = "활성화" if embedding else "비활성화"
    print(f"  모델 로드 중... (임베딩: {embedding_str})")
    try:
        model = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_threads=n_threads,
            n_gpu_layers=0,  # CPU 전용
            chat_format="llama-3",  # Llama-3 채팅 포맷
            embedding=embedding,  # 임베딩 기능 설정
            verbose=False
        )
        print(f"  ✓ 모델 로드 완료")
        return model
    except Exception as e:
        print(f"  ✗ 모델 로드 실패: {e}")
        raise

def unload_model(model: Optional[Llama]):
    """모델을 언로드하고 메모리 정리"""
    if model is not None:
        print(f"  모델 언로드 중...")
        del model
        gc.collect()
        print(f"  ✓ 모델 언로드 완료")

def test_model(model: Llama, prompt: str, test_embedding: bool = False) -> tuple[str, float, Optional[Dict]]:
    """모델에 질문하고 응답 시간을 측정, 첫 번째 토큰 임베딩도 반환"""
    try:
        start_time = time.time()
        output = model(
            prompt,
            max_tokens=20,  # 10단어 내외를 위해 토큰 수 제한
            temperature=0.7,
            top_p=0.9,
            echo=False,
            stop=["<|end|>", "\n\n"]
        )
        elapsed_time = time.time() - start_time
        
        # 응답 추출
        if hasattr(output, 'choices'):
            response = output.choices[0].text.strip()
        else:
            response = output['choices'][0]['text'].strip()
        
        # 첫 번째 토큰의 임베딩 추출 (임베딩 기능이 활성화된 경우에만)
        first_token_embedding = None
        if test_embedding:
            try:
                # 응답 텍스트를 토큰화
                response_tokens = model.tokenize(response.encode('utf-8'))
                if response_tokens:
                    # 첫 번째 토큰 문자열 추출
                    first_token_str = model.detokenize([response_tokens[0]]).decode('utf-8', errors='replace')
                    
                    # 첫 번째 토큰의 임베딩 추출
                    first_token_emb = model.embed(first_token_str)
                    
                    # numpy array를 리스트로 변환
                    if hasattr(first_token_emb, 'tolist'):
                        embedding_list = first_token_emb.tolist()
                    elif isinstance(first_token_emb, list):
                        embedding_list = first_token_emb
                    else:
                        embedding_list = list(first_token_emb)
                    
                    # 앞 3개만 표시
                    dim = len(embedding_list)
                    if dim > 3:
                        embedding_display = embedding_list[:3] + ["..."]
                    else:
                        embedding_display = embedding_list
                    
                    first_token_embedding = {
                        "token": first_token_str,
                        "embedding": embedding_display,
                        "dim": dim
                    }
            except Exception as e:
                print(f"  ⚠ 첫 번째 토큰 임베딩 추출 실패: {e}")
        
        return response, elapsed_time, first_token_embedding
    except Exception as e:
        print(f"  ✗ 질문 처리 실패: {e}")
        raise

def benchmark_model(n_threads: int = 4, embedding: bool = False) -> Optional[Dict]:
    """단일 모델에 대한 벤치마크 실행 - 모델을 로드, 테스트, 언로드"""
    embedding_str = "임베딩 활성화" if embedding else "임베딩 비활성화"
    
    print(f"\n{'=' * 70}")
    print(f"LLaMA 3.2B 4Q 모델 벤치마크 ({embedding_str})")
    print(f"{'=' * 70}")
    
    model = None
    try:
        # 1. 모델 다운로드
        print(f"[1/5] 모델 다운로드 중...")
        model_path = download_model(MODEL_REPO_ID, MODEL_FILENAME)
        
        # 2. 모델 로드
        print(f"[2/5] 모델 로드 중...")
        model = load_model(model_path, n_threads=n_threads, embedding=embedding)
        
        # 3. 질문 및 응답 시간 측정
        print(f"[3/5] 벤치마크 테스트 실행 중...")
        print(f"  질문: {QUESTION}")
        print(f"  시스템 프롬프트: {SYSTEM_PROMPT}")
        
        response, response_time, first_token_embedding = test_model(model, FULL_PROMPT, test_embedding=embedding)
        
        # 4. 결과 출력
        print(f"  ✓ 응답: {response}")
        print(f"  ✓ 응답 시간: {response_time:.3f}초")
        
        # 첫 번째 토큰 임베딩 출력 (임베딩이 활성화된 경우에만)
        if first_token_embedding:
            token = first_token_embedding["token"]
            embedding_data = first_token_embedding["embedding"]
            dim = first_token_embedding["dim"]
            embedding_str = str(embedding_data).replace("'", "")
            print(f"  ✓ 첫 번째 토큰 임베딩:")
            print(f"    {token}: {embedding_str} (dim={dim})")
        elif embedding:
            print(f"  ⚠ 임베딩이 활성화되었지만 추출 실패")
        
        # 결과 저장
        result = {
            "model_name": "Llama-3.2-3B-Instruct-Q4_K_M",
            "repo_id": MODEL_REPO_ID,
            "filename": MODEL_FILENAME,
            "embedding": embedding,
            "response": response,
            "response_time": response_time,
            "first_token_embedding": first_token_embedding
        }
        
        # 5. 모델 언로드
        print(f"[4/5] 모델 언로드 중...")
        unload_model(model)
        model = None
        
        # 메모리 정리 대기
        print(f"[5/5] 메모리 정리 중...")
        time.sleep(1)
        gc.collect()
        
        return result
        
    except Exception as e:
        print(f"  ✗ 벤치마크 실패: {e}")
        # 예외 발생 시에도 모델 언로드
        if model is not None:
            print(f"  모델 언로드 중...")
            unload_model(model)
            model = None
        return None
    finally:
        # finally 블록에서도 확실히 언로드
        if model is not None:
            print(f"  [정리] 모델 강제 언로드 중...")
            unload_model(model)
            model = None
            gc.collect()

def main():
    """메인 벤치마크 함수"""
    print("=" * 70)
    print("LLaMA 3.2B 4Q 모델 성능 벤치마크")
    print("=" * 70)
    print(f"질문: {QUESTION}")
    print(f"시스템 프롬프트: {SYSTEM_PROMPT}")
    print()
    
    # 스레드 수 설정
    try:
        n_threads = int(os.getenv('LLAMA_N_THREADS', '4'))
    except:
        n_threads = 4
    
    print(f"사용할 스레드 수: {n_threads}")
    print(f"(환경변수 LLAMA_N_THREADS로 변경 가능)")
    print()
    
    results: List[Dict] = []
    
    # 임베딩 비활성화 테스트
    print(f"\n[1/2] 임베딩 비활성화 모드")
    result = benchmark_model(n_threads=n_threads, embedding=False)
    if result:
        results.append(result)
    
    # 모델 간 대기 (메모리 정리 시간)
    print(f"\n다음 테스트 준비 중... (메모리 정리)")
    gc.collect()
    time.sleep(2)
    
    # 임베딩 활성화 테스트
    print(f"\n[2/2] 임베딩 활성화 모드")
    result = benchmark_model(n_threads=n_threads, embedding=True)
    if result:
        results.append(result)
    
    # 결과 요약
    print(f"\n{'=' * 70}")
    print("벤치마크 결과 요약")
    print(f"{'=' * 70}")
    
    if results:
        print(f"\n{'모드':<20} {'응답 시간 (초)':<20} {'응답':<40}")
        print("-" * 80)
        
        for result in results:
            embedding_status = "임베딩 활성화" if result.get('embedding', False) else "임베딩 비활성화"
            print(f"{embedding_status:<20} {result['response_time']:<20.3f} {result['response']:<40}")
        
        print(f"\n{'=' * 70}")
        print("상세 결과")
        print(f"{'=' * 70}")
        
        for result in results:
            embedding_status = "임베딩 활성화" if result.get('embedding', False) else "임베딩 비활성화"
            print(f"\n{embedding_status} ({result['response_time']:.3f}초)")
            print(f"  응답: {result['response']}")
            
            if result.get('first_token_embedding'):
                token = result['first_token_embedding']["token"]
                embedding_data = result['first_token_embedding']["embedding"]
                dim = result['first_token_embedding']["dim"]
                embedding_str = str(embedding_data).replace("'", "")
                print(f"  첫 번째 토큰 임베딩: {token}: {embedding_str} (dim={dim})")
        
        # 성능 비교
        if len(results) == 2:
            no_embedding = next((r for r in results if not r.get('embedding', False)), None)
            with_embedding = next((r for r in results if r.get('embedding', False)), None)
            
            if no_embedding and with_embedding:
                print(f"\n{'=' * 70}")
                print("성능 비교")
                print(f"{'=' * 70}")
                print(f"임베딩 비활성화: {no_embedding['response_time']:.3f}초")
                print(f"임베딩 활성화: {with_embedding['response_time']:.3f}초")
                if no_embedding['response_time'] > 0:
                    overhead = ((with_embedding['response_time'] - no_embedding['response_time']) / no_embedding['response_time']) * 100
                    print(f"임베딩 오버헤드: {overhead:+.2f}%")
    else:
        print("\n✗ 모든 테스트 실패")
    
    print(f"\n{'=' * 70}")
    print("벤치마크 완료!")
    print("=" * 70)

if __name__ == "__main__":
    main()

