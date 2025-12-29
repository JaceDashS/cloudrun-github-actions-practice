#!/usr/bin/env python3
"""
로컬 벤치마크 및 채팅 스크립트
서버 없이 직접 모델을 로드하여 성능을 측정하거나 채팅할 수 있습니다.
"""
import os
import sys
import time
import gc
from typing import Optional, Dict, List

# UnicodeEncodeError 방지
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
MODELS = {
    "llama32b": {
        "name": "Llama-3.2-3B-Instruct-Q4_K_M",
        "repo_id": "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "filename": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "chat_format": "llama-3",
        "system_prompt": "Answer in about 10 words or less."
    },
    "tinllama": {
        "name": "TinyLlama-1.1B-Chat-Q4_K_M",
        "repo_id": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "filename": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "chat_format": "chatml",  # TinyLlama은 chatml 포맷 사용
        "system_prompt": "You are a helpful assistant."
    },
    "tinllama4q": {
        "name": "TinyLlama-1.1B-Chat-Q4_0",
        "repo_id": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "filename": "tinyllama-1.1b-chat-v1.0.Q4_0.gguf",
        "chat_format": "chatml",  # TinyLlama은 chatml 포맷 사용
        "system_prompt": "You are a helpful assistant."
    }
}

# 테스트 설정
QUESTION = "who are you?"

def get_model_path(repo_id: str, filename: str) -> str:
    """모델 경로를 가져오거나 다운로드 (런타임)"""
    cache_dir = os.getenv('HF_CACHE_DIR', os.path.expanduser('~/.cache/huggingface/hub'))
    print(f"모델 확인 중: {repo_id}/{filename}")
    
    try:
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir
        )
        print(f"✓ 모델 경로: {model_path}")
        return model_path
    except Exception as e:
        print(f"✗ 모델 다운로드 실패: {e}")
        raise

def format_prompt(model_key: str, question: str, system_prompt: str = None) -> str:
    """모델에 맞는 프롬프트 포맷 생성"""
    model_config = MODELS[model_key]
    if system_prompt is None:
        system_prompt = model_config["system_prompt"]
    
    chat_format = model_config["chat_format"]
    
    if chat_format == "llama-3":
        return f"<|system|>\n{system_prompt}<|end|>\n<|user|>\n{question}<|end|>\n<|assistant|>\n"
    elif chat_format == "chatml":
        return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    else:
        return f"{system_prompt}\n\nUser: {question}\nAssistant: "

def run_benchmark_with_model(model_key: str, model: Llama, embedding: bool = False, n_threads: int = 4) -> Dict:
    """이미 로드된 모델로 벤치마크 실행"""
    model_config = MODELS[model_key]
    embedding_str = "활성화" if embedding else "비활성화"
    print(f"\n{'='*60}")
    print(f"벤치마크 실행 - {model_config['name']}")
    print(f"임베딩: {embedding_str}, 스레드: {n_threads}")
    print(f"{'='*60}")
    
    try:
        # 테스트 실행
        print(f"[1/2] 테스트 실행 중...")
        print(f"  질문: {QUESTION}")
        print(f"  시스템 프롬프트: {model_config['system_prompt']}")
        
        prompt = format_prompt(model_key, QUESTION)
        test_start = time.time()
        output = model(
            prompt,
            max_tokens=20,
            temperature=0.7,
            top_p=0.9,
            echo=False,
            stop=["<|end|>", "<|im_end|>", "\n\n"]
        )
        test_time = time.time() - test_start
        
        # 응답 추출
        if hasattr(output, 'choices'):
            response = output.choices[0].text.strip()
        else:
            response = output['choices'][0]['text'].strip()
        
        print(f"✓ 응답: {response}")
        print(f"✓ 응답 시간: {test_time:.3f}초")
        
        # 임베딩 테스트 (활성화된 경우)
        embedding_info = None
        if embedding:
            print(f"[2/2] 임베딩 테스트 중...")
            try:
                response_tokens = model.tokenize(response.encode('utf-8'))
                if response_tokens:
                    first_token_str = model.detokenize([response_tokens[0]]).decode('utf-8', errors='replace')
                    first_token_emb = model.embed(first_token_str)
                    
                    if hasattr(first_token_emb, 'tolist'):
                        emb_list = first_token_emb.tolist()
                    elif isinstance(first_token_emb, list):
                        emb_list = first_token_emb
                    else:
                        emb_list = list(first_token_emb)
                    
                    dim = len(emb_list)
                    # 앞 3개만 샘플로 저장 (평탄화)
                    if isinstance(emb_list, list) and len(emb_list) > 0 and isinstance(emb_list[0], list):
                        # 중첩 리스트인 경우 평탄화
                        flat_list = [item for sublist in emb_list for item in (sublist if isinstance(sublist, list) else [sublist])]
                        sample = flat_list[:3]
                    else:
                        sample = emb_list[:3] if len(emb_list) >= 3 else emb_list
                    
                    embedding_info = {
                        "token": first_token_str,
                        "dim": dim,
                        "sample": sample  # 앞 3개만
                    }
                    print(f"✓ 임베딩 추출 완료: {first_token_str} (dim={dim})")
            except Exception as e:
                print(f"⚠ 임베딩 추출 실패: {e}")
        else:
            print(f"[2/2] 임베딩 테스트 건너뜀 (비활성화)")
        
        result = {
            "model_key": model_key,
            "model_name": model_config["name"],
            "embedding": embedding,
            "n_threads": n_threads,
            "load_time": 0.0,  # 이미 로드되어 있으므로 0
            "response_time": test_time,
            "response": response,
            "embedding_info": embedding_info
        }
        
        return result
        
    except Exception as e:
        print(f"✗ 벤치마크 실패: {e}")
        import traceback
        traceback.print_exc()
        return None

def multi_model_chat(model_keys: List[str], embedding: bool = False, n_threads: int = 4, system_prompt: str = None):
    """여러 모델을 동시에 로드하여 한 번의 입력에 대해 모든 모델이 답변하는 채팅 모드"""
    if system_prompt is None:
        system_prompt = "Please respond in about 10 words or less."
    
    print(f"\n{'='*80}")
    print(f"  멀티 모델 채팅 모드")
    print(f"{'='*80}")
    print(f"  모델: {', '.join([MODELS[key]['name'] for key in model_keys])}")
    print(f"  임베딩: {'활성화' if embedding else '비활성화'}, 스레드: {n_threads}")
    print(f"\n  📌 시스템 프롬프트 (모든 모델에 적용):")
    print(f"     \"{system_prompt}\"")
    print(f"{'='*80}")
    print("  종료하려면 'quit', 'exit', 또는 'q'를 입력하세요.")
    print("-"*80)
    
    models = {}
    try:
        # 모든 모델 로드
        print("\n모든 모델 로드 중...")
        for model_key in model_keys:
            model_config = MODELS[model_key]
            print(f"  [{model_config['name']}] 로드 중...")
            model_path = get_model_path(model_config["repo_id"], model_config["filename"])
            models[model_key] = Llama(
                model_path=model_path,
                n_ctx=4096,
                n_threads=n_threads,
                n_gpu_layers=0,
                chat_format=model_config["chat_format"],
                embedding=embedding,
                verbose=False
            )
            print(f"  ✓ [{model_config['name']}] 로드 완료")
        print("\n✓ 모든 모델 로드 완료!\n")
        
        while True:
            try:
                # 사용자 입력
                user_input = input("\nYou: ").strip()
                
                if not user_input:
                    continue
                
                # 종료 명령
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n채팅을 종료합니다.")
                    break
                
                # 모든 모델에 대해 답변 생성
                print(f"\n{'='*80}")
                print(f"  질문: {user_input}")
                print(f"  시스템 프롬프트: \"{system_prompt}\"")
                print(f"{'='*80}")
                for model_key in model_keys:
                    model_config = MODELS[model_key]
                    model = models[model_key]
                    
                    print(f"\n  [{model_config['name']}]: ", end="", flush=True)
                    prompt = format_prompt(model_key, user_input, system_prompt)
                    start_time = time.time()
                    
                    output = model(
                        prompt,
                        max_tokens=100,
                        temperature=0.7,
                        top_p=0.9,
                        echo=False,
                        stop=["<|end|>", "<|im_end|>", "\n\n"]
                    )
                    
                    elapsed = time.time() - start_time
                    
                    if hasattr(output, 'choices'):
                        response = output.choices[0].text.strip()
                    else:
                        response = output['choices'][0]['text'].strip()
                    
                    print(response)
                    print(f"    (응답 시간: {elapsed:.3f}초)")
                print(f"{'='*80}")
                
            except KeyboardInterrupt:
                print("\n\n채팅을 종료합니다.")
                break
            except EOFError:
                print("\n\n채팅을 종료합니다.")
                break
            except Exception as e:
                print(f"\n⚠ 오류 발생: {e}")
                import traceback
                traceback.print_exc()
        
    except Exception as e:
        print(f"✗ 멀티 모델 채팅 실패: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # 모든 모델 언로드
        print("\n모든 모델 언로드 중...")
        for model_key, model in models.items():
            del model
        gc.collect()
        print("✓ 정리 완료")

def interactive_chat(model_key: str, embedding: bool = False, n_threads: int = 4, system_prompt: str = None):
    """인터랙티브 채팅 모드"""
    model_config = MODELS[model_key]
    # 채팅 모드에서는 모든 모델에 대해 통일된 시스템 프롬프트 사용
    if system_prompt is None:
        system_prompt = "Please respond in about 10 words or less."
    
    print(f"\n{'='*60}")
    print(f"인터랙티브 채팅 모드 - {model_config['name']}")
    print(f"임베딩: {'활성화' if embedding else '비활성화'}, 스레드: {n_threads}")
    print(f"시스템 프롬프트: {system_prompt}")
    print(f"{'='*60}")
    print("종료하려면 'quit', 'exit', 또는 'q'를 입력하세요.")
    print("임베딩을 보려면 'embed <텍스트>'를 입력하세요.")
    print("-"*60)
    
    model = None
    try:
        # 모델 로드 (런타임 다운로드)
        print("\n모델 준비 중...")
        model_path = get_model_path(model_config["repo_id"], model_config["filename"])
        
        print("모델 로드 중...")
        model = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_threads=n_threads,
            n_gpu_layers=0,
            chat_format=model_config["chat_format"],
            embedding=embedding,
            verbose=False
        )
        print("✓ 모델 로드 완료!\n")
        
        conversation_history = []
        
        while True:
            try:
                # 사용자 입력
                user_input = input("\nYou: ").strip()
                
                if not user_input:
                    continue
                
                # 종료 명령
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n채팅을 종료합니다.")
                    break
                
                # 임베딩 보기 명령
                if user_input.lower().startswith('embed '):
                    text = user_input[6:].strip()
                    if text and embedding:
                        try:
                            emb = model.embed(text)
                            if hasattr(emb, 'tolist'):
                                emb_list = emb.tolist()
                            elif isinstance(emb, list):
                                emb_list = emb
                            else:
                                emb_list = list(emb)
                            
                            dim = len(emb_list)
                            print(f"\n임베딩 정보:")
                            print(f"  텍스트: {text}")
                            print(f"  차원: {dim}")
                            print(f"  샘플 (앞 3개): {emb_list[:3]}")
                        except Exception as e:
                            print(f"⚠ 임베딩 추출 실패: {e}")
                    elif not embedding:
                        print("⚠ 임베딩이 비활성화되어 있습니다.")
                    continue
                
                # 대화 생성
                prompt = format_prompt(model_key, user_input, system_prompt)
                
                print("Assistant: ", end="", flush=True)
                start_time = time.time()
                
                output = model(
                    prompt,
                    max_tokens=100,
                    temperature=0.7,
                    top_p=0.9,
                    echo=False,
                    stop=["<|end|>", "<|im_end|>", "\n\n"]
                )
                
                elapsed = time.time() - start_time
                
                if hasattr(output, 'choices'):
                    response = output.choices[0].text.strip()
                else:
                    response = output['choices'][0]['text'].strip()
                
                print(response)
                print(f"  (응답 시간: {elapsed:.3f}초)")
                
                conversation_history.append({"user": user_input, "assistant": response})
                
            except KeyboardInterrupt:
                print("\n\n채팅을 종료합니다.")
                break
            except EOFError:
                print("\n\n채팅을 종료합니다.")
                break
            except Exception as e:
                print(f"\n⚠ 오류 발생: {e}")
                import traceback
                traceback.print_exc()
        
    except Exception as e:
        print(f"✗ 채팅 모드 실패: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        if model is not None:
            print("\n모델 언로드 중...")
            del model
            gc.collect()
            print("✓ 정리 완료")

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="로컬 벤치마크 및 채팅 스크립트")
    parser.add_argument("--model", choices=list(MODELS.keys()) + ["all"], default="all",
                       help="사용할 모델 (기본값: all, 모든 모델 테스트)")
    parser.add_argument("--mode", choices=["benchmark", "chat"], default="benchmark",
                       help="실행 모드 (기본값: benchmark)")
    parser.add_argument("--embedding", action="store_true",
                       help="임베딩 활성화")
    parser.add_argument("--threads", type=int, default=None,
                       help="스레드 수 (기본값: 환경변수 LLAMA_N_THREADS 또는 4)")
    
    args = parser.parse_args()
    
    # 스레드 수 설정
    if args.threads:
        n_threads = args.threads
    else:
        try:
            n_threads = int(os.getenv('LLAMA_N_THREADS', '4'))
        except:
            n_threads = 4
    
    if args.mode == "chat":
        # 채팅 모드 - 모든 모델 또는 선택한 모델
        if args.model == "all":
            models_to_chat = list(MODELS.keys())
            print(f"\n모든 모델에 대해 채팅 모드 실행: {', '.join(models_to_chat)}")
        elif args.model == "tinllama":
            # tinllama 선택 시 두 TinyLlama 모델 모두
            models_to_chat = ["tinllama", "tinllama4q"]
            print(f"\nTinyLlama 모델 채팅 모드 실행: {', '.join(models_to_chat)}")
        else:
            models_to_chat = [args.model]
        
        # 모든 모델에 대해 통일된 시스템 프롬프트 사용
        unified_system_prompt = "Please respond in about 10 words or less."
        
        # 여러 모델인 경우 멀티 모델 채팅, 단일 모델인 경우 단일 채팅
        if len(models_to_chat) > 1:
            multi_model_chat(models_to_chat, embedding=args.embedding, n_threads=n_threads, system_prompt=unified_system_prompt)
        else:
            interactive_chat(models_to_chat[0], embedding=args.embedding, n_threads=n_threads, system_prompt=unified_system_prompt)
    else:
        # 벤치마크 모드
        print("="*60)
        print("로컬 벤치마크")
        print("="*60)
        
        # 모든 모델 벤치마크 또는 선택한 모델만
        if args.model == "all":
            models_to_test = list(MODELS.keys())
            print(f"\n모든 모델 벤치마크 실행: {', '.join(models_to_test)}")
        elif args.model == "tinllama":
            # tinllama 선택 시 두 TinyLlama 모델 모두 테스트
            models_to_test = ["tinllama", "tinllama4q"]
            print(f"\nTinyLlama 모델 벤치마크 실행: {', '.join(models_to_test)}")
        else:
            models_to_test = [args.model]
        
        # 설정 출력
        print(f"\n설정:")
        print(f"  스레드 수: {n_threads} (환경변수 LLAMA_N_THREADS로 변경 가능)")
        print(f"  질문: {QUESTION}")
        
        # 모든 모델을 먼저 로드 (임베딩 비활성화 버전)
        print(f"\n{'='*60}")
        print("1단계: 모든 모델 로드 (임베딩 비활성화)")
        print(f"{'='*60}")
        models_no_emb = {}
        load_times_no_emb = {}
        
        for model_key in models_to_test:
            model_config = MODELS[model_key]
            print(f"\n[{model_config['name']}] 로드 중...")
            model_path = get_model_path(model_config["repo_id"], model_config["filename"])
            
            load_start = time.time()
            models_no_emb[model_key] = Llama(
                model_path=model_path,
                n_ctx=4096,
                n_threads=n_threads,
                n_gpu_layers=0,
                chat_format=model_config["chat_format"],
                embedding=False,
                verbose=False
            )
            load_time = time.time() - load_start
            load_times_no_emb[model_key] = load_time
            print(f"✓ [{model_config['name']}] 로드 완료 ({load_time:.2f}초)")
        
        # 모든 모델을 먼저 로드 (임베딩 활성화 버전)
        print(f"\n{'='*60}")
        print("2단계: 모든 모델 로드 (임베딩 활성화)")
        print(f"{'='*60}")
        models_with_emb = {}
        load_times_with_emb = {}
        
        for model_key in models_to_test:
            model_config = MODELS[model_key]
            print(f"\n[{model_config['name']}] 로드 중...")
            model_path = get_model_path(model_config["repo_id"], model_config["filename"])
            
            load_start = time.time()
            models_with_emb[model_key] = Llama(
                model_path=model_path,
                n_ctx=4096,
                n_threads=n_threads,
                n_gpu_layers=0,
                chat_format=model_config["chat_format"],
                embedding=True,
                verbose=False
            )
            load_time = time.time() - load_start
            load_times_with_emb[model_key] = load_time
            print(f"✓ [{model_config['name']}] 로드 완료 ({load_time:.2f}초)")
        
        # 모든 모델 테스트 실행
        print(f"\n{'='*60}")
        print("3단계: 모든 모델 테스트 실행")
        print(f"{'='*60}")
        all_results = []
        
        for model_key in models_to_test:
            model_config = MODELS[model_key]
            print(f"\n{'='*60}")
            print(f"모델: {model_config['name']}")
            print(f"{'='*60}")
            
            model_results = []
            
            # 임베딩 비활성화 테스트
            print(f"\n[{model_config['name']}] 테스트 1: 임베딩 비활성화")
            result1 = run_benchmark_with_model(model_key, models_no_emb[model_key], embedding=False, n_threads=n_threads)
            if result1:
                result1['load_time'] = load_times_no_emb[model_key]  # 실제 로드 시간 설정
                model_results.append(result1)
            
            # 임베딩 활성화 테스트
            print(f"\n[{model_config['name']}] 테스트 2: 임베딩 활성화")
            result2 = run_benchmark_with_model(model_key, models_with_emb[model_key], embedding=True, n_threads=n_threads)
            if result2:
                result2['load_time'] = load_times_with_emb[model_key]  # 실제 로드 시간 설정
                model_results.append(result2)
            
            # 모델별 요약 표시
            if len(model_results) == 2:
                all_results.append({
                    "model": model_key,
                    "results": model_results
                })
                
                # 모델별 요약 출력
                no_emb = model_results[0]
                with_emb = model_results[1]
                
                print(f"\n\n{'='*80}")
                print(f"  📊 [{model_config['name']}] 벤치마크 결과 요약")
                print(f"{'='*80}")
                print(f"  스레드 수: {no_emb['n_threads']}")
                print(f"{'='*80}")
                
                print(f"\n  {'항목':<30} {'임베딩 비활성화':<25} {'임베딩 활성화':<25}")
                print("  " + "-" * 80)
                print(f"  {'모델 로드 시간 (초)':<30} {no_emb['load_time']:<25.3f} {with_emb['load_time']:<25.3f}")
                print(f"  {'응답 생성 시간 (초)':<30} {no_emb['response_time']:<25.3f} {with_emb['response_time']:<25.3f}")
                print(f"  {'총 시간 (초)':<30} {no_emb['load_time'] + no_emb['response_time']:<25.3f} {with_emb['load_time'] + with_emb['response_time']:<25.3f}")
                
                print(f"\n  📝 응답 결과:")
                print(f"     • 임베딩 비활성화: {no_emb['response']}")
                print(f"     • 임베딩 활성화:   {with_emb['response']}")
                
                # 성능 차이 계산
                if no_emb['response_time'] > 0:
                    overhead = ((with_emb['response_time'] - no_emb['response_time']) / no_emb['response_time']) * 100
                    load_overhead = ((with_emb['load_time'] - no_emb['load_time']) / no_emb['load_time']) * 100 if no_emb['load_time'] > 0 else 0
                    print(f"\n  ⚡ 성능 분석:")
                    print(f"     • 응답 시간 오버헤드: {overhead:+.2f}%")
                    print(f"     • 로드 시간 오버헤드: {load_overhead:+.2f}%")
                    if overhead > 0:
                        print(f"     • 임베딩 활성화 시 응답이 {overhead:.1f}% 느려짐")
                    else:
                        print(f"     • 임베딩 활성화 시 응답이 {abs(overhead):.1f}% 빨라짐")
                
                # 임베딩 정보 (간소화 - 앞 3개만)
                if with_emb.get('embedding_info'):
                    emb_info = with_emb['embedding_info']
                    sample = emb_info.get('sample', [])
                    # 정확히 앞 3개 값만 표시
                    if sample:
                        if isinstance(sample, list):
                            # 중첩 리스트인 경우 평탄화
                            if len(sample) > 0 and isinstance(sample[0], list):
                                flat = [item for sublist in sample for item in (sublist if isinstance(sublist, list) else [sublist])]
                                sample_display = flat[:3]
                            else:
                                sample_display = sample[:3]
                        else:
                            sample_display = [sample]
                        sample_str = f"[{', '.join(f'{x:.4f}' for x in sample_display[:3])}]"
                    else:
                        sample_str = "N/A"
                    print(f"\n  🔢 임베딩 정보:")
                    print(f"     • 토큰: {emb_info['token']}")
                    print(f"     • 차원: {emb_info['dim']}")
                    print(f"     • 샘플 (앞 3개): {sample_str}")
                
                print(f"\n{'='*80}\n")
            
            if len(model_results) == 2:
                all_results.append({
                    "model": model_key,
                    "results": model_results
                })
        
        # 전체 비교 (여러 모델이 있는 경우만)
        if len(all_results) > 1:
            
            print(f"\n\n{'='*80}")
            print("  📈 모델 간 성능 비교 (임베딩 비활성화 기준)")
            print(f"{'='*80}")
            print(f"\n  {'모델':<35} {'로드 시간 (초)':<18} {'응답 시간 (초)':<18} {'총 시간 (초)':<18}")
            print("  " + "-" * 89)
            for model_data in all_results:
                model_key = model_data["model"]
                model_config = MODELS[model_key]
                no_emb = model_data["results"][0]
                total_time = no_emb['load_time'] + no_emb['response_time']
                print(f"  {model_config['name']:<35} {no_emb['load_time']:<18.3f} {no_emb['response_time']:<18.3f} {total_time:<18.3f}")
            
            # 가장 빠른 모델 찾기 (임베딩 비활성화)
            fastest_model = min(all_results, key=lambda x: x["results"][0]['load_time'] + x["results"][0]['response_time'])
            fastest_name = MODELS[fastest_model["model"]]["name"]
            fastest_time = fastest_model["results"][0]['load_time'] + fastest_model["results"][0]['response_time']
            print(f"\n  🏆 가장 빠른 모델 (임베딩 비활성화): {fastest_name} (총 {fastest_time:.3f}초)")
            
            # 임베딩 활성화 기준 비교표
            print(f"\n\n{'='*80}")
            print("  📈 모델 간 성능 비교 (임베딩 활성화 기준)")
            print(f"{'='*80}")
            print(f"\n  {'모델':<35} {'로드 시간 (초)':<18} {'응답 시간 (초)':<18} {'총 시간 (초)':<18}")
            print("  " + "-" * 89)
            for model_data in all_results:
                model_key = model_data["model"]
                model_config = MODELS[model_key]
                with_emb = model_data["results"][1]  # 임베딩 활성화 결과
                total_time = with_emb['load_time'] + with_emb['response_time']
                print(f"  {model_config['name']:<35} {with_emb['load_time']:<18.3f} {with_emb['response_time']:<18.3f} {total_time:<18.3f}")
            
            # 가장 빠른 모델 찾기 (임베딩 활성화)
            fastest_model_emb = min(all_results, key=lambda x: x["results"][1]['load_time'] + x["results"][1]['response_time'])
            fastest_name_emb = MODELS[fastest_model_emb["model"]]["name"]
            fastest_time_emb = fastest_model_emb["results"][1]['load_time'] + fastest_model_emb["results"][1]['response_time']
            print(f"\n  🏆 가장 빠른 모델 (임베딩 활성화): {fastest_name_emb} (총 {fastest_time_emb:.3f}초)")
            print(f"\n{'='*80}\n")
        
        print(f"{'='*80}")
        print("  ✅ 벤치마크 완료!")
        print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
