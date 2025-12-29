#!/usr/bin/env python3
"""
도커 컨테이너 벤치마크 스크립트
로컬 도커 컨테이너 또는 클라우드 서버에 HTTP 요청을 보내서 벤치마킹합니다.
"""
import json
import sys
import requests
import time
import os
from typing import Optional, List, Dict

# UnicodeEncodeError 방지를 위해 stdout 인코딩 설정
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

# 기본 URL (로컬 도커 컨테이너)
DEFAULT_URL = "http://localhost:8080"

# 시스템 프롬프트 (로컬 벤치마크와 동일)
SYSTEM_PROMPT = "Answer in about 10 words or less."

# 테스트 질문
QUESTIONS = [
    "who are you?",
    "What is Python?",
    "How does AI work?",
    "Tell me about machine learning.",
    "What is the capital of France?"
]

def test_benchmark_endpoint(url: str, questions: List[str]) -> Optional[Dict]:
    """벤치마크 엔드포인트 테스트"""
    print(f"벤치마크 질문 수: {len(questions)}")
    print(f"시스템 프롬프트: {SYSTEM_PROMPT}")
    print("-" * 80)
    
    try:
        print("벤치마크 요청 전송 중...")
        start_time = time.time()
        response = requests.post(
            f"{url}/benchmark",
            headers={"Content-Type": "application/json"},
            json={
                "questions": questions,
                "system_prompt": SYSTEM_PROMPT
            },
            timeout=600
        )
        elapsed_time = time.time() - start_time
        
        response.raise_for_status()
        
        data = response.json()
        
        # 결과 출력
        print(f"\n✓ 벤치마크 완료 (클라이언트 측 시간: {elapsed_time:.3f}초)")
        print(f"✓ 총 질문 수: {data.get('total_questions', 0)}")
        print(f"✓ 성공: {data.get('successful', 0)}")
        print(f"✓ 실패: {data.get('failed', 0)}")
        print(f"✓ 서버 측 총 시간: {data.get('total_time', 0):.3f}초")
        print(f"✓ 평균 응답 시간: {data.get('avg_time', 0):.3f}초")
        print(f"✓ 총 토큰 수: {data.get('total_tokens', 0)}")
        print(f"✓ 처리 속도: {data.get('throughput', 0):.2f} 질문/초")
        print()
        
        # 각 질문별 결과 출력
        results = data.get("results", [])
        for i, result in enumerate(results, 1):
            if result.get("success", False):
                print(f"[질문 {i}] {result.get('question', 'N/A')}")
                print(f"  응답: {result.get('response', 'N/A')[:100]}...")
                print(f"  응답 시간: {result.get('response_time', 0):.3f}초")
                print(f"  토큰 수: {result.get('tokens_count', 0)}")
            else:
                print(f"[질문 {i}] {result.get('question', 'N/A')} - 실패")
                print(f"  오류: {result.get('error', 'Unknown error')}")
            print()
        
        return data
        
    except requests.exceptions.Timeout:
        print("✗ 타임아웃: 서버가 응답하는 데 너무 오래 걸렸습니다.")
        return {"success": False, "error": "timeout"}
    except requests.exceptions.RequestException as e:
        print(f"✗ 벤치마크 엔드포인트 테스트 실패: {e}")
        if hasattr(e, 'response') and e.response is not None:
            status_code = e.response.status_code
            print(f"  Response status: {status_code}")
            try:
                print(f"  Response body: {e.response.text[:200]}")
            except:
                pass
            
            # 404 에러인 경우 특별한 안내
            if status_code == 404:
                print("\n  ⚠ 404 에러: /benchmark 엔드포인트를 찾을 수 없습니다.")
                print("  가능한 원인:")
                print("    1. 도커 컨테이너가 실행되지 않았습니다.")
                print("    2. 서버 코드에 /benchmark 엔드포인트가 없습니다.")
                print("\n  해결 방법:")
                print("    npm run docker:run 또는 npm run docker:run:detach로 도커를 실행하세요.")
        return {"success": False, "error": str(e)}
    except Exception as e:
        print(f"✗ 예상치 못한 오류: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

def test_health_endpoint(url: str) -> bool:
    """Health 체크 엔드포인트 테스트"""
    try:
        response = requests.get(f"{url}/health", timeout=10)
        response.raise_for_status()
        data = response.json()
        print(f"✓ Health 체크 성공: {data.get('status', 'unknown')}")
        return data.get('status') == 'healthy'
    except Exception as e:
        print(f"✗ Health 체크 실패: {e}")
        return False

def main():
    """메인 벤치마크 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="도커 컨테이너 벤치마크 스크립트")
    parser.add_argument("--url", type=str, default=None,
                       help=f"테스트할 서버 URL (기본값: {DEFAULT_URL})")
    parser.add_argument("--question", type=str, default=None,
                       help="단일 질문 테스트 (지정하지 않으면 모든 질문 테스트)")
    
    args = parser.parse_args()
    
    # URL 설정
    url = args.url or os.getenv("BENCHMARK_URL", DEFAULT_URL)
    if url.endswith("/"):
        url = url[:-1]
    
    print("\n" + "=" * 80)
    print("도커 컨테이너 벤치마크")
    print("=" * 80)
    print(f"테스트 URL: {url}")
    print(f"시스템 프롬프트: {SYSTEM_PROMPT}")
    print("=" * 80)
    print()
    
    # Health 체크
    print("Health 체크 중...")
    if not test_health_endpoint(url):
        print("\n⚠ 서버가 준비되지 않았습니다. 계속 진행합니다...")
    print()
    
    # 질문 선택
    if args.question:
        questions = [args.question]
        print(f"단일 질문 테스트: {args.question}")
    else:
        questions = QUESTIONS
        print(f"전체 질문 테스트: {len(questions)}개")
    
    print()
    
    # 벤치마크 엔드포인트 호출
    result = test_benchmark_endpoint(url=url, questions=questions)
    
    # 결과 요약
    print("\n" + "=" * 80)
    print("벤치마크 완료")
    print("=" * 80)
    
    if result and result.get('success', True):
        print(f"\n✓ 벤치마크 성공!")
        sys.exit(0)
    else:
        print(f"\n✗ 벤치마크 실패")
        if result:
            print(f"  오류: {result.get('error', 'Unknown error')}")
        sys.exit(1)

if __name__ == "__main__":
    main()

