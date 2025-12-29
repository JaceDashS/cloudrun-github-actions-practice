"""
헬스체크 서버
FastAPI를 사용하여 구현
llama-cpp-python을 사용한 LLaMA 모델 서빙
"""
import os
import sys
import json
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, Response, status, HTTPException
from pydantic import BaseModel
from typing import Optional

try:
    from llama_cpp import Llama
except ImportError:
    print("Warning: llama-cpp-python is not installed. Install it with: pip install llama-cpp-python", flush=True)
    Llama = None

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("Warning: huggingface-hub is not installed. Install it with: pip install huggingface-hub", flush=True)
    hf_hub_download = None

# 전역 변수로 모델 저장
llama_model = None
model_info = {
    "name": None,
    "path": None,
    "repo_id": None,
    "filename": None,
    "source": None  # "build-time" 또는 "runtime-download"
}

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 50
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9

class EmbeddingRequest(BaseModel):
    input_text: str

class TokenEmbedding(BaseModel):
    token: str
    embedding: list  # 앞 3개만 표시, 나머지는 ...
    dim: int

class EmbeddingResponse(BaseModel):
    response: str  # 모델이 생성한 실제 응답
    tokens: list[TokenEmbedding]  # 토큰별 임베딩 리스트

class BenchmarkRequest(BaseModel):
    questions: list[str]  # 벤치마크할 질문 리스트
    system_prompt: Optional[str] = "You are a helpful assistant. Please respond in about 10 sentences or less."

class BenchmarkResult(BaseModel):
    question: str
    response: str
    response_time: float
    tokens_count: int
    success: bool
    error: Optional[str] = None

class BenchmarkResponse(BaseModel):
    total_questions: int
    successful: int
    failed: int
    total_time: float
    avg_time: float
    total_tokens: int
    throughput: float  # 질문/초
    results: list[BenchmarkResult]

# 라이프스팬 상태 관리
app_state = {
    "started": False,
    "ready": False,
    "shutting_down": False,
}

# 서비스 시작 시간 기록
START_TIME = datetime.utcnow()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작 및 종료 이벤트 처리"""
    global llama_model, model_info
    
    # Startup
    port = int(os.getenv('PORT', '8080'))
    host = os.getenv('HOST', '0.0.0.0')
    
    print(f"\n{'='*60}", flush=True)
    print("LLaMA.cpp Server Starting...", flush=True)
    print(f"Version: 2.3.3", flush=True)
    print(f"Host: {host}", flush=True)
    print(f"Port: {port}", flush=True)
    print(f"{'='*60}\n", flush=True)
    
    app_state["started"] = True
    
    # 모델 파일 경로 확인 및 다운로드
    model_path = os.getenv('MODEL_PATH', None)
    hf_model_id = os.getenv('HF_MODEL_ID', None)
    hf_filename = os.getenv('HF_FILENAME', None)
    
    # MODEL_PATH가 지정되고 파일이 존재하면 로컬 파일 사용
    if model_path and os.path.exists(model_path):
        print(f"ℹ Using local model file: {model_path}", flush=True)
        model_info["path"] = model_path
        model_info["name"] = Path(model_path).stem
        # 빌드 타임에 포함된 모델인지 확인 (/app/models/ 경로에 있으면 빌드 타임)
        if model_path.startswith("/app/models/"):
            model_info["source"] = "build-time"
            print(f"  ✓ Model is from build-time (included in Docker image)", flush=True)
        else:
            model_info["source"] = "local-file"
            print(f"  ℹ Model is from local file (not from build-time)", flush=True)
        # 다운로드 스킵
        hf_model_id = None
    # 환경변수가 없으면 기본 모델 사용 (LLaMA 3.2-3B-Instruct-Q4_K_M) - 빌드타임 기본값
    elif not model_path and not hf_model_id:
        print("ℹ No MODEL_PATH or HF_MODEL_ID specified, using default model (Llama-3.2-3B-Instruct-Q4_K_M)", flush=True)
        hf_model_id = "bartowski/Llama-3.2-3B-Instruct-GGUF"
        hf_filename = "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
        print(f"  Default model: {hf_model_id}/{hf_filename}", flush=True)
    
    # 모델 정보 저장
    if hf_model_id:
        model_info["repo_id"] = hf_model_id
        model_info["filename"] = hf_filename
    
    # Hugging Face Hub에서 모델 다운로드 (로컬 파일이 없는 경우만)
    if hf_model_id and not model_path:
        if hf_hub_download is None:
            error_msg = "huggingface-hub is required for HF_MODEL_ID but not installed."
            print(f"✗ {error_msg}", flush=True)
            raise ImportError(error_msg)
        
        # HF_FILENAME이 없으면 기본값 사용
        if not hf_filename:
            if "llama-3.2-3b-instruct" in hf_model_id.lower():
                hf_filename = "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
                print(f"  Using default filename: {hf_filename}", flush=True)
            elif "tinllama" in hf_model_id.lower() or "tinyllama" in hf_model_id.lower():
                hf_filename = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
                print(f"  Using default filename: {hf_filename}", flush=True)
            else:
                error_msg = f"HF_FILENAME must be specified for model {hf_model_id}"
                print(f"✗ {error_msg}", flush=True)
                print(f"  Example: HF_FILENAME=model.q4_k_m.gguf", flush=True)
                raise ValueError(error_msg)
        
        print(f"Downloading model from Hugging Face Hub: {hf_model_id}/{hf_filename}", flush=True)
        try:
            cache_dir = os.getenv('HF_CACHE_DIR', '/tmp/models')
            model_path = hf_hub_download(
                repo_id=hf_model_id,
                filename=hf_filename,
                cache_dir=cache_dir
            )
            print(f"✓ Model downloaded to: {model_path}", flush=True)
            model_info["path"] = model_path
            model_info["source"] = "runtime-download"
            print(f"  ℹ Model was downloaded at runtime from Hugging Face Hub", flush=True)
            # 모델 이름 추출 (파일명에서 확장자 제거)
            if hf_filename:
                model_info["name"] = hf_filename.replace(".gguf", "")
        except Exception as e:
            error_msg = f"Failed to download model from Hugging Face Hub: {str(e)}"
            print(f"✗ {error_msg}", flush=True)
            raise RuntimeError(error_msg) from e
    
    # 로컬 파일 경로 확인 (다운로드한 경우)
    if model_path and not os.path.exists(model_path):
        error_msg = f"Model file not found at path: {model_path}"
        print(f"✗ {error_msg}", flush=True)
        raise FileNotFoundError(error_msg)
    
    # 모델 경로가 아직 설정되지 않았으면 에러
    if not model_path:
        error_msg = "Model path is not set. Please specify MODEL_PATH or HF_MODEL_ID."
        print(f"✗ {error_msg}", flush=True)
        raise ValueError(error_msg)
    
    # llama-cpp-python 설치 확인
    if Llama is None:
        error_msg = "llama-cpp-python is not installed. Install it with: pip install llama-cpp-python"
        print(f"✗ {error_msg}", flush=True)
        raise ImportError(error_msg)
    
    # 모델 로딩 (필수, 실패 시 서버 시작 중단)
    print(f"Loading LLaMA model from {model_path}...", flush=True)
    load_start_time = time.time()
    try:
        # 모델 타입에 따른 chat_format 결정
        chat_format = "llama-3"  # 기본값
        if hf_model_id:
            if "tinllama" in hf_model_id.lower() or "tinyllama" in hf_model_id.lower():
                chat_format = "chatml"
            elif "llama-3" in hf_model_id.lower() or "llama3" in hf_model_id.lower():
                chat_format = "llama-3"
        
        # LLaMA 모델 설정
        n_threads = int(os.getenv('LLAMA_N_THREADS', '1'))
        embedding_enabled = os.getenv('LLAMA_EMBEDDING', 'true').lower() == 'true'
        print(f"  Configuration: n_threads={n_threads}, n_ctx=4096, embedding={embedding_enabled}, chat_format={chat_format}", flush=True)
        llama_model = Llama(
            model_path=model_path,
            n_ctx=4096,  # 컨텍스트 크기
            n_threads=n_threads,  # 스레드 수 (환경변수로 설정 가능)
            n_gpu_layers=0,  # CPU 전용
            chat_format=chat_format,  # 모델에 맞는 채팅 포맷
            embedding=embedding_enabled,  # 임베딩 추출 활성화
            verbose=False
        )
        load_elapsed_time = time.time() - load_start_time
        
        # 모델 정보 업데이트
        model_info["path"] = model_path
        if not model_info["name"]:
            # 경로에서 파일명 추출
            model_name = Path(model_path).stem
            model_info["name"] = model_name
        
        # 모델 로드 완료 로그 (명확하게 표시)
        print(f"\n{'='*60}", flush=True)
        print("✓ MODEL LOADED SUCCESSFULLY", flush=True)
        print(f"{'='*60}", flush=True)
        print(f"  Model: {model_info['name']}", flush=True)
        print(f"  Path: {model_path}", flush=True)
        print(f"  Threads: {n_threads}", flush=True)
        print(f"  Context Size: 4096", flush=True)
        print(f"  Embedding: {'Enabled' if embedding_enabled else 'Disabled'}", flush=True)
        print(f"  Load Time: {load_elapsed_time:.2f}초", flush=True)
        print(f"{'='*60}", flush=True)
        print(f"\n✓ Server is ready", flush=True)
        print(f"Health Check: http://{host if host != '0.0.0.0' else 'localhost'}:{port}/health", flush=True)
        print(f"{'='*60}\n", flush=True)
        
        app_state["ready"] = True
    except Exception as e:
        error_msg = f"Failed to load model: {str(e)}"
        print(f"✗ {error_msg}", flush=True)
        raise RuntimeError(error_msg) from e
    
    yield
    
    # Shutdown
    print("Application shutdown: Cleaning up...", flush=True)
    app_state["shutting_down"] = True
    llama_model = None
    app_state["ready"] = False
    app_state["started"] = False
    print("Application shutdown: Complete", flush=True)


app = FastAPI(
    title="Cloud Run FastAPI Test",
    version="1.0.1",
    lifespan=lifespan
)


@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "service": "LLaMA.cpp Server",
        "version": "2.3.3",
        "status": "running",
        "model": model_info["name"] if llama_model is not None else None
    }


@app.get("/health")
async def health_check(response: Response):
    """
    헬스체크 엔드포인트
    - healthy: 200
    - unhealthy: 503
    """
    try:
        uptime_seconds = (datetime.utcnow() - START_TIME).total_seconds()
        
        is_healthy = app_state["ready"] and not app_state["shutting_down"] and llama_model is not None
        
        lifespan_status = (
            "shutting_down" if app_state["shutting_down"]
            else "ready" if app_state["ready"]
            else "starting" if app_state["started"]
            else "unknown"
        )
        
        model_status = {
            "loaded": llama_model is not None,
            "type": "llama-cpp-python" if llama_model is not None else None,
            "name": model_info["name"] if llama_model is not None else None,
            "path": model_info["path"] if llama_model is not None else None,
            "source": model_info["source"] if llama_model is not None else None,
            "repo_id": model_info["repo_id"] if llama_model is not None else None,
            "filename": model_info["filename"] if llama_model is not None else None
        }
        
        health_status = {
            "status": "healthy" if is_healthy else "unhealthy",
            "service": "LLaMA.cpp Server",
            "version": app.version,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "uptime_seconds": int(uptime_seconds),
            "python_version": sys.version.split()[0],
            "model": model_status,
            "lifespan": {
                "started": app_state["started"],
                "ready": app_state["ready"],
                "shutting_down": app_state["shutting_down"],
                "status": lifespan_status,
            },
        }
        
        if is_healthy:
            response.status_code = status.HTTP_200_OK
        else:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        
        return health_status
    
    except Exception as e:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "lifespan": {
                "started": app_state.get("started", False),
                "ready": app_state.get("ready", False),
                "shutting_down": app_state.get("shutting_down", False),
                "status": "error",
            },
        }


@app.get("/healthz")
async def healthz(response: Response):
    """
    표준 헬스체크 엔드포인트 (/healthz)
    """
    return await health_check(response)


@app.post("/generate")
def generate_text(request: GenerateRequest):
    """텍스트 생성 엔드포인트"""
    if llama_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Set MODEL_PATH or HF_MODEL_ID environment variable.")
    
    try:
        # LLaMA 모델로 텍스트 생성
        output = llama_model(
            request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            echo=False
        )
        
        # llama-cpp-python은 dict-like 객체를 반환
        if hasattr(output, 'choices'):
            generated_text = output.choices[0].text.strip()
        else:
            generated_text = output['choices'][0]['text'].strip()
        
        return {
            "prompt": request.prompt,
            "generated_text": generated_text,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")

@app.post("/completion")
def completion(request: GenerateRequest):
    """OpenAI 호환 completion 엔드포인트"""
    if llama_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Set MODEL_PATH or HF_MODEL_ID environment variable.")
    
    try:
        output = llama_model(
            request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            echo=False
        )
        
        # llama-cpp-python은 dict-like 객체를 반환
        if hasattr(output, 'choices'):
            generated_text = output.choices[0].text.strip()
        else:
            generated_text = output['choices'][0]['text'].strip()
        
        # usage 정보는 llama-cpp-python에서 제공하지 않을 수 있음
        usage_info = {}
        if hasattr(output, 'usage'):
            usage_info = {
                "prompt_tokens": getattr(output.usage, 'prompt_tokens', 0),
                "completion_tokens": getattr(output.usage, 'completion_tokens', 0),
                "total_tokens": getattr(output.usage, 'total_tokens', 0)
            }
        elif isinstance(output, dict) and 'usage' in output:
            usage_info = output['usage']
        else:
            usage_info = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        
        return {
            "id": "llama-cpp-completion",
            "object": "text_completion",
            "created": 0,
            "model": "llama-cpp",
            "choices": [
                {
                    "text": generated_text,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "stop"
                }
            ],
            "usage": usage_info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Completion error: {str(e)}")


@app.post("/embedding", response_model=EmbeddingResponse)
def get_embedding(request: EmbeddingRequest):
    """임베딩 벡터 추출 엔드포인트 - 토큰별 임베딩과 모델 응답 반환"""
    if llama_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please wait for model to load.")
    
    if not request.input_text or not request.input_text.strip():
        raise HTTPException(status_code=400, detail="input_text is required and cannot be empty")
    
    try:
        # 1. 모델로 텍스트 생성 (응답 생성)
        # 동일한 답변을 유도하기 위해 temperature를 낮추고 명확한 지시 추가
        print(f"[EMBEDDING] Generating response for: {request.input_text[:50]}...", flush=True)
        output = llama_model(
            request.input_text,
            max_tokens=50,
            temperature=0.1,  # 낮은 temperature로 일관성 있는 답변 유도
            top_p=0.9,
            echo=False,
            stop=["<|end|>", "\n\n"]
        )
        
        # 생성된 응답 텍스트 추출
        if hasattr(output, 'choices'):
            generated_text = output.choices[0].text.strip()
        else:
            generated_text = output['choices'][0]['text'].strip()
        
        print(f"[EMBEDDING] Generated response: {generated_text[:50]}...", flush=True)
        
        # 2. 입력 텍스트를 토큰화
        print(f"[EMBEDDING] Tokenizing input text...", flush=True)
        input_tokens = llama_model.tokenize(request.input_text.encode('utf-8'))
        input_token_strs = [llama_model.detokenize([t]).decode('utf-8', errors='replace') for t in input_tokens]
        
        # 빈 토큰 제거
        filtered_tokens = [(token_str, token_id) for token_str, token_id in zip(input_token_strs, input_tokens) if token_str.strip()]
        input_token_strs = [t for t, _ in filtered_tokens]
        
        print(f"[EMBEDDING] Filtered tokens: {len(input_token_strs)}", flush=True)
        
        # 3. gpt-visualizer 방식: llama.embed()를 바로 호출 (토큰별 임베딩 리스트 반환)
        print(f"[EMBEDDING] Extracting embeddings using gpt-visualizer method...", flush=True)
        token_embeddings = []
        
        try:
            # gpt-visualizer 방식: llama.embed()는 토큰별 임베딩 리스트를 반환
            input_embeddings = llama_model.embed(request.input_text)
            
            # gpt-visualizer처럼 토큰과 임베딩을 zip으로 묶기
            # input_embeddings는 토큰별 임베딩 리스트여야 함
            if isinstance(input_embeddings, list) and len(input_embeddings) > 0:
                # 토큰과 임베딩을 zip으로 묶기 (gpt-visualizer 방식)
                input_filtered = [(token_str, emb) for token_str, emb in zip(input_token_strs, input_embeddings) if token_str.strip()]
                filtered_token_strs = [t for t, _ in input_filtered]
                filtered_embeddings = [e for _, e in input_filtered]
                
                # 모든 토큰 처리
                if filtered_token_strs and filtered_embeddings:
                    # 각 토큰과 임베딩을 처리
                    for token_str, token_embedding in zip(filtered_token_strs, filtered_embeddings):
                        # numpy array일 수 있으므로 리스트로 변환
                        if hasattr(token_embedding, 'tolist'):
                            embedding_list = token_embedding.tolist()
                        elif isinstance(token_embedding, list):
                            embedding_list = token_embedding
                        else:
                            embedding_list = list(token_embedding)
                        
                        dim = len(embedding_list)
                        if dim > 3:
                            embedding_display = embedding_list[:3] + ["..."]
                        else:
                            embedding_display = embedding_list
                        
                        token_embeddings.append(TokenEmbedding(
                            token=token_str,
                            embedding=embedding_display,
                            dim=dim
                        ))
                    print(f"[EMBEDDING] Extracted {len(token_embeddings)} token embeddings using gpt-visualizer method", flush=True)
                else:
                    raise ValueError("No valid tokens or embeddings after filtering")
            else:
                # embed()가 리스트가 아닌 경우 (단일 벡터 반환)
                # 단일 벡터를 모든 토큰에 할당 (동일한 임베딩 사용)
                if input_token_strs:
                    if hasattr(input_embeddings, 'tolist'):
                        embedding_list = input_embeddings.tolist()
                    elif isinstance(input_embeddings, list):
                        embedding_list = input_embeddings
                    else:
                        embedding_list = list(input_embeddings)
                    
                    dim = len(embedding_list)
                    if dim > 3:
                        embedding_display = embedding_list[:3] + ["..."]
                    else:
                        embedding_display = embedding_list
                    
                    # 모든 토큰에 동일한 임베딩 할당
                    for token_str in input_token_strs:
                        token_embeddings.append(TokenEmbedding(
                            token=token_str,
                            embedding=embedding_display,
                            dim=dim
                        ))
                    print(f"[EMBEDDING] Extracted {len(token_embeddings)} token embeddings from single vector", flush=True)
                else:
                    raise ValueError("No tokens available")
        except Exception as e:
            print(f"[ERROR] Failed to extract token embeddings: {e}", flush=True)
            import traceback
            traceback.print_exc()
            # 최종 폴백: 각 토큰을 개별 처리
            print(f"[EMBEDDING] Falling back to individual token processing...", flush=True)
            for token_str in input_token_strs:
                token_embedding = llama_model.embed(token_str)
                
                if hasattr(token_embedding, 'tolist'):
                    embedding_list = token_embedding.tolist()
                elif isinstance(token_embedding, list):
                    embedding_list = token_embedding
                else:
                    embedding_list = list(token_embedding)
                
                dim = len(embedding_list)
                if dim > 3:
                    embedding_display = embedding_list[:3] + ["..."]
                else:
                    embedding_display = embedding_list
                
                token_embeddings.append(TokenEmbedding(
                    token=token_str,
                    embedding=embedding_display,
                    dim=dim
                ))
        
        print(f"[EMBEDDING] Extracted {len(token_embeddings)} input token embeddings", flush=True)
        
        # 4. 출력 텍스트(생성된 응답)의 토큰 임베딩도 추출
        print(f"[EMBEDDING] Extracting output embeddings...", flush=True)
        output_tokens = llama_model.tokenize(generated_text.encode('utf-8'))
        output_token_strs = [llama_model.detokenize([t]).decode('utf-8', errors='replace') for t in output_tokens]
        
        # 빈 토큰 제거
        filtered_output_tokens = [(token_str, token_id) for token_str, token_id in zip(output_token_strs, output_tokens) if token_str.strip()]
        output_token_strs = [t for t, _ in filtered_output_tokens]
        
        print(f"[EMBEDDING] Filtered output tokens: {len(output_token_strs)}", flush=True)
        
        # 출력 텍스트의 임베딩 추출
        try:
            output_embeddings = llama_model.embed(generated_text)
            
            if isinstance(output_embeddings, list) and len(output_embeddings) > 0:
                # 토큰과 임베딩을 zip으로 묶기
                output_filtered = [(token_str, emb) for token_str, emb in zip(output_token_strs, output_embeddings) if token_str.strip()]
                filtered_output_token_strs = [t for t, _ in output_filtered]
                filtered_output_embeddings = [e for _, e in output_filtered]
                
                # 각 출력 토큰과 임베딩을 처리
                for token_str, token_embedding in zip(filtered_output_token_strs, filtered_output_embeddings):
                    # numpy array일 수 있으므로 리스트로 변환
                    if hasattr(token_embedding, 'tolist'):
                        embedding_list = token_embedding.tolist()
                    elif isinstance(token_embedding, list):
                        embedding_list = token_embedding
                    else:
                        embedding_list = list(token_embedding)
                    
                    dim = len(embedding_list)
                    if dim > 3:
                        embedding_display = embedding_list[:3] + ["..."]
                    else:
                        embedding_display = embedding_list
                    
                    token_embeddings.append(TokenEmbedding(
                        token=token_str,
                        embedding=embedding_display,
                        dim=dim
                    ))
                print(f"[EMBEDDING] Extracted {len(filtered_output_token_strs)} output token embeddings", flush=True)
            else:
                # 단일 벡터인 경우 모든 출력 토큰에 동일한 임베딩 할당
                if hasattr(output_embeddings, 'tolist'):
                    embedding_list = output_embeddings.tolist()
                elif isinstance(output_embeddings, list):
                    embedding_list = output_embeddings
                else:
                    embedding_list = list(output_embeddings)
                
                dim = len(embedding_list)
                if dim > 3:
                    embedding_display = embedding_list[:3] + ["..."]
                else:
                    embedding_display = embedding_list
                
                for token_str in output_token_strs:
                    token_embeddings.append(TokenEmbedding(
                        token=token_str,
                        embedding=embedding_display,
                        dim=dim
                    ))
                print(f"[EMBEDDING] Extracted {len(output_token_strs)} output token embeddings from single vector", flush=True)
        except Exception as e:
            print(f"[WARNING] Failed to extract output embeddings: {e}", flush=True)
            # 출력 임베딩 추출 실패해도 입력 임베딩은 반환
        
        print(f"[EMBEDDING] Total token embeddings: {len(token_embeddings)} (input + output)", flush=True)
        
        # 모든 토큰 반환 (입력 + 출력)
        print(f"[EMBEDDING] Returning response and all token embeddings", flush=True)
        
        return EmbeddingResponse(
            response=generated_text,
            tokens=token_embeddings
        )
    except Exception as e:
        print(f"[ERROR] Embedding extraction failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Embedding extraction error: {str(e)}")


@app.post("/benchmark", response_model=BenchmarkResponse)
def run_benchmark(request: BenchmarkRequest):
    """벤치마크 엔드포인트 - 여러 질문을 처리하고 성능 지표를 반환"""
    print(f"[BENCHMARK] 벤치마크 요청 수신: {len(request.questions)}개 질문", flush=True)
    
    if llama_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please wait for model to load.")
    
    if not request.questions or len(request.questions) == 0:
        raise HTTPException(status_code=400, detail="questions list cannot be empty")
    
    system_prompt = request.system_prompt or "Answer in about 10 words or less."
    
    results = []
    total_start_time = time.time()
    print(f"[BENCHMARK] 벤치마크 시작: {len(request.questions)}개 질문 처리", flush=True)
    
    # 모델 타입에 따른 chat_format 결정
    chat_format = "llama-3"  # 기본값
    if model_info.get("repo_id"):
        repo_id_lower = model_info["repo_id"].lower()
        if "tinllama" in repo_id_lower or "tinyllama" in repo_id_lower:
            chat_format = "chatml"
        elif "llama-3" in repo_id_lower or "llama3" in repo_id_lower:
            chat_format = "llama-3"
    
    for i, question in enumerate(request.questions, 1):
        try:
            print(f"[BENCHMARK] 질문 {i}/{len(request.questions)} 처리 중: {question[:50]}...", flush=True)
            
            # 프롬프트 포맷팅
            if chat_format == "llama-3":
                full_prompt = f"<|system|>\n{system_prompt}<|end|>\n<|user|>\n{question}<|end|>\n<|assistant|>\n"
            elif chat_format == "chatml":
                full_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
            else:
                full_prompt = f"{system_prompt}\n\nUser: {question}\nAssistant: "
            
            # 응답 생성
            start_time = time.time()
            output = llama_model(
                full_prompt,
                max_tokens=100,
                temperature=0.7,
                top_p=0.9,
                echo=False,
                stop=["<|end|>", "<|im_end|>", "\n\n"]
            )
            elapsed_time = time.time() - start_time
            
            # 응답 추출
            if hasattr(output, 'choices'):
                response = output.choices[0].text.strip()
            else:
                response = output['choices'][0]['text'].strip()
            
            # 토큰 수 계산
            response_tokens = llama_model.tokenize(response.encode('utf-8'))
            tokens_count = len(response_tokens)
            
            print(f"[BENCHMARK] 질문 {i} 완료: 응답 시간 {elapsed_time:.3f}초, 토큰 수 {tokens_count}", flush=True)
            
            results.append(BenchmarkResult(
                question=question,
                response=response,
                response_time=elapsed_time,
                tokens_count=tokens_count,
                success=True
            ))
            
        except Exception as e:
            print(f"[BENCHMARK] Error processing question '{question}': {e}", flush=True)
            results.append(BenchmarkResult(
                question=question,
                response="",
                response_time=0.0,
                tokens_count=0,
                success=False,
                error=str(e)
            ))
    
    total_time = time.time() - total_start_time
    successful_results = [r for r in results if r.success]
    failed_results = [r for r in results if not r.success]
    
    total_tokens = sum(r.tokens_count for r in successful_results)
    avg_time = sum(r.response_time for r in successful_results) / len(successful_results) if successful_results else 0.0
    throughput = len(successful_results) / total_time if total_time > 0 else 0.0
    
    print(f"[BENCHMARK] 벤치마크 완료: 총 시간 {total_time:.3f}초, 성공 {len(successful_results)}/{len(request.questions)}", flush=True)
    
    return BenchmarkResponse(
        total_questions=len(request.questions),
        successful=len(successful_results),
        failed=len(failed_results),
        total_time=total_time,
        avg_time=avg_time,
        total_tokens=total_tokens,
        throughput=throughput,
        results=results
    )


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
