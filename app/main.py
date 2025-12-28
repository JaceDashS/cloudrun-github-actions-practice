"""
헬스체크 서버
FastAPI를 사용하여 구현
"""
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import FastAPI, Response, status

# #region agent log (주석처리)
# DEBUG_LOG_PATH = r"c:\dev\workspace\hf-docker-space-cicd-test\.cursor\debug.log"
# def debug_log(session_id, run_id, hypothesis_id, location, message, data):
#     try:
#         with open(DEBUG_LOG_PATH, 'a', encoding='utf-8') as f:
#             log_entry = {
#                 "sessionId": session_id,
#                 "runId": run_id,
#                 "hypothesisId": hypothesis_id,
#                 "location": location,
#                 "message": message,
#                 "data": data,
#                 "timestamp": int(__import__('time').time() * 1000)
#             }
#             f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
#     except Exception:
#         pass
# #endregion

# #region 모델 로드 관련 (주석처리)
# try:
#     from llama_cpp import Llama
# except ImportError:
#     print("Warning: llama-cpp-python is not installed. Install it with: pip install llama-cpp-python", flush=True)
#     Llama = None

# try:
#     from huggingface_hub import hf_hub_download
# except ImportError:
#     print("Warning: huggingface-hub is not installed. Install it with: pip install huggingface-hub", flush=True)
#     hf_hub_download = None

# # 전역 변수로 모델 저장
# llama_model = None
# model_info = {
#     "name": None,
#     "path": None,
#     "repo_id": None,
#     "filename": None
# }
# #endregion

# #region 모델 관련 모델 클래스 (주석처리)
# class GenerateRequest(BaseModel):
#     prompt: str
#     max_tokens: Optional[int] = 50
#     temperature: Optional[float] = 0.7
#     top_p: Optional[float] = 0.9

# class EmbeddingRequest(BaseModel):
#     input_text: str

# class TokenEmbedding(BaseModel):
#     token: str
#     embedding: list  # 앞 3개만 표시, 나머지는 ...
#     dim: int

# class EmbeddingResponse(BaseModel):
#     response: str  # 모델이 생성한 실제 응답
#     tokens: list[TokenEmbedding]  # 토큰별 임베딩 리스트
# #endregion

# 라이프스팬 상태 관리
app_state = {
    "started": False,
    "ready": False,
    "shutting_down": False,
}

# 서비스 시작 시간 기록
START_TIME = datetime.utcnow()


async def initialize_app():
    """애플리케이션 초기화 작업"""
    pass


async def cleanup_app():
    """애플리케이션 정리 작업"""
    pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작 및 종료 이벤트 처리"""
    # Startup
    print("Application startup: Initializing...", flush=True)
    app_state["started"] = True
    
    await initialize_app()
    
    app_state["ready"] = True
    print("Application startup: Ready!", flush=True)
    
    yield
    
    # Shutdown
    print("Application shutdown: Cleaning up...", flush=True)
    app_state["shutting_down"] = True
    
    await cleanup_app()
    
    app_state["ready"] = False
    app_state["started"] = False
    print("Application shutdown: Complete", flush=True)

    # #region 모델 로드 관련 로직 (주석처리)
    # global llama_model, model_info
    
    # port = int(os.getenv('PORT', '7860'))
    # host = os.getenv('HOST', '0.0.0.0')
    
    # print(f"\n{'='*60}", flush=True)
    # print("LLaMA.cpp Server Starting...", flush=True)
    # print(f"Version: 2.3.3", flush=True)
    # print(f"Host: {host}", flush=True)
    # print(f"Port: {port}", flush=True)
    # print(f"{'='*60}\n", flush=True)
    
    # # 모델 파일 경로 확인 및 다운로드
    # model_path = os.getenv('MODEL_PATH', None)
    # hf_model_id = os.getenv('HF_MODEL_ID', None)
    # hf_filename = os.getenv('HF_FILENAME', None)
    
    # if not model_path and not hf_model_id:
    #     print("ℹ No MODEL_PATH or HF_MODEL_ID specified, using default model", flush=True)
    #     hf_model_id = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
    #     hf_filename = "qwen2.5-0.5b-instruct-q4_k_m.gguf"
    #     print(f"  Default model: {hf_model_id}/{hf_filename}", flush=True)
    
    # model_info["repo_id"] = hf_model_id
    # model_info["filename"] = hf_filename
    
    # # Hugging Face Hub에서 모델 다운로드 및 로드 로직...
    # # (전체 모델 로드 코드 주석처리)
    # #endregion


app = FastAPI(
    title="Cloud Run FastAPI Test",
    version="1.0.1",
    lifespan=lifespan
)


@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {"message": "Hello from Cloud Run!", "status": "running"}


@app.get("/health")
async def health_check(response: Response):
    """
    헬스체크 엔드포인트
    - healthy: 200
    - unhealthy: 503
    """
    try:
        uptime_seconds = (datetime.utcnow() - START_TIME).total_seconds()
        
        is_healthy = app_state["ready"] and not app_state["shutting_down"]
        
        lifespan_status = (
            "shutting_down" if app_state["shutting_down"]
            else "ready" if app_state["ready"]
            else "starting" if app_state["started"]
            else "unknown"
        )
        
        health_status = {
            "status": "healthy" if is_healthy else "unhealthy",
            "service": "fastapi-app",
            "version": app.version,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "uptime_seconds": int(uptime_seconds),
            "python_version": sys.version.split()[0],
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


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)

# @app.post("/generate")
# def generate_text(request: GenerateRequest):
#     """텍스트 생성 엔드포인트"""
#     if llama_model is None:
#         raise HTTPException(status_code=503, detail="Model not loaded. Set MODEL_PATH environment variable.")
    
#     try:
#         # LLaMA 모델로 텍스트 생성
#         output = llama_model(
#             request.prompt,
#             max_tokens=request.max_tokens,
#             temperature=request.temperature,
#             top_p=request.top_p,
#             echo=False
#         )
        
#         # llama-cpp-python은 dict-like 객체를 반환
#         if hasattr(output, 'choices'):
#             generated_text = output.choices[0].text.strip()
#         else:
#             generated_text = output['choices'][0]['text'].strip()
        
#         return {
#             "prompt": request.prompt,
#             "generated_text": generated_text,
#             "max_tokens": request.max_tokens,
#             "temperature": request.temperature,
#             "top_p": request.top_p
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")

# @app.post("/completion")
# def completion(request: GenerateRequest):
#     """OpenAI 호환 completion 엔드포인트"""
#     if llama_model is None:
#         raise HTTPException(status_code=503, detail="Model not loaded. Set MODEL_PATH environment variable.")
    
#     try:
#         output = llama_model(
#             request.prompt,
#             max_tokens=request.max_tokens,
#             temperature=request.temperature,
#             top_p=request.top_p,
#             echo=False
#         )
        
#         # llama-cpp-python은 dict-like 객체를 반환
#         if hasattr(output, 'choices'):
#             generated_text = output.choices[0].text.strip()
#         else:
#             generated_text = output['choices'][0]['text'].strip()
        
#         # usage 정보는 llama-cpp-python에서 제공하지 않을 수 있음
#         usage_info = {}
#         if hasattr(output, 'usage'):
#             usage_info = {
#                 "prompt_tokens": getattr(output.usage, 'prompt_tokens', 0),
#                 "completion_tokens": getattr(output.usage, 'completion_tokens', 0),
#                 "total_tokens": getattr(output.usage, 'total_tokens', 0)
#             }
#         elif isinstance(output, dict) and 'usage' in output:
#             usage_info = output['usage']
#         else:
#             usage_info = {
#                 "prompt_tokens": 0,
#                 "completion_tokens": 0,
#                 "total_tokens": 0
#             }
        
#         return {
#             "id": "llama-cpp-completion",
#             "object": "text_completion",
#             "created": 0,
#             "model": "llama-cpp",
#             "choices": [
#                 {
#                     "text": generated_text,
#                     "index": 0,
#                     "logprobs": None,
#                     "finish_reason": "stop"
#                 }
#             ],
#             "usage": usage_info
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Completion error: {str(e)}")


# @app.post("/embedding", response_model=EmbeddingResponse)
# def get_embedding(request: EmbeddingRequest):
#     """임베딩 벡터 추출 엔드포인트 - 토큰별 임베딩과 모델 응답 반환"""
#     if llama_model is None:
#         raise HTTPException(status_code=503, detail="Model not loaded. Please wait for model to load.")
    
#     if not request.input_text or not request.input_text.strip():
#         raise HTTPException(status_code=400, detail="input_text is required and cannot be empty")
    
#     try:
#         # 1. 모델로 텍스트 생성 (응답 생성)
#         # 동일한 답변을 유도하기 위해 temperature를 낮추고 명확한 지시 추가
#         print(f"[EMBEDDING] Generating response for: {request.input_text[:50]}...", flush=True)
#         output = llama_model(
#             request.input_text,
#             max_tokens=50,
#             temperature=0.1,  # 낮은 temperature로 일관성 있는 답변 유도
#             top_p=0.9,
#             echo=False,
#             stop=["\n"]
#         )
        
#         # 생성된 응답 텍스트 추출
#         if hasattr(output, 'choices'):
#             generated_text = output.choices[0].text.strip()
#         else:
#             generated_text = output['choices'][0]['text'].strip()
        
#         print(f"[EMBEDDING] Generated response: {generated_text[:50]}...", flush=True)
        
#         # 2. 입력 텍스트를 토큰화
#         print(f"[EMBEDDING] Tokenizing input text...", flush=True)
#         input_tokens = llama_model.tokenize(request.input_text.encode('utf-8'))
#         input_token_strs = [llama_model.detokenize([t]).decode('utf-8', errors='replace') for t in input_tokens]
        
#         # 빈 토큰 제거
#         filtered_tokens = [(token_str, token_id) for token_str, token_id in zip(input_token_strs, input_tokens) if token_str.strip()]
#         input_token_strs = [t for t, _ in filtered_tokens]
#         input_token_ids = [tid for _, tid in filtered_tokens]
        
#         print(f"[EMBEDDING] Filtered tokens: {len(input_token_strs)}", flush=True)
        
#         # 3. gpt-visualizer 방식: llama.embed()를 바로 호출 (토큰별 임베딩 리스트 반환)
#         print(f"[EMBEDDING] Extracting embeddings using gpt-visualizer method...", flush=True)
#         token_embeddings = []
        
#         # #region agent log
#         debug_log("debug-session", "run1", "A", "server/main.py:386", "Starting embedding extraction (gpt-visualizer style)", {
#             "token_count": len(input_token_strs)
#         })
#         # #endregion
        
#         try:
#             # gpt-visualizer 방식: llama.embed()는 토큰별 임베딩 리스트를 반환
#             # #region agent log
#             debug_log("debug-session", "run1", "A", "server/main.py:392", "Calling llama.embed() with full text", {
#                 "text_length": len(request.input_text)
#             })
#             # #endregion
#             input_embeddings = llama_model.embed(request.input_text)
#             # #region agent log
#             debug_log("debug-session", "run1", "A", "server/main.py:395", "llama.embed() returned", {
#                 "is_list": isinstance(input_embeddings, list),
#                 "length": len(input_embeddings) if hasattr(input_embeddings, '__len__') else "no length",
#                 "type": str(type(input_embeddings))
#             })
#             # #endregion
            
#             # gpt-visualizer처럼 토큰과 임베딩을 zip으로 묶기
#             # input_embeddings는 토큰별 임베딩 리스트여야 함
#             if isinstance(input_embeddings, list) and len(input_embeddings) > 0:
#                 # 토큰과 임베딩을 zip으로 묶기 (gpt-visualizer 방식)
#                 input_filtered = [(token_str, emb) for token_str, emb in zip(input_token_strs, input_embeddings) if token_str.strip()]
#                 filtered_token_strs = [t for t, _ in input_filtered]
#                 filtered_embeddings = [e for _, e in input_filtered]
                
#                 # #region agent log
#                 debug_log("debug-session", "run1", "A", "server/main.py:406", "Filtered tokens and embeddings", {
#                     "filtered_token_count": len(filtered_token_strs),
#                     "filtered_embedding_count": len(filtered_embeddings)
#                 })
#                 # #endregion
                
#                 # 모든 토큰 처리
#                 if filtered_token_strs and filtered_embeddings:
#                     # 각 토큰과 임베딩을 처리
#                     for token_str, token_embedding in zip(filtered_token_strs, filtered_embeddings):
#                         # numpy array일 수 있으므로 리스트로 변환
#                         if hasattr(token_embedding, 'tolist'):
#                             embedding_list = token_embedding.tolist()
#                         elif isinstance(token_embedding, list):
#                             embedding_list = token_embedding
#                         else:
#                             embedding_list = list(token_embedding)
                        
#                         dim = len(embedding_list)
#                         if dim > 3:
#                             embedding_display = embedding_list[:3] + ["..."]
#                         else:
#                             embedding_display = embedding_list
                        
#                         token_embeddings.append(TokenEmbedding(
#                             token=token_str,
#                             embedding=embedding_display,
#                             dim=dim
#                         ))
#                     print(f"[EMBEDDING] Extracted {len(token_embeddings)} token embeddings using gpt-visualizer method", flush=True)
#                 else:
#                     raise ValueError("No valid tokens or embeddings after filtering")
#             else:
#                 # embed()가 리스트가 아닌 경우 (단일 벡터 반환)
#                 # #region agent log
#                 debug_log("debug-session", "run1", "A", "server/main.py:432", "embed() returned non-list, treating as single vector", {})
#                 # #endregion
#                 # 단일 벡터를 모든 토큰에 할당 (동일한 임베딩 사용)
#                 if input_token_strs:
#                     if hasattr(input_embeddings, 'tolist'):
#                         embedding_list = input_embeddings.tolist()
#                     elif isinstance(input_embeddings, list):
#                         embedding_list = input_embeddings
#                     else:
#                         embedding_list = list(input_embeddings)
                    
#                     dim = len(embedding_list)
#                     if dim > 3:
#                         embedding_display = embedding_list[:3] + ["..."]
#                     else:
#                         embedding_display = embedding_list
                    
#                     # 모든 토큰에 동일한 임베딩 할당
#                     for token_str in input_token_strs:
#                         token_embeddings.append(TokenEmbedding(
#                             token=token_str,
#                             embedding=embedding_display,
#                             dim=dim
#                         ))
#                     print(f"[EMBEDDING] Extracted {len(token_embeddings)} token embeddings from single vector", flush=True)
#                 else:
#                     raise ValueError("No tokens available")
#         except Exception as e:
#             # 폴백: 기존 방식 (사용되지 않을 것으로 예상)
#             if False:  # gpt-visualizer 방식이 작동하지 않는 경우에만 사용
#                 # #region agent log
#                 debug_log("debug-session", "run1", "A", "server/main.py:465", "Skipping eval(), directly accessing _ctx", {})
#                 # #endregion
#                 print(f"[EMBEDDING] Skipping eval(), directly accessing internal context...", flush=True)
                
#                 # eval() 없이 직접 내부 상태 접근 시도
#                 # 내부 상태에서 각 토큰의 임베딩 추출 시도
#                 # llama-cpp-python의 내부 API를 사용
#                 if hasattr(llama_model, 'get_embeddings'):
#                     # #region agent log
#                     debug_log("debug-session", "run1", "B", "server/main.py:419", "get_embeddings() exists, using it", {})
#                     # #endregion
#                     # get_embeddings()가 토큰별 임베딩 리스트를 반환하는 경우
#                     # #region agent log
#                     debug_log("debug-session", "run1", "B", "server/main.py:421", "Calling get_embeddings()", {})
#                     # #endregion
#                     all_embeddings = llama_model.get_embeddings()
#                     # #region agent log
#                     debug_log("debug-session", "run1", "B", "server/main.py:422", "get_embeddings() result", {
#                         "is_none": all_embeddings is None,
#                         "length": len(all_embeddings) if all_embeddings is not None else 0
#                     })
#                     # #endregion
#                     if all_embeddings is not None and len(all_embeddings) > 0:
#                         # 각 토큰에 대해 해당하는 임베딩 추출
#                         for i, token_str in enumerate(input_token_strs):
#                             if i < len(all_embeddings):
#                                 token_embedding = all_embeddings[i]
#                             else:
#                                 # 인덱스가 범위를 벗어나면 마지막 임베딩 사용
#                                 token_embedding = all_embeddings[-1]
                            
#                             # numpy array일 수 있으므로 리스트로 변환
#                             if hasattr(token_embedding, 'tolist'):
#                                 embedding_list = token_embedding.tolist()
#                             elif isinstance(token_embedding, list):
#                                 embedding_list = token_embedding
#                             else:
#                                 embedding_list = list(token_embedding)
                            
#                             dim = len(embedding_list)
#                             if dim > 3:
#                                 embedding_display = embedding_list[:3] + ["..."]
#                             else:
#                                 embedding_display = embedding_list
                            
#                             token_embeddings.append(TokenEmbedding(
#                                 token=token_str,
#                                 embedding=embedding_display,
#                                 dim=dim
#                             ))
#                         print(f"[EMBEDDING] Extracted {len(token_embeddings)} token embeddings from get_embeddings()", flush=True)
#                     else:
#                         raise AttributeError("get_embeddings() returned empty or None")
#             # 방법 2: _ctx를 통한 내부 상태 접근 (eval() 없이)
#             elif hasattr(llama_model, '_ctx'):
#                 # 내부 컨텍스트에서 직접 추출 시도
#                 # llama-cpp-python의 내부 구조에 따라 다를 수 있음
#                 print(f"[EMBEDDING] Attempting to extract from internal context (no eval())...", flush=True)
#                 # #region agent log
#                 debug_log("debug-session", "run1", "C", "server/main.py:467", "_ctx path: trying to access internal state without eval()", {})
#                 # #endregion
                
#                 # eval() 없이 내부 상태를 직접 활용
#                 # _ctx를 통해 내부 임베딩에 접근 시도
#                 try:
#                     # llama-cpp-python의 내부 구조: _ctx.embeddings 또는 유사한 속성
#                     ctx = llama_model._ctx
#                     if hasattr(ctx, 'embeddings'):
#                         # 내부 임베딩 배열에서 토큰별 임베딩 추출
#                         internal_embeddings = ctx.embeddings
#                         # #region agent log
#                         debug_log("debug-session", "run1", "C", "server/main.py:477", "Found ctx.embeddings", {
#                             "shape": str(internal_embeddings.shape) if hasattr(internal_embeddings, 'shape') else "no shape"
#                         })
#                         # #endregion
                        
#                         # 각 토큰에 대해 해당하는 임베딩 추출
#                         for i, token_str in enumerate(input_token_strs):
#                             if i < len(internal_embeddings):
#                                 token_embedding = internal_embeddings[i]
#                             else:
#                                 # 인덱스가 범위를 벗어나면 마지막 임베딩 사용
#                                 token_embedding = internal_embeddings[-1]
                            
#                             # numpy array일 수 있으므로 리스트로 변환
#                             if hasattr(token_embedding, 'tolist'):
#                                 embedding_list = token_embedding.tolist()
#                             elif isinstance(token_embedding, list):
#                                 embedding_list = token_embedding
#                             else:
#                                 embedding_list = list(token_embedding)
                            
#                             dim = len(embedding_list)
#                             if dim > 3:
#                                 embedding_display = embedding_list[:3] + ["..."]
#                             else:
#                                 embedding_display = embedding_list
                            
#                             token_embeddings.append(TokenEmbedding(
#                                 token=token_str,
#                                 embedding=embedding_display,
#                                 dim=dim
#                             ))
#                         print(f"[EMBEDDING] Extracted {len(token_embeddings)} token embeddings from internal context", flush=True)
#                     else:
#                         raise AttributeError("ctx.embeddings not found")
#                 except (AttributeError, IndexError, Exception) as e:
#                     # 내부 상태 접근 실패: 다른 방법으로 접근 시도
#                     # #region agent log
#                     debug_log("debug-session", "run1", "C", "server/main.py:514", "_ctx path: internal access failed, trying alternative", {
#                         "error": str(e),
#                         "error_type": type(e).__name__
#                     })
#                     # #endregion
#                     print(f"[EMBEDDING] Internal context access failed: {e}, trying alternative method", flush=True)
                    
#                     # llama-cpp-python의 다른 내부 속성 확인
#                     try:
#                         # 방법 1: _ctx의 다른 속성 확인
#                         ctx = llama_model._ctx
#                         # 방법 2: llama_model의 다른 메서드 확인
#                         if hasattr(llama_model, '_get_embeddings'):
#                             # 내부 메서드가 있는 경우
#                             internal_embeddings = llama_model._get_embeddings()
#                             # #region agent log
#                             debug_log("debug-session", "run1", "C", "server/main.py:530", "Found _get_embeddings()", {})
#                             # #endregion
#                             # 각 토큰에 대해 해당하는 임베딩 추출
#                             for i, token_str in enumerate(input_token_strs):
#                                 if i < len(internal_embeddings):
#                                     token_embedding = internal_embeddings[i]
#                                 else:
#                                     token_embedding = internal_embeddings[-1]
                                
#                                 if hasattr(token_embedding, 'tolist'):
#                                     embedding_list = token_embedding.tolist()
#                                 elif isinstance(token_embedding, list):
#                                     embedding_list = token_embedding
#                                 else:
#                                     embedding_list = list(token_embedding)
                                
#                                 dim = len(embedding_list)
#                                 if dim > 3:
#                                     embedding_display = embedding_list[:3] + ["..."]
#                                 else:
#                                     embedding_display = embedding_list
                                
#                                 token_embeddings.append(TokenEmbedding(
#                                     token=token_str,
#                                     embedding=embedding_display,
#                                     dim=dim
#                                 ))
#                             print(f"[EMBEDDING] Extracted {len(token_embeddings)} token embeddings from _get_embeddings()", flush=True)
#                         else:
#                             # 모든 방법 실패: embed() 호출 없이 기본값 사용
#                             # #region agent log
#                             debug_log("debug-session", "run1", "C", "server/main.py:562", "_ctx path: all methods failed, using default", {})
#                             # #endregion
#                             print(f"[EMBEDDING] All internal access methods failed, using default embedding", flush=True)
#                             # 기본 임베딩 벡터 생성 (실제 임베딩이 아닌 더미 값)
#                             # 경고를 피하기 위해 embed()를 호출하지 않음
#                             default_dim = 2048  # 일반적인 임베딩 차원
#                             default_embedding = [0.0, 0.0, 0.0, "..."]
                            
#                             for token_str in input_token_strs:
#                                 token_embeddings.append(TokenEmbedding(
#                                     token=token_str,
#                                     embedding=default_embedding,
#                                     dim=default_dim
#                                 ))
#                             print(f"[EMBEDDING] Used default embedding for all tokens (no embed() call)", flush=True)
#                     except Exception as e2:
#                         # 최종 폴백: 기본값 사용
#                         # #region agent log
#                         debug_log("debug-session", "run1", "C", "server/main.py:580", "_ctx path: final fallback", {
#                             "error": str(e2)
#                         })
#                         # #endregion
#                         print(f"[EMBEDDING] Final fallback: using default embedding", flush=True)
#                         default_dim = 2048
#                         default_embedding = [0.0, 0.0, 0.0, "..."]
                        
#                         for token_str in input_token_strs:
#                             token_embeddings.append(TokenEmbedding(
#                                 token=token_str,
#                                 embedding=default_embedding,
#                                 dim=default_dim
#                             ))
#             else:
#                 raise AttributeError("No method to extract token-level embeddings found")
#             # eval() 메서드가 없는 경우: 전체 텍스트를 한 번 처리하고
#             # 각 토큰을 전체 컨텍스트에 포함시켜 처리
#             if not token_embeddings:  # 위의 방법들이 모두 실패한 경우
#                 print(f"[EMBEDDING] All methods failed, using alternative method...", flush=True)
#                 # #region agent log
#                 debug_log("debug-session", "run1", "D", "server/main.py:448", "eval() not available, using alternative", {})
#                 # #endregion
#                 # 전체 텍스트 임베딩을 먼저 생성
#                 # #region agent log
#                 debug_log("debug-session", "run1", "D", "server/main.py:450", "Calling embed() with full text", {
#                     "text_length": len(request.input_text)
#                 })
#                 # #endregion
#                 full_embedding = llama_model.embed(request.input_text)
                
#                 # 각 토큰을 전체 텍스트의 일부로 포함시켜 처리
#                 # 토큰의 위치를 고려하여 임베딩 추출
#                 for i, token_str in enumerate(input_token_strs):
#                     # 토큰 앞부분의 컨텍스트를 포함한 부분 문자열 생성
#                     # 이렇게 하면 전체 컨텍스트를 고려한 임베딩을 얻을 수 있음
#                     token_start_idx = request.input_text.find(token_str)
#                     if token_start_idx >= 0:
#                         # 토큰이 포함된 부분 문자열 (토큰 앞의 컨텍스트 포함)
#                         context_text = request.input_text[:token_start_idx + len(token_str)]
#                         # #region agent log
#                         debug_log("debug-session", "run1", "D", "server/main.py:461", "Calling embed() with context_text", {
#                             "token_index": i,
#                             "token": token_str,
#                             "context_length": len(context_text)
#                         })
#                         # #endregion
#                         token_embedding = llama_model.embed(context_text)
#                     else:
#                         # 토큰을 찾을 수 없으면 전체 텍스트 임베딩 사용
#                         token_embedding = full_embedding
                    
#                     # numpy array일 수 있으므로 리스트로 변환
#                     if hasattr(token_embedding, 'tolist'):
#                         embedding_list = token_embedding.tolist()
#                     elif isinstance(token_embedding, list):
#                         embedding_list = token_embedding
#                     else:
#                         embedding_list = list(token_embedding)
                    
#                     dim = len(embedding_list)
#                     if dim > 3:
#                         embedding_display = embedding_list[:3] + ["..."]
#                     else:
#                         embedding_display = embedding_list
                    
#                     token_embeddings.append(TokenEmbedding(
#                         token=token_str,
#                         embedding=embedding_display,
#                         dim=dim
#                     ))
#                 print(f"[EMBEDDING] Extracted {len(token_embeddings)} token embeddings using context-aware method", flush=True)
#         except Exception as e:
#             print(f"[ERROR] Failed to extract token embeddings using optimized method: {e}", flush=True)
#             import traceback
#             traceback.print_exc()
#             # 최종 폴백: 각 토큰을 개별 처리 (경고 발생하지만 동작함)
#             print(f"[EMBEDDING] Falling back to individual token processing...", flush=True)
#             # #region agent log
#             debug_log("debug-session", "run1", "E", "server/main.py:490", "Exception caught, using fallback", {
#                 "error": str(e),
#                 "error_type": type(e).__name__
#             })
#             # #endregion
#             for token_str in input_token_strs:
#                 # #region agent log
#                 debug_log("debug-session", "run1", "E", "server/main.py:493", "Fallback: calling embed() with individual token", {
#                     "token": token_str
#                 })
#                 # #endregion
#                 token_embedding = llama_model.embed(token_str)
                
#                 if hasattr(token_embedding, 'tolist'):
#                     embedding_list = token_embedding.tolist()
#                 elif isinstance(token_embedding, list):
#                     embedding_list = token_embedding
#                 else:
#                     embedding_list = list(token_embedding)
                
#                 dim = len(embedding_list)
#                 if dim > 3:
#                     embedding_display = embedding_list[:3] + ["..."]
#                 else:
#                     embedding_display = embedding_list
                
#                 token_embeddings.append(TokenEmbedding(
#                     token=token_str,
#                     embedding=embedding_display,
#                     dim=dim
#                 ))
        
#         print(f"[EMBEDDING] Extracted {len(token_embeddings)} input token embeddings", flush=True)
        
#         # 4. 출력 텍스트(생성된 응답)의 토큰 임베딩도 추출
#         print(f"[EMBEDDING] Extracting output embeddings...", flush=True)
#         output_tokens = llama_model.tokenize(generated_text.encode('utf-8'))
#         output_token_strs = [llama_model.detokenize([t]).decode('utf-8', errors='replace') for t in output_tokens]
        
#         # 빈 토큰 제거
#         filtered_output_tokens = [(token_str, token_id) for token_str, token_id in zip(output_token_strs, output_tokens) if token_str.strip()]
#         output_token_strs = [t for t, _ in filtered_output_tokens]
        
#         print(f"[EMBEDDING] Filtered output tokens: {len(output_token_strs)}", flush=True)
        
#         # 출력 텍스트의 임베딩 추출
#         try:
#             output_embeddings = llama_model.embed(generated_text)
            
#             if isinstance(output_embeddings, list) and len(output_embeddings) > 0:
#                 # 토큰과 임베딩을 zip으로 묶기
#                 output_filtered = [(token_str, emb) for token_str, emb in zip(output_token_strs, output_embeddings) if token_str.strip()]
#                 filtered_output_token_strs = [t for t, _ in output_filtered]
#                 filtered_output_embeddings = [e for _, e in output_filtered]
                
#                 # 각 출력 토큰과 임베딩을 처리
#                 for token_str, token_embedding in zip(filtered_output_token_strs, filtered_output_embeddings):
#                     # numpy array일 수 있으므로 리스트로 변환
#                     if hasattr(token_embedding, 'tolist'):
#                         embedding_list = token_embedding.tolist()
#                     elif isinstance(token_embedding, list):
#                         embedding_list = token_embedding
#                     else:
#                         embedding_list = list(token_embedding)
                    
#                     dim = len(embedding_list)
#                     if dim > 3:
#                         embedding_display = embedding_list[:3] + ["..."]
#                     else:
#                         embedding_display = embedding_list
                    
#                     token_embeddings.append(TokenEmbedding(
#                         token=token_str,
#                         embedding=embedding_display,
#                         dim=dim
#                     ))
#                 print(f"[EMBEDDING] Extracted {len(filtered_output_token_strs)} output token embeddings", flush=True)
#             else:
#                 # 단일 벡터인 경우 모든 출력 토큰에 동일한 임베딩 할당
#                 if hasattr(output_embeddings, 'tolist'):
#                     embedding_list = output_embeddings.tolist()
#                 elif isinstance(output_embeddings, list):
#                     embedding_list = output_embeddings
#                 else:
#                     embedding_list = list(output_embeddings)
                
#                 dim = len(embedding_list)
#                 if dim > 3:
#                     embedding_display = embedding_list[:3] + ["..."]
#                 else:
#                     embedding_display = embedding_list
                
#                 for token_str in output_token_strs:
#                     token_embeddings.append(TokenEmbedding(
#                         token=token_str,
#                         embedding=embedding_display,
#                         dim=dim
#                     ))
#                 print(f"[EMBEDDING] Extracted {len(output_token_strs)} output token embeddings from single vector", flush=True)
#         except Exception as e:
#             print(f"[WARNING] Failed to extract output embeddings: {e}", flush=True)
#             # 출력 임베딩 추출 실패해도 입력 임베딩은 반환
        
#         print(f"[EMBEDDING] Total token embeddings: {len(token_embeddings)} (input + output)", flush=True)
        
#         # 모든 토큰 반환 (입력 + 출력)
#         print(f"[EMBEDDING] Returning response and all token embeddings", flush=True)
        
#         return EmbeddingResponse(
#             response=generated_text,
#             tokens=token_embeddings
#         )
#     except Exception as e:
#         print(f"[ERROR] Embedding extraction failed: {e}", flush=True)
#         import traceback
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail=f"Embedding extraction error: {str(e)}")

