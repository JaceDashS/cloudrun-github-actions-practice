# Python 3.11 슬림 이미지 사용
FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# 빌드 도구 설치 (llama-cpp-python 컴파일을 위해 필요)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# 의존성 파일 복사
COPY requirements.txt .

# 의존성 설치
RUN pip install --no-cache-dir -r requirements.txt

# 모델 파일 복사 (미리 다운로드된 모델)
COPY models/ /app/models/

# 애플리케이션 코드 복사
COPY app/ ./app/

# 포트 8080 노출 (Cloud Run 기본 포트)
EXPOSE 8080

# 환경변수 설정
ENV PORT=8080
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf

# uvicorn으로 FastAPI 서버 실행
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1

