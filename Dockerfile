# Python 3.11 슬림 이미지 사용
FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# 의존성 파일 복사
COPY requirements.txt .

# 의존성 설치
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY app/ ./app/

# 포트 8080 노출 (Cloud Run 기본 포트)
EXPOSE 8080

# 환경변수 설정
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# uvicorn으로 FastAPI 서버 실행
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1

