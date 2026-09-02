FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    LOKY_MAX_CPU_COUNT=1

WORKDIR /app

COPY backend/requirements.txt /tmp/backend-requirements.txt
RUN pip install --no-cache-dir -r /tmp/backend-requirements.txt

COPY backend /app/backend
COPY src /app/src
COPY model.py /app/model.py
RUN mkdir -p /app/data

EXPOSE 5000

CMD ["python", "-m", "backend.app"]
