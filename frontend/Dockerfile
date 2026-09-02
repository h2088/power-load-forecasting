FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY frontend/requirements.txt /tmp/frontend-requirements.txt
RUN pip install --no-cache-dir -r /tmp/frontend-requirements.txt

COPY frontend /app/frontend

EXPOSE 8501

CMD ["streamlit", "run", "frontend/frontend.py", "--server.address=0.0.0.0", "--server.port=8501"]
