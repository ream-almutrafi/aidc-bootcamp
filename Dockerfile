FROM python:3.11-slim

# Create non-root user and ensure full ownership of home & cache folders
RUN useradd --create-home app && \
    mkdir -p /home/app/.cache/huggingface && \
    chown -R app:app /home/app

ENV HF_HOME=/home/app/.cache/huggingface

WORKDIR /app

COPY app/requirements.txt .
RUN pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu \
      --extra-index-url https://pypi.org/simple \
      -r requirements.txt

COPY app/ .

# Switch to non-root user
USER app

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
