FROM python:3.12-slim-bookworm

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgomp1 \
        libzbar0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# CPU wheels by default. A CUDA image is documented in README.md.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY src ./src

EXPOSE 8092

CMD ["uvicorn", "api.server:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8092"]
