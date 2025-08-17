FROM python:3.11-alpine

# Install minimal dependencies
RUN apk add --no-cache gcc musl-dev libffi-dev

WORKDIR /app

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip cache purge && \
    apk del gcc musl-dev libffi-dev && \
    rm -rf ~/.cache

# Copy only essential files
COPY app.py ./
COPY Procfile ./
COPY *.json ./

ENV PYTHONUNBUFFERED=1

EXPOSE 8080

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--timeout", "300", "app:app"]
