FROM python:3.11-slim

WORKDIR /app

# Install system deps for PyMuPDF and Tesseract OCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p vectorstore data templates static

EXPOSE 5000

# Default: run the web server
# To ingest: docker run <image> python ingest.py
CMD ["python", "app.py"]
