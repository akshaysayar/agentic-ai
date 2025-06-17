# Use official slim Python 3.12 image
FROM python:3.12-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements first for caching
COPY requirements/requirements.txt ./

# Upgrade pip and install dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# Copy the source code
COPY src/ ./src/

# Set PYTHONPATH so python can find your package
ENV PYTHONPATH=/app/src

# Default command runs the ingest pipeline (change if needed)
CMD ["python", "src/research_tool_rag/rag/ingest_pipeline.py"]
