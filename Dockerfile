FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (filtered by .dockerignore)
COPY . .
# Ensure data directory exists with reasonable permissions
RUN mkdir -p data/PaperBananaBench/diagram data/PaperBananaBench/plot && chmod -R 770 data

# Expose port
EXPOSE 8080

# Run Streamlit
CMD ["streamlit", "run", "demo.py", "--server.port", "8080", "--server.address", "0.0.0.0"]
