FROM python:3.11-slim-bookworm

LABEL author="Praagnya"
LABEL description="LING 539 Kaggle Competition -- 3-class text classification"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -U pip \
    && pip install --no-cache-dir -r requirements.txt \
    && jupyter contrib nbextension install --user

# Copy project files
COPY . .

# Make scripts executable
RUN chmod u+x scripts/*

EXPOSE 9999

# Default: launch Jupyter notebook
# To run a solution instead:
#   docker run -v $(pwd)/data:/app/data <image> python solutions/solution_v4.py
#   docker run -v $(pwd)/data:/app/data <image> python solutions/solution_v5.py
CMD ["bash", "scripts/launch-notebook"]
