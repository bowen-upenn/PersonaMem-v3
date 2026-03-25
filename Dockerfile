# Use the official vLLM image as base (contains CUDA, PyTorch, vLLM)
FROM vllm/vllm-openai:v0.13.0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    tmux \
    wget \
    curl \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install Python dependencies
RUN pip install --no-cache-dir \
    json_repair \
    python-dotenv \
    tqdm \
    timeout-decorator \
    pytz \
    openai \
    anthropic \
    google-generativeai

# Install Node.js (LTS) and Claude Code CLI
RUN curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/* \
    && npm install -g @anthropic-ai/claude-code

# Reset the entrypoint (vllm image sets it to run the server by default)
ENTRYPOINT []

# Default command
CMD ["/bin/bash"]


# Quick Start:
# docker build -t personamem-v3 .
# docker run -it --gpus all -v /pool/bwjiang/personamem-v3:/workspace personamem-v3 /bin/bash
