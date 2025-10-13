# Deep Research AI Agent - Multi-stage Docker Build
#
# This Dockerfile creates an optimized production container for the research agent.
# It uses a multi-stage build to minimize the final image size and improve security.

# ==================== Stage 1: Builder ====================
# This stage installs build-time dependencies and Python packages into a virtual environment.
FROM python:3.13-slim AS builder

# Set working directory
WORKDIR /build

# Install build dependencies required for some Python packages (e.g., psycopg2)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements file to leverage Docker layer caching
COPY requirements.txt .

# Create a virtual environment and add it to the PATH
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies into the virtual environment
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ==================== Stage 2: Runtime ====================
# This stage creates the final, lightweight production image.
FROM python:3.13-slim

# Set environment variables for Python and the application
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app

# Set working directory
WORKDIR /app

# Install runtime dependencies (libpq5 for psycopg2, curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user and group for better security
RUN groupadd -r appuser && \
    useradd -r -g appuser -u 1000 -d /app -s /sbin/nologin appuser

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy only the necessary application files
# This is more secure and explicit than copying everything.
COPY --chown=appuser:appuser src ./src
COPY --chown=appuser:appuser chainlit.md .
COPY --chown=appuser:appuser alembic.ini .
COPY --chown=appuser:appuser pyproject.toml .

# Create and set permissions for directories the app might write to
RUN mkdir -p /app/logs /app/reports && \
    chown -R appuser:appuser /app/logs /app/reports

COPY --chown=appuser:appuser .chainlit /app/.chainlit

# Change ownership of the entire app directory to the non-root user
RUN chown -R appuser:appuser /app

# Switch to the non-root user
USER appuser

# Expose the port the application will run on
EXPOSE 8000

# Health check to verify the Chainlit server is running
# Chainlit serves a 200 OK on the root path.
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000 || exit 1

# The command to run the application
# This mirrors the local command `PYTHONPATH=. chainlit run ...`
CMD ["chainlit", "run", "src/ui/chainlit_app.py", "--host", "0.0.0.0", "--port", "8000"]