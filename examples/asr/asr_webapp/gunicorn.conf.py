# https://docs.gunicorn.org/en/stable/settings.html

# NOTE: Do not add server hooks or import asr_service.py here
# Gunicorn performs fork - and CUDA cannot be used in forked multiprocess.

# General config
bind = "0.0.0.0:8000"
workers = 2

# Worker specific config
worker_connections = 1000
timeout = 180  # 3 minutes of timeout

