# https://docs.gunicorn.org/en/stable/settings.html

# General config
bind = "127.0.0.1:8000"
workers = 2

# Worker specific config
worker_connections = 1000
timeout = 30
