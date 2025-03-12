"""
Configuration file for Instagram Knowledge Base
"""
import os

# Content sources configuration
# Currently supports Instagram, with framework for adding more sources
CONTENT_SOURCES = [
    {"type": "instagram", "username": "rajistics", "private": False}
    # Add more sources as needed
    # {"type": "youtube", "channel_id": "CHANNEL_ID"},  # YouTube example
    # {"type": "twitter", "username": "username"},  # Twitter example
]

# Instagram accounts list (for backward compatibility)
INSTAGRAM_ACCOUNTS = [account for account in CONTENT_SOURCES if account["type"] == "instagram"]

# Instagram credentials (only needed for private accounts)
# IMPORTANT: Add your Instagram credentials here to avoid 401 Unauthorized errors
# You can either set environment variables or directly add your credentials below:
# INSTAGRAM_USERNAME = "your_instagram_username"
# INSTAGRAM_PASSWORD = "your_instagram_password"
INSTAGRAM_USERNAME = os.getenv("INSTAGRAM_USERNAME", "")
INSTAGRAM_PASSWORD = os.getenv("INSTAGRAM_PASSWORD", "")

# Multiple Instagram accounts for rotation (to reduce rate limiting)
# If using account rotation, populate this list with your accounts
INSTAGRAM_ACCOUNT_ROTATION = {
    "enabled": False,  # Disabled due to 2FA requirements
    "accounts": [
        {"username": "adi_ka_dusra_account", "password": "Ishaan07"},
        {"username": "adi_khetarpal", "password": "vijne1-Xingir-pomcob"},
        # Add your Instagram accounts here for rotation
    ]
}

# Proxy configuration
# Add your proxy servers here to rotate IPs and reduce rate limiting
PROXY_SERVERS = [
    "http://brd-customer-hl_c7bff232-zone-residential_proxy1-country-us:w46vs0z46xmc@brd.superproxy.io:33335",
    # Add more proxies as needed
]

# Configure if you want to use a specific country for proxies (US, UK, etc.)
PROXY_COUNTRY = "us"  # Change as needed, or set to None for random

# Claude API key for summarization
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# Directory settings
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DOWNLOAD_DIR = os.path.join(DATA_DIR, "downloads")
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
TRANSCRIPT_DIR = os.path.join(DATA_DIR, "transcripts")

# Whisper model size: tiny, base, small, medium, large
WHISPER_MODEL = "base"

# Database settings
DB_PATH = os.path.join(DATA_DIR, "knowledge_base.db")

# Web interface settings
WEB_PORT = 5001
DEBUG_MODE = True

# Rate limiting settings (to avoid IP blocks)
DOWNLOAD_DELAY = 10  # seconds between downloads (increased from 5 to reduce rate limiting)
MAX_DOWNLOADS_PER_RUN = 2  # Maximum videos to download per run
ACCOUNT_COOLDOWN_MINUTES = 60  # How long to wait before using an account again after a failure
PROXY_COOLDOWN_MINUTES = 30  # How long to wait before using a proxy again after a failure
RATE_LIMIT_WAIT = 3600  # Seconds to wait after hitting a rate limit 