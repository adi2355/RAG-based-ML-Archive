<<<<<<< HEAD
# Instagram-Scraper
Instagram scraper that i created to pass through instagram's strict rate limiting using cheeky tricks

# Instagram Knowledge Base

A comprehensive solution for building a personal knowledge base from Instagram videos. This system downloads videos from Instagram accounts, transcribes them using OpenAI's Whisper, summarizes them using Claude, and makes them searchable through a web interface.

## Features

- **Multi-account support**: Configure multiple Instagram accounts in the config file
- **Metadata extraction**: Stores engagement metrics, hashtags, captions, and timestamps
- **AI-powered summarization**: Generate concise summaries of video transcripts using Claude
- **Full-text search**: Find specific content across all transcripts and summaries
- **Prioritized search results**: Results from summaries and captions are prioritized over full transcripts
- **Filter capabilities**: Filter by account, hashtag, or date
- **Visualization**: Statistics dashboard to explore your content collection
- **Rate limiting**: Respectful downloading with exponential backoff to avoid IP blocks
- **Modular design**: Run individual components as needed
- **Performance optimization**: Caching for frequently accessed data
- **Resilient downloads**: Robust retry mechanism with exponential backoff

## Requirements

- Python 3.8+
- FFmpeg (for audio extraction)
- Instagram account (for accessing content)
- Anthropic API key (for Claude summarization)

## Installation

1. Clone this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Install FFmpeg if not already installed (system dependent):
   - Ubuntu/Debian: `sudo apt-get install ffmpeg`
   - macOS: `brew install ffmpeg`
   - Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html)

## Configuration

1. Edit `config.py` to add Instagram accounts you have permission to access
2. Set environment variables:
   ```bash
   # For Instagram authentication (strongly recommended)
   export INSTAGRAM_USERNAME="your_username"
   export INSTAGRAM_PASSWORD="your_password"
   
   # For Claude summarization
   export ANTHROPIC_API_KEY="your_anthropic_api_key"
   ```
   - For Windows, use `set` instead of `export`
   - For production, consider using a `.env` file with python-dotenv

## Usage

### Running the Complete Pipeline

Run all steps (download, transcribe, summarize, index, web):

```bash
python run.py --all
```

### Running Individual Components

Download videos:
```bash
python run.py --download
```

Transcribe videos:
```bash
python run.py --transcribe
```

Summarize transcripts using Claude:
```bash
python run.py --summarize
```

Index transcripts and summaries:
```bash
python run.py --index
```

Start the web interface:
```bash
python run.py --web
```

## Accessing the Web Interface

Once running, access the web interface at:
```
http://localhost:5000
```

## Advanced Features

### Transcript Summarization

The system now includes Claude-powered summarization that:
- Generates concise summaries (150-200 words) for each video transcript
- Highlights key insights and actionable information
- Implements efficient batch processing to respect API rate limits
- Caches summaries to avoid redundant API calls
- Enhances search by prioritizing matches in summaries

### Performance Optimizations

- **LRU Caching**: Frequently accessed data is cached to reduce database load
- **Improved Indexing**: Better database schema with optimized indices for common query patterns
- **Enhanced Search**: Prioritized search results with highlighted matches
- **Resilient Downloads**: Smart retry with exponential backoff for more reliable downloads

### Content Analytics

The system tracks:
- Word count for each transcript
- Video duration (when available)
- Key content metrics accessible through the statistics dashboard

## Ethical Usage

This tool is for personal use only, with content you have permission to access. Always respect copyright and terms of service for the platforms you use.

## License

MIT License 
>>>>>>> b643539 (Instagram scraper and summarizer)
