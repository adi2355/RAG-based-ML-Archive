# Instagram Scraper

Instagram scraper that works around rate limiting. Download, transcribe, and analyze Instagram content using AI.

## Features

- **Instagram Content Downloading**: Download videos and metadata from Instagram accounts
- **Audio Transcription**: Extract audio and transcribe using Whisper
- **AI-Powered Summarization**: Generate structured summaries with Claude
- **Batch Processing**: Cost-efficient processing of transcripts using Claude Batches API
- **Search & Retrieval**: Index and search content with full-text search

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/Instagram-Scraper.git
   cd Instagram-Scraper
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your configuration in `config.py` including:
   - Instagram credentials
   - Anthropic API key for Claude

## Usage

### Basic Commands

```bash
# Run all steps
python run.py --all

# Download content from Instagram
python run.py --download

# Extract and transcribe audio from videos
python run.py --transcribe 

# Generate AI summaries of transcripts
python run.py --summarize

# Run the web interface
python run.py --web
```

### Advanced Options

```bash
# Disable batch processing for summarization (costs more but may be faster for small datasets)
python run.py --summarize --no-batch

# Force refresh Instagram content
python run.py --download --refresh-force

# Run Instagram download without authentication 
python run.py --download --no-auth
```

## AI Summarization with Claude Batch Processing

The system uses Claude to analyze and summarize video transcripts. By default, it uses Claude's Message Batches API, which offers:

- **50% Cost Reduction**: Batch processing is half the cost of regular API calls
- **Asynchronous Processing**: Process large volumes of transcripts efficiently
- **Structured Output**: Generates structured summaries with key topics, insights and more

### How Batch Processing Works

1. Transcripts are collected and prepared in batches of up to 100 items
2. The batch is submitted to Claude's Message Batches API for asynchronous processing
3. The system polls for batch completion and processes results when ready
4. Summaries are stored in the database and cached for future use

### Cost Efficiency

With batch processing enabled (default), Claude API costs are reduced by 50%:

| Model              | Batch Input   | Batch Output  | Standard Input | Standard Output |
|--------------------|---------------|---------------|----------------|-----------------|
| Claude 3.7 Sonnet  | $1.50 / MTok  | $7.50 / MTok  | $3.00 / MTok   | $15.00 / MTok   |
| Claude 3.5 Sonnet  | $1.50 / MTok  | $7.50 / MTok  | $3.00 / MTok   | $15.00 / MTok   |
| Claude 3 Haiku     | $0.125 / MTok | $0.625 / MTok | $0.25 / MTok   | $1.25 / MTok    |

The system uses Claude 3 Haiku by default for optimal cost efficiency.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
