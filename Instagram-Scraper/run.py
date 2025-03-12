"""
Main script to run the complete Instagram Knowledge Base system
"""
import os
import argparse
import logging
from time import time

# Import our modules
from config import DATA_DIR
import downloader
import transcriber
import indexer
import summarizer
from app import app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('instagram_kb.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('main')

def setup():
    """Setup necessary directories"""
    os.makedirs(DATA_DIR, exist_ok=True)

def run_downloader():
    """Run the Instagram downloader"""
    logger.info("Starting Instagram content download")
    start_time = time()
    downloader.download_from_instagram()
    logger.info(f"Download completed in {time() - start_time:.2f} seconds")

def run_transcriber():
    """Run the audio extraction and transcription"""
    logger.info("Starting audio extraction and transcription")
    start_time = time()
    transcriber.process_videos()
    logger.info(f"Transcription completed in {time() - start_time:.2f} seconds")

def run_summarizer():
    """Run the transcript summarization using Claude"""
    logger.info("Starting transcript summarization using Claude")
    start_time = time()
    summarizer.summarize_transcripts()
    logger.info(f"Summarization completed in {time() - start_time:.2f} seconds")

def run_indexer():
    """Run the knowledge base indexer"""
    logger.info("Starting indexing of transcripts")
    start_time = time()
    indexer.index_transcripts()
    logger.info(f"Indexing completed in {time() - start_time:.2f} seconds")

def run_web_interface():
    """Run the web interface"""
    logger.info("Starting web interface")
    app.run(host='0.0.0.0', port=5000)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Instagram Knowledge Base')
    parser.add_argument('--download', action='store_true', help='Run the downloader module')
    parser.add_argument('--transcribe', action='store_true', help='Run the transcription module')
    parser.add_argument('--summarize', action='store_true', help='Run the summarization module')
    parser.add_argument('--index', action='store_true', help='Run the indexer module')
    parser.add_argument('--web', action='store_true', help='Run the web interface')
    parser.add_argument('--all', action='store_true', help='Run the complete pipeline')
    
    args = parser.parse_args()
    
    # Setup directories
    setup()
    
    # Run requested modules
    if args.all or args.download:
        run_downloader()
    
    if args.all or args.transcribe:
        run_transcriber()
    
    if args.all or args.summarize:
        run_summarizer()
    
    if args.all or args.index:
        run_indexer()
    
    if args.all or args.web:
        run_web_interface()
    
    # If no arguments provided, show help
    if not (args.download or args.transcribe or args.summarize or args.index or args.web or args.all):
        parser.print_help()

if __name__ == "__main__":
    main() 