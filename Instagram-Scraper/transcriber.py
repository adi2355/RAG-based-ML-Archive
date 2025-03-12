"""
Module for extracting audio from videos and transcribing with Whisper
"""
import os
import json
import glob
import subprocess
import logging
from tqdm import tqdm
import whisper

from config import (
    DOWNLOAD_DIR,
    AUDIO_DIR,
    TRANSCRIPT_DIR,
    WHISPER_MODEL
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('transcriber.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('transcriber')

def setup_directories():
    """Create necessary directories if they don't exist"""
    os.makedirs(AUDIO_DIR, exist_ok=True)
    os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
    
    # Create account-specific directories
    for account_dir in os.listdir(DOWNLOAD_DIR):
        if os.path.isdir(os.path.join(DOWNLOAD_DIR, account_dir)):
            os.makedirs(os.path.join(AUDIO_DIR, account_dir), exist_ok=True)
            os.makedirs(os.path.join(TRANSCRIPT_DIR, account_dir), exist_ok=True)

def extract_audio(video_path, audio_path):
    """Extract audio from video using FFmpeg"""
    try:
        cmd = f'ffmpeg -i "{video_path}" -vn -acodec pcm_s16le -ar 16000 -ac 1 "{audio_path}" -y'
        subprocess.call(cmd, shell=True)
        return True
    except Exception as e:
        logger.error(f"Error extracting audio from {video_path}: {str(e)}")
        return False

def process_videos():
    """Process all downloaded videos that haven't been transcribed yet"""
    setup_directories()
    
    # Load Whisper model
    logger.info(f"Loading Whisper model: {WHISPER_MODEL}")
    model = whisper.load_model(WHISPER_MODEL)
    logger.info("Model loaded successfully")
    
    # Get all video files
    all_videos = []
    for account_dir in os.listdir(DOWNLOAD_DIR):
        account_path = os.path.join(DOWNLOAD_DIR, account_dir)
        if os.path.isdir(account_path):
            videos = glob.glob(os.path.join(account_path, "*.mp4"))
            all_videos.extend(videos)
    
    logger.info(f"Found {len(all_videos)} total videos to process")
    
    # Process each video
    for video_path in tqdm(all_videos, desc="Processing videos"):
        # Extract account name and filename
        parts = video_path.split(os.sep)
        account = parts[-2]
        filename = os.path.basename(video_path)
        base_name = os.path.splitext(filename)[0]
        
        # Define paths
        audio_path = os.path.join(AUDIO_DIR, account, f"{base_name}.wav")
        transcript_path = os.path.join(TRANSCRIPT_DIR, account, f"{base_name}.json")
        
        # Skip if already transcribed
        if os.path.exists(transcript_path):
            continue
        
        # Extract audio if needed
        if not os.path.exists(audio_path):
            logger.info(f"Extracting audio from {filename}")
            os.makedirs(os.path.dirname(audio_path), exist_ok=True)
            if not extract_audio(video_path, audio_path):
                continue
        
        # Transcribe audio
        try:
            logger.info(f"Transcribing {filename}")
            result = model.transcribe(audio_path)
            
            # Get metadata if available
            metadata_path = os.path.join(DOWNLOAD_DIR, account, "metadata", f"{base_name}.json")
            metadata = {}
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            
            # Create transcript with metadata
            transcript_data = {
                "text": result["text"],
                "segments": result["segments"],
                "language": result["language"],
                "filename": filename,
                "account": account,
                **metadata
            }
            
            # Save transcript
            os.makedirs(os.path.dirname(transcript_path), exist_ok=True)
            with open(transcript_path, 'w', encoding='utf-8') as f:
                json.dump(transcript_data, f, ensure_ascii=False, indent=4)
            
            logger.info(f"Transcription complete for {filename}")
            
        except Exception as e:
            logger.error(f"Error transcribing {filename}: {str(e)}")
    
    logger.info("Transcription process completed")

if __name__ == "__main__":
    process_videos() 