#!/usr/bin/env python
"""
Script to generate tags for Instagram videos
This will:
1. Find all videos in the specified directory
2. Extract audio from each video (if not already done)
3. Transcribe the audio
4. Generate relevant tags based on the transcription
"""
import os
import sys
import json
import logging
import subprocess
import lzma
from datetime import datetime
from pathlib import Path

# Import from our existing modules
from config import DATA_DIR, DOWNLOAD_DIR, AUDIO_DIR, TRANSCRIPT_DIR, WHISPER_MODEL
import transcriber

# Add additional imports for transcription
import whisper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(DATA_DIR, 'logs', 'tag_generator.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('tag_generator')

# Special path for Instaloader downloads
INSTALOADER_PATH = os.path.join(DOWNLOAD_DIR, 'rajistics', 'rajistics')
UNICODE_INSTALOADER_PATH = "∕home∕adi235∕MistralOCR∕Instagram-Scraper∕data∕downloads∕rajistics/rajistics"

# Load Whisper model (only once)
whisper_model = None
def get_whisper_model():
    """Load the Whisper model if not already loaded"""
    global whisper_model
    if whisper_model is None:
        logger.info(f"Loading Whisper model: {WHISPER_MODEL}")
        whisper_model = whisper.load_model(WHISPER_MODEL)
        logger.info("Whisper model loaded successfully")
    return whisper_model

def setup_directories():
    """Create necessary directories if they don't exist"""
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    os.makedirs(AUDIO_DIR, exist_ok=True)
    os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "tags"), exist_ok=True)
    
    # Create metadata directories for each account
    for account in os.listdir(DOWNLOAD_DIR):
        if os.path.isdir(os.path.join(DOWNLOAD_DIR, account)):
            metadata_dir = os.path.join(DOWNLOAD_DIR, account, "metadata")
            os.makedirs(metadata_dir, exist_ok=True)

def find_videos(base_dir=DOWNLOAD_DIR, account=None):
    """Find all MP4 videos in the specified directory"""
    videos = []
    
    if account and account == "rajistics":
        # First check the special Instaloader path with Unicode slashes
        logger.info(f"Checking special Instaloader path for rajistics: {UNICODE_INSTALOADER_PATH}")
        try:
            # Use subprocess to find videos in the special path
            result = subprocess.run(
                f'find "{UNICODE_INSTALOADER_PATH}" -name "*.mp4"',
                shell=True, capture_output=True, text=True
            )
            if result.returncode == 0:
                instaloader_videos = result.stdout.strip().split('\n')
                if instaloader_videos and instaloader_videos[0]:  # Check if there's at least one video
                    logger.info(f"Found {len(instaloader_videos)} videos in Instaloader path")
                    videos.extend(instaloader_videos)
        except Exception as e:
            logger.error(f"Error finding videos in Instaloader path: {str(e)}")
    
    if account:
        # Look in specific account directory
        account_dir = os.path.join(base_dir, account)
        if os.path.exists(account_dir):
            for root, _, files in os.walk(account_dir):
                # Skip the special rajistics/rajistics subdirectory if we already processed it
                if account == "rajistics" and "rajistics/rajistics" in root:
                    continue
                
                for file in files:
                    if file.endswith('.mp4'):
                        videos.append(os.path.join(root, file))
    else:
        # Look in all account directories
        for root, _, files in os.walk(base_dir):
            # Skip the special rajistics/rajistics subdirectory if we already processed it
            if "rajistics/rajistics" in root:
                continue
                
            for file in files:
                if file.endswith('.mp4'):
                    videos.append(os.path.join(root, file))
        
        # Also check the special Instaloader path
        try:
            result = subprocess.run(
                f'find "{UNICODE_INSTALOADER_PATH}" -name "*.mp4"',
                shell=True, capture_output=True, text=True
            )
            if result.returncode == 0:
                instaloader_videos = result.stdout.strip().split('\n')
                if instaloader_videos and instaloader_videos[0]:  # Check if there's at least one video
                    logger.info(f"Found {len(instaloader_videos)} videos in Instaloader path")
                    videos.extend(instaloader_videos)
        except Exception as e:
            logger.error(f"Error finding videos in Instaloader path: {str(e)}")
    
    return videos

def generate_tags_from_transcript(transcript_text):
    """Generate relevant tags from transcript text"""
    # This is a simple implementation - you could use NLP or LLMs for better results
    
    # Split into words, lowercase, and remove punctuation
    words = transcript_text.lower().replace('.', ' ').replace(',', ' ').replace('!', ' ').replace('?', ' ')
    words = words.replace(':', ' ').replace(';', ' ').replace('-', ' ').split()
    
    # Remove common stopwords
    stopwords = {'the', 'and', 'is', 'in', 'to', 'a', 'of', 'for', 'with', 'on', 'at', 'from', 
                'by', 'about', 'as', 'an', 'are', 'that', 'this', 'it', 'be', 'i', 'you', 'we'}
    
    filtered_words = [word for word in words if word not in stopwords and len(word) > 2]
    
    # Count word frequencies
    word_counts = {}
    for word in filtered_words:
        if word in word_counts:
            word_counts[word] += 1
        else:
            word_counts[word] = 1
    
    # Get the most frequent words as tags
    tags = [word for word, count in sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:15]]
    
    # Additional domain-specific tags for data science/ML content
    domain_tags = ['machinelearning', 'datascience', 'ai', 'python', 'analytics', 'statistics',
                  'deeplearning', 'algorithms', 'data', 'coding', 'programming', 'visualization']
    
    return list(set(tags + [tag for tag in domain_tags if tag in filtered_words]))

def process_video(video_path, force=False):
    """Process a single video to generate tags"""
    logger.info(f"Processing video: {video_path}")
    
    # Check if this is a video from the special Instaloader path
    is_instaloader_video = UNICODE_INSTALOADER_PATH in video_path
    
    # Derive paths
    video_name = os.path.basename(video_path)
    video_name_without_ext = os.path.splitext(video_name)[0]
    
    # Extract account name
    account_name = "rajistics" if is_instaloader_video else "unknown"
    
    if not is_instaloader_video:
        # Standard path, extract account from directory structure
        path_parts = video_path.split(os.sep)
        for i, part in enumerate(path_parts):
            if part == "downloads" and i+1 < len(path_parts):
                account_name = path_parts[i+1]
                break
    
    # Create account directories if they don't exist
    account_audio_dir = os.path.join(AUDIO_DIR, account_name)
    os.makedirs(account_audio_dir, exist_ok=True)
    
    # Create metadata directory if it doesn't exist
    metadata_dir = os.path.join(DOWNLOAD_DIR, account_name, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)
    
    audio_path = os.path.join(account_audio_dir, f"{video_name_without_ext}.mp3")
    transcript_path = os.path.join(TRANSCRIPT_DIR, f"{video_name_without_ext}.txt")
    
    # Extract shortcode from filename or related JSON.XZ file
    shortcode = None
    date_str = None
    time_str = None
    
    # Parse the date and time from the filename
    parts = video_name_without_ext.split('_')
    if len(parts) >= 3:
        date_str = parts[0]
        time_str = parts[1]
        if parts[2] != "UTC":
            shortcode = parts[2]
    
    # For Instaloader videos, try to extract shortcode from corresponding JSON.XZ file
    if is_instaloader_video and not shortcode:
        json_xz_path = f"{video_path[:-4]}.json.xz"  # Replace .mp4 with .json.xz
        
        # Check if the JSON.XZ file exists
        try:
            if os.path.exists(json_xz_path):
                logger.info(f"Found JSON.XZ file: {json_xz_path}")
                
                # Extract shortcode from JSON.XZ
                with lzma.open(json_xz_path, 'rt', encoding='utf-8') as f:
                    try:
                        insta_data = json.load(f)
                        if 'node' in insta_data and 'shortcode' in insta_data['node']:
                            shortcode = insta_data['node']['shortcode']
                            logger.info(f"Extracted shortcode from JSON.XZ: {shortcode}")
                    except Exception as e:
                        logger.error(f"Error parsing JSON.XZ: {str(e)}")
        except Exception as e:
            logger.error(f"Error accessing JSON.XZ file: {str(e)}")
    
    # If still no shortcode, try to find an existing metadata file or generate a fallback
    if not shortcode:
        if date_str and time_str:
            potential_files = [f for f in os.listdir(metadata_dir) if f.endswith('.json')]
            for file in potential_files:
                if date_str in file and time_str in file:
                    shortcode = file.split('.')[0]  # Remove .json extension
                    break
        
        # If still no shortcode, use a fallback
        if not shortcode:
            shortcode = f"unknown_{video_name_without_ext}"
    
    # Path for the metadata file (same format as downloader.py)
    metadata_path = os.path.join(metadata_dir, f"{shortcode}.json")
    
    # Skip if metadata already has tags and not forcing
    if os.path.exists(metadata_path) and not force:
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                if 'tags' in metadata and metadata['tags']:
                    logger.info(f"Tags already exist for {video_name}, skipping (use --force to reprocess)")
                    return
        except Exception as e:
            logger.error(f"Error reading existing metadata: {str(e)}")
            # If there's an error reading the file, we'll just proceed to generate tags
    
    # Extract audio if needed
    if not os.path.exists(audio_path) or force:
        logger.info(f"Extracting audio to: {audio_path}")
        try:
            # Make sure the directory exists
            os.makedirs(os.path.dirname(audio_path), exist_ok=True)
            
            # Use ffmpeg to extract audio
            result = subprocess.run([
                'ffmpeg', '-i', video_path, '-q:a', '0', '-map', 'a', '-vn', '-y', audio_path
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Failed to extract audio: {result.stderr}")
                return
        except Exception as e:
            logger.error(f"Error extracting audio: {str(e)}")
            return
    
    # Transcribe audio if needed
    if not os.path.exists(transcript_path) or force:
        logger.info(f"Transcribing audio to: {transcript_path}")
        try:
            # Use our direct transcription function
            transcript_text = transcribe_audio(audio_path, transcript_path)
        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}")
            return
    
    # Read transcript
    try:
        with open(transcript_path, 'r') as f:
            transcript_text = f.read()
    except Exception as e:
        logger.error(f"Error reading transcript: {str(e)}")
        return
    
    # Generate tags
    tags = generate_tags_from_transcript(transcript_text)
    
    # Load existing metadata if available
    existing_metadata = {}
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                existing_metadata = json.load(f)
        except Exception as e:
            logger.error(f"Error reading existing metadata: {str(e)}")
    
    # Update metadata with tags in the format used by downloader.py
    metadata = {
        **existing_metadata,
        'shortcode': shortcode,
        'account': account_name,
        'tags': tags,
        'transcript': transcript_text[:500] + '...' if len(transcript_text) > 500 else transcript_text,  # Store a preview of the transcript
        'transcript_path': transcript_path,
        'audio_path': audio_path,
        'video_path': video_path,
        'tag_generation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Save metadata
    try:
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)
        logger.info(f"Generated {len(tags)} tags for {video_name}")
        logger.info(f"Tags: {', '.join(tags)}")
    except Exception as e:
        logger.error(f"Error saving metadata: {str(e)}")

def transcribe_audio(audio_path, transcript_path):
    """Transcribe audio file using Whisper"""
    try:
        logger.info(f"Starting transcription of: {audio_path}")
        model = get_whisper_model()
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(transcript_path), exist_ok=True)
        
        # Run transcription
        result = model.transcribe(audio_path)
        transcript_text = result["text"]
        
        # Save transcript to file
        with open(transcript_path, 'w', encoding='utf-8') as f:
            f.write(transcript_text)
        
        logger.info(f"Transcription completed for: {audio_path}")
        return transcript_text
    except Exception as e:
        logger.error(f"Error transcribing audio: {str(e)}")
        raise

def main():
    """Main function to process videos and generate tags"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate tags for Instagram videos')
    parser.add_argument('--account', help='Process only videos from this account')
    parser.add_argument('--force', action='store_true', help='Force reprocessing of videos')
    parser.add_argument('--video', help='Process only this specific video')
    parser.add_argument('--instaloader', action='store_true', help='Process videos from Instaloader directory')
    args = parser.parse_args()
    
    setup_directories()
    
    if args.instaloader:
        logger.info("Processing videos from Instaloader directory")
        args.account = "rajistics"  # Instaloader videos are from rajistics
    
    if args.video:
        # Process a single video
        if os.path.exists(args.video):
            process_video(args.video, args.force)
        else:
            logger.error(f"Video not found: {args.video}")
            return
    else:
        # Process all videos for the specified account or all accounts
        videos = find_videos(account=args.account)
        logger.info(f"Found {len(videos)} videos to process")
        
        for i, video_path in enumerate(videos):
            logger.info(f"Processing video {i+1}/{len(videos)}")
            try:
                process_video(video_path, args.force)
            except Exception as e:
                logger.error(f"Error processing video {video_path}: {str(e)}")
                continue  # Continue with next video
    
    logger.info("Tag generation completed")

if __name__ == "__main__":
    main() 