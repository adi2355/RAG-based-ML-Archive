"""
Module for summarizing video transcripts using Claude API
"""
import os
import json
import time
import sqlite3
import logging
import re
from anthropic import Anthropic

from config import DB_PATH, TRANSCRIPT_DIR, DATA_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('summarizer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('summarizer')

class ClaudeSummarizer:
    def __init__(self, api_key=None):
        """Initialize the Claude summarizer with API key"""
        # Update to use the current API initialization pattern (v0.49.0)
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.client = Anthropic(api_key=api_key)
        
        self.cache_dir = os.path.join(DATA_DIR, "summaries_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_file = os.path.join(self.cache_dir, "summary_cache.json")
        self._load_cache()
    
    def _load_cache(self):
        """Load summary cache from file"""
        try:
            with open(self.cache_file, 'r') as f:
                self.cache = json.load(f)
                logger.info(f"Loaded {len(self.cache)} cached summaries")
        except (FileNotFoundError, json.JSONDecodeError):
            self.cache = {}
            logger.info("Created new summary cache")
    
    def _save_cache(self):
        """Save summary cache to file"""
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f)
            logger.info(f"Saved {len(self.cache)} summaries to cache")
    
    def _create_enhanced_prompt(self, transcript, metadata=None):
        """Create an enhanced prompt for Claude with context about the video"""
        context = ""
        if metadata:
            context = f"Video by: {metadata.get('account', 'Unknown')}\n"
            if metadata.get('timestamp'):
                context += f"Posted: {metadata.get('timestamp', '').split('T')[0]}\n"
            if metadata.get('caption'):
                context += f"Caption: {metadata.get('caption', 'No caption')}\n"
        
        prompt = f"""Analyze this Instagram video transcript and provide structured information in your response:

{context}
TRANSCRIPT:
{transcript}

Please extract the following information:
1. Summary: Concise overview of the main points (80-100 words max)
2. Key Topics: 3-5 main topics covered
3. Entities Mentioned: People, products, services, or concepts mentioned
4. Tone & Style: Formal/informal, educational/conversational, etc.
5. Key Insights: 2-3 main takeaways or insights
6. Actionable Information: Any specific advice, steps, or actionable items
7. Content Type: What kind of content is this (tutorial, conversation, review, educational, etc.)

Provide your analysis in this format:

SUMMARY: [Your concise summary]

KEY TOPICS:
- [Topic 1]
- [Topic 2]
- [Topic 3]

ENTITIES MENTIONED:
- [Entity 1] (person/product/concept)
- [Entity 2] (person/product/concept)

TONE & STYLE: [Analysis of tone and presentation style]

KEY INSIGHTS:
- [Insight 1]
- [Insight 2]

ACTIONABLE INFORMATION:
- [Actionable item 1] 
- [Actionable item 2]

CONTENT TYPE: [content type classification]
"""
        return prompt
    
    def _extract_key_phrases(self, transcript, summary):
        """Extract key phrases and terms from transcript and summary"""
        combined_text = f"{summary} {transcript}"
        # Split into words, lowercase, and remove punctuation
        words = re.findall(r'\b\w+\b', combined_text.lower())
        
        # Count word frequencies (excluding common words)
        common_words = {'the', 'and', 'that', 'for', 'you', 'with', 'this', 'was', 'are', 'have', 
                       'its', 'they', 'from', 'but', 'not', 'what', 'all', 'were', 'when', 'your',
                       'can', 'said', 'there', 'use', 'been', 'has', 'would', 'each', 'which', 'she',
                       'how', 'their', 'will', 'other', 'about', 'out', 'many', 'then', 'them', 'these',
                       'some', 'her', 'him', 'into', 'more', 'could', 'know', 'like', 'just'}
        
        word_freq = {}
        for word in words:
            if word not in common_words and len(word) > 3:  # Exclude common words and short words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top N words by frequency
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Extract 2-3 word phrases using regex patterns
        phrase_patterns = [
            r'\b\w+\s+\w+\b',           # 2-word phrases
            r'\b\w+\s+\w+\s+\w+\b'      # 3-word phrases
        ]
        
        phrases = []
        for pattern in phrase_patterns:
            phrases.extend(re.findall(pattern, combined_text.lower()))
        
        # Count phrase frequencies and exclude common phrases
        phrase_freq = {}
        for phrase in phrases:
            if not any(w in common_words for w in phrase.split() if len(w) <= 3):
                if len(phrase.split()) > 1:  # Ensure it's a real phrase
                    phrase_freq[phrase] = phrase_freq.get(phrase, 0) + 1
        
        # Get top N phrases by frequency
        top_phrases = sorted(phrase_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Combine top words and phrases
        key_phrases = [word for word, _ in top_words]
        key_phrases.extend([phrase for phrase, _ in top_phrases])
        
        # Remove duplicates and limit to 10 items
        unique_key_phrases = []
        for phrase in key_phrases:
            if not any(phrase in p for p in unique_key_phrases):
                unique_key_phrases.append(phrase)
                if len(unique_key_phrases) >= 10:
                    break
        
        return unique_key_phrases
    
    def _parse_structured_summary(self, claude_response):
        """Parse Claude's structured response into components"""
        result = {
            'summary': '',
            'key_topics': [],
            'entities': [],
            'tone': '',
            'key_insights': [],
            'actionable_items': [],
            'content_type': ''
        }
        
        # Extract the summary
        summary_match = re.search(r'SUMMARY:\s*(.*?)(?=\n\n|\Z)', claude_response, re.DOTALL)
        if summary_match:
            result['summary'] = summary_match.group(1).strip()
        
        # Extract key topics
        topics_section = re.search(r'KEY TOPICS:(.*?)(?=\n\n|\Z)', claude_response, re.DOTALL)
        if topics_section:
            topics = re.findall(r'-\s*(.*?)(?=\n|$)', topics_section.group(1))
            result['key_topics'] = [topic.strip() for topic in topics if topic.strip()]
        
        # Extract entities
        entities_section = re.search(r'ENTITIES MENTIONED:(.*?)(?=\n\n|\Z)', claude_response, re.DOTALL)
        if entities_section:
            entities = re.findall(r'-\s*(.*?)(?=\n|$)', entities_section.group(1))
            result['entities'] = [entity.strip() for entity in entities if entity.strip()]
        
        # Extract tone
        tone_match = re.search(r'TONE & STYLE:\s*(.*?)(?=\n\n|\Z)', claude_response, re.DOTALL)
        if tone_match:
            result['tone'] = tone_match.group(1).strip()
        
        # Extract insights
        insights_section = re.search(r'KEY INSIGHTS:(.*?)(?=\n\n|\Z)', claude_response, re.DOTALL)
        if insights_section:
            insights = re.findall(r'-\s*(.*?)(?=\n|$)', insights_section.group(1))
            result['key_insights'] = [insight.strip() for insight in insights if insight.strip()]
        
        # Extract actionable items
        actionable_section = re.search(r'ACTIONABLE INFORMATION:(.*?)(?=\n\n|\Z)', claude_response, re.DOTALL)
        if actionable_section:
            actions = re.findall(r'-\s*(.*?)(?=\n|$)', actionable_section.group(1))
            result['actionable_items'] = [action.strip() for action in actions if action.strip()]
        
        # Extract content type
        content_match = re.search(r'CONTENT TYPE:\s*(.*?)(?=\n\n|\Z)', claude_response, re.DOTALL)
        if content_match:
            result['content_type'] = content_match.group(1).strip()
        
        return result
    
    def summarize(self, transcript, shortcode, metadata=None, max_retries=3):
        """Generate a summary using Claude API with retry logic"""
        # Check cache first
        if shortcode in self.cache:
            logger.info(f"Using cached summary for {shortcode}")
            return self.cache[shortcode]
        
        # If transcript is too short, use it as the summary
        if len(transcript.split()) < 30:
            logger.info(f"Transcript too short for {shortcode}, using as summary")
            
            # Even for short transcripts, provide some structure
            structured_response = {
                'summary': transcript,
                'key_topics': self._extract_key_phrases(transcript, transcript)[:3],
                'entities': [],
                'tone': 'Brief',
                'key_insights': [],
                'actionable_items': [],
                'content_type': 'Short clip'
            }
            
            summary_text = f"SUMMARY: {structured_response['summary']}\n\n"
            summary_text += "KEY TOPICS:\n" + "\n".join([f"- {topic}" for topic in structured_response['key_topics']])
            summary_text += "\n\nCONTENT TYPE: Short clip"
            
            self.cache[shortcode] = summary_text
            self._save_cache()
            return summary_text
        
        # Retry logic with exponential backoff
        retries = 0
        while retries <= max_retries:
            try:
                logger.info(f"Generating summary for {shortcode} (Attempt {retries+1}/{max_retries+1})")
                
                # Create enhanced prompt
                prompt = self._create_enhanced_prompt(transcript, metadata)
                
                # Use the current API pattern for Anthropic v0.49.0
                response = self.client.messages.create(
                    model="claude-3-haiku-20240307",  # More cost-effective model for batch processing
                    max_tokens=1024,
                    messages=[
                        {
                            "role": "user", 
                            "content": prompt
                        }
                    ]
                )
                
                # Get the text content from the response
                summary = response.content[0].text
                
                # Extract structured information from the response
                structured_data = self._parse_structured_summary(summary)
                
                # Extract key phrases if not already present in the response
                if not structured_data['key_topics'] or len(structured_data['key_topics']) < 3:
                    key_phrases = self._extract_key_phrases(transcript, structured_data['summary'])
                    structured_data['key_topics'] = key_phrases[:5]
                
                # Cache the result (store the full text response)
                self.cache[shortcode] = summary
                self._save_cache()
                
                logger.info(f"Successfully generated summary for {shortcode}")
                return summary
                
            except Exception as e:
                retries += 1
                logger.error(f"Error generating summary (Attempt {retries}/{max_retries+1}): {str(e)}")
                if retries <= max_retries:
                    wait_time = 2 ** retries + (retries * 2)  # Exponential backoff
                    logger.info(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to generate summary after {max_retries+1} attempts")
                    # Provide a basic structured response for failed summaries
                    basic_response = (
                        "SUMMARY: Summary generation failed due to API errors.\n\n"
                        "KEY TOPICS:\n- Unknown\n\n"
                        "CONTENT TYPE: Unknown"
                    )
                    return basic_response
        
        return "Summary generation failed"


def get_video_metadata(shortcode, conn):
    """Fetch metadata for a video from the database"""
    cursor = conn.cursor()
    cursor.execute(
        "SELECT account, timestamp, caption FROM videos WHERE shortcode = ?",
        (shortcode,)
    )
    result = cursor.fetchone()
    if result:
        return {
            'account': result['account'],
            'timestamp': result['timestamp'],
            'caption': result['caption']
        }
    return None


def analyze_content_quality(transcript):
    """Calculate metrics about content quality"""
    if not transcript:
        return {}
        
    # Calculate basic metrics
    words = transcript.split()
    word_count = len(words)
    
    # Estimate dialogue percentage (if there are colons, quotes, etc.)
    dialogue_markers = [":", "?", '"', "'", "says", "said", "asked"]
    dialogue_score = sum(transcript.count(marker) for marker in dialogue_markers) / max(1, len(transcript) / 100)
    
    # Calculate average word length (a proxy for complexity)
    avg_word_length = sum(len(word) for word in words) / max(1, word_count)
    
    # Estimate reading time (average person reads ~200-250 words per minute)
    reading_time_seconds = (word_count / 200) * 60
    
    return {
        "word_count": word_count,
        "dialogue_ratio": min(1.0, dialogue_score),
        "avg_word_length": avg_word_length,
        "estimated_read_time_seconds": reading_time_seconds
    }


def process_transcripts_with_claude(batch_size=10, delay_between_batches=5):
    """Process transcripts in batches using Claude API"""
    try:
        # Check for API key
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            logger.error("ANTHROPIC_API_KEY environment variable not set. Please set it before running the summarizer.")
            return
            
        summarizer = ClaudeSummarizer()
        
        # Get videos that need summarization
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Find videos with transcripts but no summaries
        cursor.execute(
            "SELECT id, shortcode, account, transcript FROM videos WHERE transcript IS NOT NULL AND transcript != '' AND (summary IS NULL OR summary = '')"
        )
        videos = cursor.fetchall()
        
        total_videos = len(videos)
        logger.info(f"Found {total_videos} videos that need summarization")
        
        if total_videos == 0:
            logger.info("No videos to summarize. Exiting.")
            conn.close()
            return
        
        # Process in batches
        for i in range(0, total_videos, batch_size):
            batch = videos[i:i+batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total_videos - 1) // batch_size + 1
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} videos)")
            
            for video in batch:
                video_id = video['id']
                shortcode = video['shortcode']
                account = video['account']
                transcript = video['transcript']
                
                # Get full metadata for enhanced context
                metadata = get_video_metadata(shortcode, conn)
                
                # Calculate content quality metrics
                quality_metrics = analyze_content_quality(transcript)
                word_count = quality_metrics.get('word_count', 0)
                
                logger.info(f"Summarizing video {shortcode} ({word_count} words)")
                summary = summarizer.summarize(transcript, shortcode, metadata=metadata)
                
                # Extract key phrases for database storage
                structured_data = summarizer._parse_structured_summary(summary)
                key_phrases_json = json.dumps(structured_data.get('key_topics', []))
                
                # Update database with summary, word count and content metrics
                cursor.execute(
                    """UPDATE videos SET 
                       summary = ?, 
                       word_count = ?,
                       key_phrases = ?,
                       duration_seconds = ?
                       WHERE id = ?""",
                    (summary, word_count, key_phrases_json, 
                     quality_metrics.get('estimated_read_time_seconds', 0), video_id)
                )
                conn.commit()
                logger.info(f"Updated summary and metrics for video {shortcode}")
            
            # Respect API rate limits between batches
            processed_count = min(i + batch_size, total_videos)
            logger.info(f"Processed {processed_count}/{total_videos} videos")
            
            if i + batch_size < total_videos:
                logger.info(f"Waiting {delay_between_batches} seconds before next batch")
                time.sleep(delay_between_batches)
        
        conn.close()
        logger.info("Summarization process complete")
        
    except Exception as e:
        logger.error(f"Error in batch processing: {str(e)}")
        raise


def summarize_transcripts():
    """Main entry point for transcript summarization"""
    logger.info("Starting transcript summarization using Claude")
    start_time = time.time()
    process_transcripts_with_claude()
    logger.info(f"Summarization completed in {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    summarize_transcripts() 