"""
Module for collecting and processing AI/ML research papers from ArXiv
"""
import os
import json
import time
import logging
import sqlite3
import requests
import feedparser
from datetime import datetime, timedelta
import PyPDF2
import io
import config

# Configure logging
log_dir = os.path.join(config.DATA_DIR, 'logs')
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'arxiv_collector.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('arxiv_collector')

def setup_directories():
    """Create necessary directories for storing paper data"""
    papers_dir = os.path.join(config.DATA_DIR, "papers")
    papers_pdf_dir = os.path.join(papers_dir, "pdfs")
    papers_text_dir = os.path.join(papers_dir, "text")
    
    os.makedirs(papers_dir, exist_ok=True)
    os.makedirs(papers_pdf_dir, exist_ok=True)
    os.makedirs(papers_text_dir, exist_ok=True)
    
    return papers_dir, papers_pdf_dir, papers_text_dir

def extract_text_from_pdf(pdf_path):
    """Extract text content from a PDF file"""
    try:
        text = ""
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            num_pages = len(reader.pages)
            
            # Extract text from each page
            for page_num in range(num_pages):
                page = reader.pages[page_num]
                text += page.extract_text() + "\n\n"
                
            logger.info(f"Extracted text from {pdf_path} ({num_pages} pages)")
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF {pdf_path}: {str(e)}")
        return None

def parse_sections(text):
    """Attempt to parse sections from extracted PDF text"""
    sections = {
        "abstract": "",
        "introduction": "",
        "methodology": "",
        "results": "",
        "conclusion": "",
        "references": ""
    }
    
    if not text:
        return sections
        
    # Simple heuristic section detection
    lines = text.split('\n')
    current_section = "abstract"
    
    for line in lines:
        line = line.strip()
        line_lower = line.lower()
        
        # Check for section headers
        if "abstract" in line_lower and len(line) < 30:
            current_section = "abstract"
            continue
        elif any(x in line_lower for x in ["introduction", "background"]) and len(line) < 30:
            current_section = "introduction"
            continue
        elif any(x in line_lower for x in ["method", "approach", "model", "implementation"]) and len(line) < 30:
            current_section = "methodology"
            continue
        elif any(x in line_lower for x in ["result", "evaluation", "experiment", "performance"]) and len(line) < 30:
            current_section = "results"
            continue
        elif any(x in line_lower for x in ["conclusion", "discussion", "future work"]) and len(line) < 30:
            current_section = "conclusion"
            continue
        elif any(x in line_lower for x in ["reference", "bibliography"]) and len(line) < 30:
            current_section = "references"
            continue
            
        # Add text to current section
        if line and current_section in sections:
            sections[current_section] += line + "\n"
    
    # Clean up sections
    for section in sections:
        if len(sections[section]) > 100000:  # Limit section size
            sections[section] = sections[section][:100000] + "... [truncated]"
    
    return sections

def collect_papers(max_papers=None, force_update=False):
    """
    Collect papers from ArXiv based on configured topics
    
    Args:
        max_papers: Maximum number of papers to collect (None for no limit)
        force_update: Whether to force update of existing papers
        
    Returns:
        Number of new papers added
    """
    papers_dir, papers_pdf_dir, papers_text_dir = setup_directories()
    
    # Connect to database
    conn = sqlite3.connect(config.DB_PATH)
    
    # Get source type ID for 'research_paper'
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM source_types WHERE name = 'research_paper'")
    result = cursor.fetchone()
    if result:
        source_type_id = result[0]
    else:
        logger.error("Source type 'research_paper' not found in database")
        conn.close()
        return 0
    
    papers_processed = 0
    papers_added = 0
    paper_topics = config.RESEARCH_PAPER_CONFIG.get('topics', [
        "large language models",
        "diffusion models",
        "transformers", 
        "generative ai",
        "reinforcement learning",
        "computer vision"
    ])
    
    max_papers_per_topic = config.RESEARCH_PAPER_CONFIG.get('max_papers_per_topic', 5)
    max_age_days = config.RESEARCH_PAPER_CONFIG.get('max_age_days', 60)
    
    # Adjust max_papers_per_topic if max_papers is specified
    if max_papers and len(paper_topics) > 0:
        max_papers_per_topic = min(max_papers_per_topic, max_papers // len(paper_topics) + 1)
    
    # Process each configured topic
    for topic in paper_topics:
        logger.info(f"Collecting papers for topic: {topic}")
        
        # Construct ArXiv API query
        query = f'all:"{topic}" AND (cat:cs.AI OR cat:cs.LG OR cat:cs.CL OR cat:cs.CV)'
        url = f"http://export.arxiv.org/api/query?search_query={query.replace(' ', '+')}&sortBy=submittedDate&sortOrder=descending&max_results={max_papers_per_topic}"
        
        try:
            # Get papers from ArXiv
            response = feedparser.parse(url)
            
            if not response.entries:
                logger.warning(f"No papers found for topic: {topic}")
                continue
                
            logger.info(f"Found {len(response.entries)} papers for topic: {topic}")
            
            # Process each paper
            for entry in response.entries:
                # Debug to see the structure of the entry
                entry_keys = entry.keys()
                logger.debug(f"Available keys in entry: {entry_keys}")
                
                try:
                    # Safe extraction of paper ID
                    paper_id = entry.id.split('/abs/')[-1] if hasattr(entry, 'id') else f"unknown_{time.time()}"
                    papers_processed += 1
                    
                    if max_papers and papers_added >= max_papers:
                        logger.info(f"Reached maximum paper limit: {max_papers}")
                        break
                    
                    # Check if paper already exists
                    cursor.execute(
                        "SELECT id, last_crawled FROM research_papers WHERE doi = ?", 
                        (paper_id,)
                    )
                    existing = cursor.fetchone()
                    
                    # Skip if paper exists and was recently updated (unless force_update is True)
                    if existing and not force_update:
                        if existing[1]:  # Check if last_crawled is not None
                            try:
                                last_updated = datetime.fromisoformat(existing[1].replace('Z', '+00:00'))
                                if (datetime.now() - last_updated).days < 30:
                                    logger.debug(f"Skipping recent paper: {paper_id}")
                                    continue
                            except (ValueError, AttributeError):
                                # If date parsing fails, process the paper anyway
                                pass
                    
                    # Safe extraction of publication date
                    published = datetime.now()  # Default to current time
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        try:
                            published_date = entry.published_parsed
                            published = datetime(
                                year=published_date[0],
                                month=published_date[1],
                                day=published_date[2],
                                hour=published_date[3],
                                minute=published_date[4],
                                second=published_date[5]
                            )
                        except (IndexError, TypeError, ValueError) as e:
                            logger.warning(f"Error parsing published date for {paper_id}: {str(e)}")
                    
                    if not force_update and (datetime.now() - published).days > max_age_days:
                        logger.debug(f"Skipping old paper: {paper_id} ({published})")
                        continue
                    
                    # Safe extraction of paper details
                    title = entry.get('title', f"Unknown Paper {paper_id}")
                    
                    # Safe extraction of authors
                    authors = ""
                    if hasattr(entry, 'authors'):
                        try:
                            authors = ", ".join(author.get('name', '') for author in entry.authors)
                        except (AttributeError, TypeError) as e:
                            logger.warning(f"Error extracting authors for {paper_id}: {str(e)}")
                    
                    # Safe extraction of summary
                    summary = entry.get('summary', "No abstract available")
                    
                    # Get PDF link
                    pdf_link = None
                    if hasattr(entry, 'links'):
                        for link in entry.links:
                            if hasattr(link, 'rel') and hasattr(link, 'type'):
                                if link.rel == 'alternate' and link.type == 'application/pdf':
                                    pdf_link = link.href
                                    break
                            if hasattr(link, 'title') and link.title == 'pdf':
                                pdf_link = link.href
                                break
                    
                    # Extract article URL
                    article_url = None
                    if hasattr(entry, 'links'):
                        for link in entry.links:
                            if hasattr(link, 'rel') and hasattr(link, 'type'):
                                if link.rel == 'alternate' and link.type == 'text/html':
                                    article_url = link.href
                                    break
                    
                    if not article_url:
                        article_url = f"https://arxiv.org/abs/{paper_id}"
                    
                    # Extract PDF content if available
                    sections = {"abstract": summary}
                    pdf_path = None
                    extracted_text = None
                    
                    if pdf_link:
                        try:
                            # Save PDF locally
                            pdf_path = os.path.join(papers_pdf_dir, f"{paper_id.replace('/', '_')}.pdf")
                            if not os.path.exists(pdf_path) or force_update:
                                logger.info(f"Downloading PDF for {paper_id} from {pdf_link}")
                                
                                # Add a delay to be respectful of the API
                                time.sleep(2)
                                
                                pdf_response = requests.get(pdf_link, timeout=30)
                                with open(pdf_path, 'wb') as f:
                                    f.write(pdf_response.content)
                                logger.info(f"Downloaded PDF for {paper_id}")
                            
                            # Extract text from PDF
                            extracted_text = extract_text_from_pdf(pdf_path)
                            if extracted_text:
                                # Save extracted text
                                text_path = os.path.join(papers_text_dir, f"{paper_id.replace('/', '_')}.txt")
                                with open(text_path, 'w', encoding='utf-8') as f:
                                    f.write(extracted_text)
                                
                                # Parse sections
                                sections = parse_sections(extracted_text)
                                if not sections["abstract"].strip():
                                    sections["abstract"] = summary
                            
                        except Exception as e:
                            logger.error(f"Error processing PDF for {paper_id}: {str(e)}")
                    
                    # Calculate publication year
                    publication_year = published.year
                    
                    # Prepare data for database
                    if existing:
                        # Update existing paper record
                        cursor.execute("""
                            UPDATE research_papers
                            SET title = ?, authors = ?, abstract = ?, 
                                url = ?, pdf_path = ?, content = ?, last_crawled = ?
                            WHERE id = ?
                        """, (
                            title,
                            authors,
                            sections["abstract"],
                            article_url,
                            pdf_path if pdf_path else None,
                            extracted_text,
                            datetime.now().isoformat(),
                            existing[0]
                        ))
                        
                        # Update ai_content record
                        cursor.execute("""
                            UPDATE ai_content
                            SET title = ?, description = ?, content = ?,
                                url = ?, date_collected = ?, metadata = ?
                            WHERE source_type_id = ? AND source_id = ?
                        """, (
                            title,
                            sections["abstract"],
                            extracted_text if extracted_text else sections["abstract"],
                            article_url,
                            datetime.now().isoformat(),
                            json.dumps({
                                "authors": authors,
                                "published_date": published.isoformat(),
                                "year": publication_year,
                                "sections": sections
                            }),
                            source_type_id,
                            str(existing[0])
                        ))
                        
                        logger.info(f"Updated paper: {title}")
                    else:
                        # Insert into research_papers
                        cursor.execute("""
                            INSERT INTO research_papers (
                                title, authors, abstract, publication, year,
                                url, doi, pdf_path, content, last_crawled
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            title,
                            authors,
                            sections["abstract"],
                            "arXiv",  # Publication name
                            publication_year,
                            article_url,
                            paper_id,  # Using arXiv ID as DOI
                            pdf_path if pdf_path else None,
                            extracted_text,
                            datetime.now().isoformat()
                        ))
                        
                        # Get the inserted ID
                        paper_db_id = cursor.lastrowid
                        
                        # Insert into ai_content
                        cursor.execute("""
                            INSERT INTO ai_content (
                                source_type_id, source_id, title, description, content,
                                url, date_created, date_collected, metadata, is_indexed
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            source_type_id,
                            str(paper_db_id),
                            title,
                            sections["abstract"],
                            extracted_text if extracted_text else sections["abstract"],
                            article_url,
                            published.isoformat(),
                            datetime.now().isoformat(),
                            json.dumps({
                                "authors": authors,
                                "published_date": published.isoformat(),
                                "year": publication_year,
                                "sections": sections
                            }),
                            0  # Not indexed yet
                        ))
                        
                        papers_added += 1
                        logger.info(f"Added new paper: {title}")
                    
                    conn.commit()
                
                except Exception as e:
                    logger.error(f"Error processing paper entry: {str(e)}")
                
                # Add a delay to be respectful of the API
                time.sleep(3)
                
        except Exception as e:
            logger.error(f"Error collecting papers for topic {topic}: {str(e)}")
            continue
    
    conn.close()
    logger.info(f"Paper collection complete. Processed {papers_processed} papers, added {papers_added} new papers.")
    return papers_added

if __name__ == "__main__":
    collect_papers() 