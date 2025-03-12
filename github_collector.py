"""
GitHub collector for AI/ML Knowledge Base
Collects and processes repositories from GitHub
"""
import os
import json
import logging
import time
import base64
import sqlite3
from datetime import datetime
import requests
from urllib.parse import urlparse
import random

from config import DATA_DIR, DB_PATH, PROXY_SERVERS

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(DATA_DIR, 'logs', 'github_collector.log'), mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('github_collector')

# Target repositories with high-quality AI/ML content
GITHUB_REPOS = [
    # Foundational ML/DL Libraries
    {"repo": "tensorflow/tensorflow", "resource_type": "code", "priority": "high"},
    {"repo": "pytorch/pytorch", "resource_type": "code", "priority": "high"},
    {"repo": "scikit-learn/scikit-learn", "resource_type": "code", "priority": "high"},
    
    # LLM & Generative AI
    {"repo": "huggingface/transformers", "resource_type": "code", "priority": "high"},
    {"repo": "openai/openai-cookbook", "resource_type": "tutorial", "priority": "high"},
    {"repo": "AUTOMATIC1111/stable-diffusion-webui", "resource_type": "application", "priority": "medium"},
    
    # Training & Infrastructure
    {"repo": "ray-project/ray", "resource_type": "code", "priority": "medium"},
    {"repo": "iterative/dvc", "resource_type": "tool", "priority": "medium"},
    
    # Research Implementations
    {"repo": "facebookresearch/llama", "resource_type": "code", "priority": "high"},
    {"repo": "microsoft/DeepSpeed", "resource_type": "code", "priority": "medium"},
    
    # Learning Resources
    {"repo": "deepmind/educational", "resource_type": "tutorial", "priority": "medium"},
    {"repo": "microsoft/ML-For-Beginners", "resource_type": "tutorial", "priority": "medium"},
]

def setup_directories():
    """Create necessary directories"""
    os.makedirs(os.path.join(DATA_DIR, 'github'), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, 'logs'), exist_ok=True)

def get_proxy():
    """Get a random proxy from the configured proxies"""
    if not PROXY_SERVERS:
        return None
    
    proxy_url = random.choice(PROXY_SERVERS)
    
    try:
        # Extract proxy username and password from URL
        parsed_url = urlparse(proxy_url)
        username = parsed_url.username or ""
        password = parsed_url.password or ""
        
        # Create proxy dictionary in the format required by requests
        scheme = parsed_url.scheme
        netloc = parsed_url.netloc
        if '@' in netloc:
            netloc = netloc.split('@')[1]  # Remove credentials from netloc
        
        proxies = {
            "http": f"{scheme}://{username}:{password}@{netloc}",
            "https": f"{scheme}://{username}:{password}@{netloc}"
        }
        
        logger.info(f"Using proxy: {netloc}")
        return proxies
    except Exception as e:
        logger.error(f"Error setting up proxy: {str(e)}")
        return None

def get_github_api_client(token=None):
    """
    Create a GitHub API client session
    
    If token is provided, use it for authentication
    Otherwise try to use the GITHUB_TOKEN environment variable
    """
    if not token:
        token = os.environ.get('GITHUB_TOKEN')
    
    if not token:
        logger.warning("No GitHub token provided. API rate limits will be severely restricted.")
    
    session = requests.Session()
    session.headers.update({
        'Accept': 'application/vnd.github.v3+json',
        'User-Agent': 'AI-Knowledge-Base-Collector/1.0'
    })
    
    if token:
        session.headers.update({'Authorization': f'token {token}'})
    
    # Set up proxy if available
    proxies = get_proxy()
    if proxies:
        session.proxies.update(proxies)
    
    return session

def collect_repo_info(session, repo_name):
    """
    Collect basic information about a repository
    
    Args:
        session: GitHub API session
        repo_name: Repository name in the format "owner/repo"
        
    Returns:
        Dictionary with repository information or None if failed
    """
    logger.info(f"Collecting info for repository: {repo_name}")
    
    try:
        # Get repository information
        response = session.get(f"https://api.github.com/repos/{repo_name}")
        response.raise_for_status()
        repo_data = response.json()
        
        # Get topics
        topics_response = session.get(f"https://api.github.com/repos/{repo_name}/topics")
        topics_response.raise_for_status()
        topics = topics_response.json().get('names', [])
        
        # Extract the information we need
        repo_info = {
            'name': repo_data['name'],
            'full_name': repo_data['full_name'],
            'description': repo_data['description'],
            'url': repo_data['html_url'],
            'stars': repo_data['stargazers_count'],
            'watchers': repo_data['watchers_count'],
            'forks': repo_data['forks_count'],
            'language': repo_data['language'],
            'last_push': repo_data['pushed_at'],
            'created_at': repo_data['created_at'],
            'updated_at': repo_data['updated_at'],
            'topics': json.dumps(topics),
            'last_crawled': datetime.now().isoformat()
        }
        
        return repo_info
    
    except Exception as e:
        logger.error(f"Error collecting info for repository {repo_name}: {str(e)}")
        return None

def collect_repo_readme(session, repo_name):
    """
    Collect the README file from a repository
    
    Args:
        session: GitHub API session
        repo_name: Repository name in the format "owner/repo"
        
    Returns:
        README content as string or None if not found
    """
    logger.info(f"Collecting README for repository: {repo_name}")
    
    # Try different README filenames
    readme_files = ['README.md', 'README.rst', 'README.txt', 'README']
    
    for readme_file in readme_files:
        try:
            # Get README content
            response = session.get(f"https://api.github.com/repos/{repo_name}/contents/{readme_file}")
            
            if response.status_code == 200:
                readme_data = response.json()
                
                # GitHub API returns content as base64 encoded
                if 'content' in readme_data and readme_data.get('encoding') == 'base64':
                    content = base64.b64decode(readme_data['content']).decode('utf-8')
                    return content
                
            # If we get a 404, try the next filename
            elif response.status_code == 404:
                continue
            
            # For other status codes, raise an exception
            else:
                response.raise_for_status()
        
        except Exception as e:
            logger.error(f"Error collecting README ({readme_file}) for repository {repo_name}: {str(e)}")
            
            # If we hit the rate limit, sleep and return None
            if response.status_code == 403 and 'rate limit' in response.text.lower():
                reset_time = int(response.headers.get('X-RateLimit-Reset', 0))
                wait_time = max(reset_time - time.time(), 0) + 10
                logger.warning(f"Rate limit hit. Waiting for {wait_time:.0f} seconds.")
                time.sleep(wait_time)
                return None
    
    # If we couldn't find any README, return None
    logger.warning(f"No README found for repository {repo_name}")
    return None

def process_readme_content(content, repo_name):
    """
    Process README content to make it more suitable for storage and analysis
    
    Args:
        content: README content as string
        repo_name: Repository name for context
        
    Returns:
        Processed content as string
    """
    if not content:
        return None
    
    # Remove very long code blocks to save space
    lines = content.split('\n')
    processed_lines = []
    in_code_block = False
    code_block_length = 0
    max_code_block_length = 50  # Maximum lines to keep in a code block
    
    for line in lines:
        # Check for code block markers
        if line.strip().startswith('```'):
            in_code_block = not in_code_block
            
            # Reset code block length counter if we're ending a block
            if not in_code_block:
                code_block_length = 0
            
            processed_lines.append(line)
        elif in_code_block:
            code_block_length += 1
            
            # Skip if we've exceeded the maximum code block length
            if code_block_length <= max_code_block_length:
                processed_lines.append(line)
            elif code_block_length == max_code_block_length + 1:
                # Add a note that we're truncating the code block
                processed_lines.append("... [code block truncated] ...")
        else:
            processed_lines.append(line)
    
    processed_content = '\n'.join(processed_lines)
    
    # Add context about the repository
    header = f"# {repo_name}\n\n"
    
    return header + processed_content

def store_repo_in_db(repo_info, readme_content):
    """
    Store repository information in the database
    
    Args:
        repo_info: Dictionary with repository information
        readme_content: Processed README content
        
    Returns:
        Boolean indicating success
    """
    if not repo_info:
        return False
    
    # Add README content to repo_info
    repo_info['readme'] = readme_content
    
    try:
        # Connect to the database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check if the repository already exists
        cursor.execute("SELECT id FROM github_repos WHERE full_name = ?", (repo_info['full_name'],))
        existing_id = cursor.fetchone()
        
        if existing_id:
            # Update existing repository
            repo_id = existing_id[0]
            
            # Create placeholders for the update query
            set_clause = ', '.join([f"{key} = ?" for key in repo_info.keys()])
            values = list(repo_info.values())
            
            # Update the repository
            cursor.execute(f"UPDATE github_repos SET {set_clause} WHERE id = ?", values + [repo_id])
            logger.info(f"Updated repository {repo_info['full_name']} in database")
        else:
            # Insert new repository
            columns = ', '.join(repo_info.keys())
            placeholders = ', '.join(['?'] * len(repo_info))
            
            cursor.execute(f"INSERT INTO github_repos ({columns}) VALUES ({placeholders})", 
                          list(repo_info.values()))
            repo_id = cursor.lastrowid
            logger.info(f"Inserted repository {repo_info['full_name']} into database")
        
        # Also add to the unified ai_content table
        if readme_content:
            # Get GitHub source type ID
            cursor.execute("SELECT id FROM source_types WHERE name = 'github'")
            github_source_type_id = cursor.fetchone()[0]
            
            # Check if already exists in ai_content
            cursor.execute("SELECT id FROM ai_content WHERE source_type_id = ? AND source_id = ?", 
                          (github_source_type_id, repo_info['full_name']))
            existing_content_id = cursor.fetchone()
            
            if existing_content_id:
                # Update existing content
                cursor.execute("""
                UPDATE ai_content SET 
                    title = ?, 
                    content = ?, 
                    url = ?, 
                    timestamp = ?, 
                    star_count = ?, 
                    last_updated = ?
                WHERE id = ?
                """, (
                    repo_info['description'] or repo_info['name'],
                    readme_content,
                    repo_info['url'],
                    repo_info['updated_at'],
                    repo_info['stars'],
                    datetime.now().isoformat(),
                    existing_content_id[0]
                ))
                logger.info(f"Updated repository {repo_info['full_name']} in ai_content table")
            else:
                # Insert new content
                cursor.execute("""
                INSERT INTO ai_content (
                    source_type_id, 
                    source_id, 
                    title, 
                    content, 
                    url, 
                    timestamp, 
                    star_count, 
                    last_updated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    github_source_type_id,
                    repo_info['full_name'],
                    repo_info['description'] or repo_info['name'],
                    readme_content,
                    repo_info['url'],
                    repo_info['updated_at'],
                    repo_info['stars'],
                    datetime.now().isoformat()
                ))
                logger.info(f"Inserted repository {repo_info['full_name']} into ai_content table")
        
        # Commit changes and close connection
        conn.commit()
        conn.close()
        
        return True
    
    except Exception as e:
        logger.error(f"Error storing repository {repo_info['full_name']} in database: {str(e)}")
        return False

def collect_github_repos(token=None, max_repos=None):
    """
    Main function to collect GitHub repositories
    
    Args:
        token: GitHub API token
        max_repos: Maximum number of repositories to collect (None for all)
        
    Returns:
        Number of successfully processed repositories
    """
    setup_directories()
    
    # Limit the number of repositories if specified
    repos_to_process = GITHUB_REPOS
    if max_repos is not None:
        repos_to_process = GITHUB_REPOS[:max_repos]
    
    # Create GitHub API client
    session = get_github_api_client(token)
    
    # Process repositories
    success_count = 0
    
    for repo_config in repos_to_process:
        repo_name = repo_config['repo']
        
        try:
            # Add a delay between requests to avoid rate limiting
            time.sleep(random.uniform(1, 3))
            
            # Collect repository information
            repo_info = collect_repo_info(session, repo_name)
            
            if not repo_info:
                logger.warning(f"Failed to collect information for repository {repo_name}")
                continue
            
            # Collect README content
            readme_content = collect_repo_readme(session, repo_name)
            
            # Process README content if available
            if readme_content:
                processed_readme = process_readme_content(readme_content, repo_name)
            else:
                processed_readme = None
            
            # Store repository in the database
            if store_repo_in_db(repo_info, processed_readme):
                success_count += 1
            
            logger.info(f"Successfully processed repository {repo_name}")
            
        except Exception as e:
            logger.error(f"Error processing repository {repo_name}: {str(e)}")
    
    logger.info(f"Completed GitHub repository collection. Processed {success_count}/{len(repos_to_process)} repositories.")
    return success_count

if __name__ == "__main__":
    # Check for GitHub token
    github_token = os.environ.get('GITHUB_TOKEN')
    if not github_token:
        logger.warning("No GITHUB_TOKEN environment variable found. API rate limits will be severely restricted.")
    
    # Collect repositories
    collect_github_repos(token=github_token) 