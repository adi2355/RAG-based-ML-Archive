"""
Module for downloading Instagram content with proper rate limiting
"""
import os
import time
import json
import logging
import random
from datetime import datetime, timedelta
import instaloader

# Add imports for proxy testing
import requests
from urllib.parse import urlparse
from urllib.parse import parse_qs
import socket
import json as json_lib
import urllib3

# Disable SSL warnings for testing
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from config import (
    INSTAGRAM_ACCOUNTS, 
    INSTAGRAM_USERNAME, 
    INSTAGRAM_PASSWORD,
    INSTAGRAM_ACCOUNT_ROTATION,
    PROXY_SERVERS,
    PROXY_COUNTRY,
    DOWNLOAD_DIR, 
    DOWNLOAD_DELAY, 
    MAX_DOWNLOADS_PER_RUN,
    DATA_DIR,
    ACCOUNT_COOLDOWN_MINUTES,
    PROXY_COOLDOWN_MINUTES
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('downloader.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('downloader')

def setup_directories():
    """Create necessary directories if they don't exist"""
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "state"), exist_ok=True)
    
    # Create account-specific directories
    for account in INSTAGRAM_ACCOUNTS:
        account_dir = os.path.join(DOWNLOAD_DIR, account["username"])
        os.makedirs(account_dir, exist_ok=True)

def get_random_delay():
    """Return a more human-like delay between actions"""
    # Base delay plus random variation to appear more human-like
    return DOWNLOAD_DELAY + random.uniform(-2, 5)

def get_next_account():
    """Get the next available account from rotation"""
    # If no accounts in rotation, use the default credentials
    if not INSTAGRAM_ACCOUNT_ROTATION:
        return INSTAGRAM_USERNAME, INSTAGRAM_PASSWORD
    
    state_file = os.path.join(DATA_DIR, "state", "account_state.json")
    if os.path.exists(state_file):
        try:
            with open(state_file) as f:
                state = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            state = {"last_index": -1, "account_states": {}}
    else:
        state = {"last_index": -1, "account_states": {}}
    
    # Find the next available account
    for i in range(len(INSTAGRAM_ACCOUNT_ROTATION)):
        idx = (state["last_index"] + i + 1) % len(INSTAGRAM_ACCOUNT_ROTATION)
        account = INSTAGRAM_ACCOUNT_ROTATION[idx]
        username = account["username"]
        
        # Skip accounts that are in cooldown
        account_state = state["account_states"].get(username, {})
        next_available_str = account_state.get("next_available")
        
        if not next_available_str or datetime.now().isoformat() >= next_available_str:
            # Update state
            state["last_index"] = idx
            with open(state_file, "w") as f:
                json.dump(state, f)
            
            logger.info(f"Using account {username} from rotation")
            return account["username"], account["password"]
    
    # If all accounts are in cooldown, fallback to default account
    logger.warning("All accounts in rotation are in cooldown, using default account")
    return INSTAGRAM_USERNAME, INSTAGRAM_PASSWORD

def mark_account_cooldown(username, cooldown_minutes=ACCOUNT_COOLDOWN_MINUTES):
    """Mark an account as in cooldown after a failure"""
    if not username:
        return
        
    state_file = os.path.join(DATA_DIR, "state", "account_state.json")
    if os.path.exists(state_file):
        try:
            with open(state_file) as f:
                state = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            state = {"last_index": -1, "account_states": {}}
    else:
        state = {"last_index": -1, "account_states": {}}
    
    # Set cooldown period
    next_available = (datetime.now() + timedelta(minutes=cooldown_minutes)).isoformat()
    
    # Update account state
    if username not in state["account_states"]:
        state["account_states"][username] = {}
        
    state["account_states"][username]["next_available"] = next_available
    state["account_states"][username]["last_failure"] = datetime.now().isoformat()
    
    # Save state
    with open(state_file, "w") as f:
        json.dump(state, f)
        
    logger.info(f"Account {username} marked for cooldown until {next_available}")

def get_proxy(country=None):
    """
    Get a proxy from the available pool
    
    Args:
        country: Optional two-letter country code (us, uk, etc.)
    """
    # If no proxies configured, return None
    if not PROXY_SERVERS:
        return None
    
    # Use country from config if not specified in function call
    if country is None and PROXY_COUNTRY:
        country = PROXY_COUNTRY
        
    state_file = os.path.join(DATA_DIR, "state", "proxy_state.json")
    
    # Make sure the directory exists
    os.makedirs(os.path.dirname(state_file), exist_ok=True)
    
    # Load proxy state
    if os.path.exists(state_file):
        try:
            with open(state_file) as f:
                state = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            state = {"last_index": -1, "proxy_states": {}}
    else:
        state = {"last_index": -1, "proxy_states": {}}
    
    # Find the next available proxy
    for i in range(len(PROXY_SERVERS)):
        idx = (state["last_index"] + i + 1) % len(PROXY_SERVERS)
        base_proxy = PROXY_SERVERS[idx]
        
        # If country is specified, modify the proxy URL
        proxy = base_proxy
        if country and "zone-residential" in base_proxy:
            # Extract components from the proxy URL
            parts = base_proxy.split('@')
            if len(parts) == 2:
                auth_part = parts[0]
                host_part = parts[1]
                
                # Check if country parameter is already in the auth part
                if "country-" in auth_part:
                    # Replace existing country
                    auth_parts = auth_part.split('-country-')
                    if len(auth_parts) == 2:
                        country_and_after = auth_parts[1].split(':', 1)
                        if len(country_and_after) == 2:
                            new_auth = f"{auth_parts[0]}-country-{country}:{country_and_after[1]}"
                            proxy = f"{new_auth}@{host_part}"
                else:
                    # Add country before the password
                    auth_parts = auth_part.split(':')
                    if len(auth_parts) >= 2:
                        password_idx = len(auth_parts) - 1
                        auth_parts[password_idx] = f"country-{country}:{auth_parts[password_idx].split(':')[-1]}"
                        proxy = f"{':'.join(auth_parts)}@{host_part}"
        
        # Skip proxies that are in cooldown
        proxy_state = state["proxy_states"].get(proxy, {})
        next_available_str = proxy_state.get("next_available")
        
        if not next_available_str or datetime.now().isoformat() >= next_available_str:
            # Update state
            state["last_index"] = idx
            with open(state_file, "w") as f:
                json.dump(state, f)
            
            # Test the proxy before returning
            if test_proxy(proxy):
                logger.info(f"Using proxy: {proxy}")
                return proxy
            else:
                # Mark as in cooldown if test fails
                mark_proxy_cooldown(proxy, cooldown_minutes=30)
                logger.warning(f"Proxy test failed, marking for cooldown: {proxy}")
                continue
    
    # If all proxies are in cooldown, log warning and return None
    logger.warning("All proxies are in cooldown or not working, proceeding without proxy")
    return None

def test_proxy(proxy_url, test_url="https://www.instagram.com/favicon.ico", timeout=30):
    """Test if a proxy server is working correctly"""
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
        
        # Test the proxy by making a request to the test URL
        response = requests.get(test_url, proxies=proxies, timeout=timeout, verify=False)
        
        if response.status_code == 200:
            logger.info(f"Proxy test successful: {netloc}")
            return True
        else:
            logger.warning(f"Proxy test failed with status code {response.status_code}: {netloc}")
            return False
            
    except Exception as e:
        logger.error(f"Error testing proxy: {str(e)}")
        return False

def mark_proxy_cooldown(proxy, cooldown_minutes=PROXY_COOLDOWN_MINUTES):
    """Mark a proxy as in cooldown after a failure"""
    if not proxy:
        return
        
    state_file = os.path.join(DATA_DIR, "state", "proxy_state.json")
    if os.path.exists(state_file):
        try:
            with open(state_file) as f:
                state = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            state = {"last_index": -1, "proxy_states": {}}
    else:
        state = {"last_index": -1, "proxy_states": {}}
    
    # Set cooldown period
    next_available = (datetime.now() + timedelta(minutes=cooldown_minutes)).isoformat()
    
    # Update proxy state
    if proxy not in state["proxy_states"]:
        state["proxy_states"][proxy] = {}
        
    state["proxy_states"][proxy]["next_available"] = next_available
    state["proxy_states"][proxy]["last_failure"] = datetime.now().isoformat()
    
    # Save state
    with open(state_file, "w") as f:
        json.dump(state, f)
        
    logger.info(f"Proxy {proxy} marked for cooldown until {next_available}")

def login_with_session(L, username, password):
    """Login with proper session management and error handling"""
    session_file = os.path.join(DATA_DIR, "state", f"insta_session_{username}.txt")
    
    # Try to load existing session first
    if os.path.exists(session_file) and username:
        try:
            L.load_session_from_file(username, session_file)
            logger.info(f"Loaded existing session for {username}")
            
            # Test the session validity by trying a simple operation
            try:
                test_profile = instaloader.Profile.from_username(L.context, username)
                logger.info("Session validation successful")
                return True
            except Exception:
                logger.warning("Loaded session is invalid, will login again")
        except Exception as e:
            logger.warning(f"Could not load session: {str(e)}")
    
    # If we need to login again
    if not username or not password:
        logger.warning("No login credentials provided")
        return False
        
    try:
        # Add random delay before login to look more human-like
        delay = random.uniform(1, 3)
        logger.info(f"Waiting {delay:.1f}s before login attempt...")
        time.sleep(delay)
        
        L.login(username, password)
        
        # Save the session for future use
        L.save_session_to_file(session_file)
        logger.info(f"Login successful and session saved for {username}")
        return True
    except Exception as e:
        logger.error(f"Login failed for {username}: {str(e)}")
        # Mark account for cooldown after failure
        mark_account_cooldown(username)
        return False

def create_instaloader_instance(use_login=True, account=None, proxy=None):
    """Create an Instaloader instance with appropriate settings for our use case"""
    
    # Get a random user agent to appear more human-like
    user_agent = get_random_user_agent()
    
    # Create an Instaloader instance with our required settings
    loader = instaloader.Instaloader(
        download_videos=True,
        download_video_thumbnails=False,
        download_geotags=False,
        download_comments=False,
        save_metadata=True,
        compress_json=False,
        user_agent=user_agent,
        max_connection_attempts=3,
        sleep=True,  # Respect Instagram's rate limits
    )
    
    logger.info(f"Initialized Instaloader with user agent: {user_agent[:30]}...")
    
    # If a proxy is provided, set it on the session
    if proxy:
        try:
            # Extract proxy username and password from URL
            parsed_url = urlparse(proxy)
            username = parsed_url.username or ""
            password = parsed_url.password or ""
            
            # Create proxy string in the format required by requests
            scheme = parsed_url.scheme
            netloc = parsed_url.netloc
            if '@' in netloc:
                netloc = netloc.split('@')[1]  # Remove credentials from netloc
            
            proxy_str = f"{scheme}://{username}:{password}@{netloc}"
            
            # Set the proxy on the session
            loader.context._session.proxies = {
                "http": proxy_str,
                "https": proxy_str
            }
            logger.info(f"Set proxy on Instaloader session: {netloc}")
        except Exception as e:
            logger.error(f"Error setting proxy on Instaloader session: {str(e)}")
    
    # Log in if requested and credentials are available
    if use_login:
        if account is None:
            # Use default credentials if no specific account is provided
            username = INSTAGRAM_USERNAME
            password = INSTAGRAM_PASSWORD
        else:
            # Use the provided account credentials
            username = account.get('username')
            password = account.get('password')
        
        if username and password:
            try:
                # Add a random delay before login to avoid detection
                delay = random.uniform(1, 3)
                logger.debug(f"Adding random delay of {delay:.2f}s before login attempt")
                time.sleep(delay)
                
                loader.login(username, password)
                logger.info(f"Logged in as {username}")
            except Exception as e:
                logger.error(f"Login failed for {username}: {str(e)}")
    
    return loader

def retry_with_backoff(func, max_retries=3, initial_delay=5):
    """Execute a function with exponential backoff retries"""
    retries = 0
    while retries <= max_retries:
        try:
            return func()
        except Exception as e:  # Use generic Exception instead of specific ones
            retries += 1
            if retries > max_retries:
                raise
            wait_time = initial_delay * (2 ** retries) + random.uniform(1, 5)
            logger.warning(f"Retry {retries}/{max_retries} after error: {str(e)}. Waiting {wait_time:.1f}s")
            time.sleep(wait_time)
    return None

def download_from_instagram(accounts=None):
    """
    Download content from Instagram accounts with proper rate limiting
    and proxy rotation
    """
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    downloaded_count = 0
    success_count = 0
    
    # Use the accounts from config if none are provided
    if accounts is None:
        accounts = INSTAGRAM_ACCOUNTS
    
    # Store failed accounts to retry later
    failed_accounts = []
    
    # Randomize the account order to distribute load
    accounts_to_process = list(accounts)
    random.shuffle(accounts_to_process)
    
    # Process each target account
    for account_idx, account_info in enumerate(accounts_to_process):
        # Extract the account name depending on the type
        if isinstance(account_info, dict):
            account_name = account_info.get("username")
        else:
            account_name = account_info
            
        if not account_name:
            logger.warning(f"Skipping invalid account info: {account_info}")
            continue
            
        # Check if we've reached the download limit
        if downloaded_count >= MAX_DOWNLOADS_PER_RUN:
            logger.info(f"Reached maximum download limit of {MAX_DOWNLOADS_PER_RUN}")
            break
            
        # Skip accounts that are not due for refresh
        if not is_account_due_for_refresh(account_name):
            logger.info(f"Skipping account {account_name} - not due for refresh")
            continue
            
        # Get a proxy if available
        proxy = get_proxy()
        if not proxy:
            logger.warning("No proxy available, proceeding without proxy")
        else:
            logger.info(f"Using proxy: {proxy}")
        
        # Log the attempt
        logger.info(f"Processing account {account_idx+1}/{len(accounts_to_process)}: {account_name}")
        
        # Add a delay before processing to avoid detection
        delay = random.uniform(2, 5)
        logger.debug(f"Adding random delay of {delay:.2f}s before processing account")
        time.sleep(delay)
        
        # Try up to 2 different methods to download content
        profile = None
        posts = None
        attempt_count = 0
        max_attempts = 2
        
        while attempt_count < max_attempts and posts is None:
            attempt_count += 1
            logger.info(f"Attempt {attempt_count}/{max_attempts} for account {account_name}")
            
            try:
                # Create Instaloader instance
                use_login = (attempt_count > 1)  # Try without login first, then with login if available
                L = create_instaloader_instance(use_login=use_login, proxy=proxy)
                
                # Try to get profile
                if profile is None:
                    profile = instaloader.Profile.from_username(L.context, account_name)
                    
                    # Check if the profile has posts
                    if not hasattr(profile, 'get_posts'):
                        logger.error(f"Profile {account_name} does not have get_posts method")
                        raise ValueError(f"Invalid profile structure for {account_name}")
                    
                    # Mark this account as processed now (whether it succeeds or fails)
                    mark_account_processed(account_name)
                    
                    # Handle private account
                    if profile.is_private and not use_login:
                        logger.warning(f"Account {account_name} is private - will retry with login if credentials available")
                        continue  # Skip to next attempt (which will use login)
                    
                    logger.info(f"Successfully retrieved profile for {account_name}")
                
                # Get the iterator for posts with proper error handling
                try:
                    # Add a random delay before fetching posts to appear more human-like
                    time.sleep(random.uniform(1, 3))
                    
                    # Get posts iterator
                    posts = profile.get_posts()
                    
                    # Try to access the first post to validate the iterator
                    next(posts)
                    # Reset the iterator
                    posts = profile.get_posts()
                    
                except StopIteration:
                    logger.warning(f"No posts found for account {account_name}")
                    posts = []  # Empty list to indicate success but no posts
                    
                except Exception as e:
                    error_msg = str(e).lower()
                    logger.error(f"Error getting posts for {account_name} (attempt {attempt_count}): {str(e)}")
                    
                    # Check for specific errors that indicate we should try alternative approaches
                    if "401" in error_msg or "unauthorized" in error_msg:
                        if attempt_count < max_attempts:
                            logger.info(f"401 Unauthorized error - will retry with different approach")
                            # Mark previous proxy for cooldown and get a new one for next attempt
                            if proxy:
                                mark_proxy_cooldown(proxy)
                                proxy = get_proxy()
                            
                            posts = None  # Reset so we try again
                            time.sleep(5)  # Wait before retrying
                            continue
                    
                    # If we get here, we couldn't get posts after all retries
                    failed_accounts.append(account_name)
                    break
                
                # Process posts if we have them
                if posts is not None:
                    process_posts(L, profile, account_name, posts, 
                                 downloaded_count, success_count)
                    
            except Exception as e:
                logger.error(f"Error processing account {account_name} (attempt {attempt_count}): {str(e)}")
                
                if attempt_count < max_attempts:
                    # Mark previous proxy for cooldown and get a new one for next attempt
                    if proxy:
                        mark_proxy_cooldown(proxy)
                        proxy = get_proxy()
                    time.sleep(5)  # Wait before retrying
                else:
                    failed_accounts.append(account_name)
    
    # Log summary
    logger.info(f"Download session completed. Downloaded {success_count} videos from {len(accounts_to_process) - len(failed_accounts)}/{len(accounts_to_process)} accounts")
    
    if failed_accounts:
        logger.warning(f"Failed to process {len(failed_accounts)} accounts: {', '.join(str(a) for a in failed_accounts)}")
    
    return success_count, downloaded_count, failed_accounts

def process_posts(L, profile, account_name, posts, downloaded_count, success_count):
    """Process posts for an account"""
    # Process posts
    posts_processed = 0
    
    # Create account directory
    account_dir = os.path.join(DOWNLOAD_DIR, account_name)
    os.makedirs(account_dir, exist_ok=True)
    
    # Create a directory for metadata
    metadata_dir = os.path.join(account_dir, 'metadata')
    os.makedirs(metadata_dir, exist_ok=True)
    
    try:
        for post in posts:
            try:
                # Check if we've reached the maximum downloads limit
                if downloaded_count >= MAX_DOWNLOADS_PER_RUN:
                    logger.info(f"Reached maximum download limit of {MAX_DOWNLOADS_PER_RUN}")
                    break
                
                # Skip if it's not a video
                if not post.is_video:
                    logger.debug(f"Skipping non-video post from {account_name}: {post.shortcode}")
                    continue
                
                # Check if this video has already been downloaded
                video_filename = f"{post.date_utc.strftime('%Y-%m-%d_%H-%M-%S')}_{post.shortcode}.mp4"
                video_path = os.path.join(account_dir, video_filename)
                
                if os.path.exists(video_path):
                    logger.debug(f"Skipping already downloaded video: {video_filename}")
                    continue
                
                # Download the post
                logger.info(f"Downloading video post from {account_name}: {post.shortcode}")
                
                try:
                    # Download only the video
                    L.download_post(post, target=account_dir)
                    downloaded_count += 1
                    success_count += 1
                    
                    # Save post metadata to a separate JSON file
                    metadata = {
                        'shortcode': post.shortcode,
                        'date_utc': post.date_utc.strftime('%Y-%m-%d %H:%M:%S'),
                        'caption': post.caption if post.caption else '',
                        'likes': post.likes,
                        'comments': post.comments,
                        'url': f"https://www.instagram.com/p/{post.shortcode}/",
                        'account': account_name
                    }
                    
                    metadata_path = os.path.join(metadata_dir, f"{post.shortcode}.json")
                    with open(metadata_path, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, ensure_ascii=False, indent=4)
                    
                    # Respect Instagram's rate limits by adding a delay between downloads
                    time.sleep(DOWNLOAD_DELAY)
                    posts_processed += 1
                    
                except Exception as e:
                    logger.error(f"Error downloading post {post.shortcode} from {account_name}: {str(e)}")
                    # Continue with the next post despite this error
                    continue
            
            except Exception as post_error:
                logger.error(f"Error processing post from {account_name}: {str(post_error)}")
                # Continue with the next post
                continue
        
        logger.info(f"Processed {posts_processed} posts from {account_name}, downloaded {success_count} videos")
        return True, posts_processed
        
    except Exception as e:
        logger.error(f"Error processing posts for {account_name}: {str(e)}")
        return False, 0

def get_posts_alternative(profile, username):
    """Alternative approach to fetch posts when the standard iterator fails"""
    try:
        # Try direct URL construction approach
        base_url = f"https://www.instagram.com/{username}/"
        logger.info(f"Attempting alternative post fetching from {base_url}")
        
        # Return posts that we already have in the directory
        metadata_dir = os.path.join(DOWNLOAD_DIR, username, "metadata")
        existing_posts = []
        
        if os.path.exists(metadata_dir):
            for file in os.listdir(metadata_dir):
                if file.endswith('.json'):
                    try:
                        with open(os.path.join(metadata_dir, file), 'r') as f:
                            metadata = json.load(f)
                            if 'shortcode' in metadata:
                                existing_posts.append(metadata['shortcode'])
                    except Exception as e:
                        logger.error(f"Error reading metadata file {file}: {str(e)}")
        
        logger.info(f"Found {len(existing_posts)} existing posts metadata to process")
        return existing_posts
        
    except Exception as e:
        logger.error(f"Alternative post fetching failed: {str(e)}")
        return []

def get_random_user_agent():
    """Return a random user agent to appear more human-like"""
    user_agents = [
        # Desktop browsers
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0",
        # Mobile browsers
        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
        "Mozilla/5.0 (Linux; Android 13; SM-S908B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.6099.43 Mobile Safari/537.36"
    ]
    return random.choice(user_agents)

def schedule_refresh(username, backoff_minutes=60):
    """Schedule a refresh attempt with exponential backoff"""
    state_file = os.path.join(DATA_DIR, "refresh_state.json")
    
    # Load or initialize state
    if os.path.exists(state_file):
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
        except json.JSONDecodeError:
            state = {"accounts": {}}
    else:
        state = {"accounts": {}}
    
    # Get account state or initialize
    account_state = state["accounts"].get(username, {
        "last_attempt": None,
        "backoff_minutes": backoff_minutes,
        "consecutive_failures": 0
    })
    
    # Update for next attempt
    now = datetime.now().isoformat()
    account_state["last_attempt"] = now
    
    if account_state["consecutive_failures"] > 0:
        # Exponential backoff
        account_state["backoff_minutes"] *= 2
    
    # Schedule next attempt
    next_attempt = (datetime.now() + 
                   timedelta(minutes=account_state["backoff_minutes"]))
    account_state["next_attempt"] = next_attempt.isoformat()
    
    # Save state
    state["accounts"][username] = account_state
    with open(state_file, 'w') as f:
        json.dump(state, f)
    
    logger.info(f"Scheduled next refresh for {username} at {next_attempt.isoformat()}")
    return next_attempt.isoformat()

def should_refresh_account(username):
    """Check if an account is due for refresh based on backoff schedule"""
    state_file = os.path.join(DATA_DIR, "refresh_state.json")
    
    # If no state file, always refresh
    if not os.path.exists(state_file):
        return True
        
    try:
        with open(state_file, 'r') as f:
            state = json.load(f)
            
        # If account not in state, always refresh
        if username not in state.get("accounts", {}):
            return True
            
        account_state = state["accounts"][username]
        next_attempt_str = account_state.get("next_attempt")
        
        # If no next attempt scheduled, always refresh
        if not next_attempt_str:
            return True
            
        # Parse next attempt time
        next_attempt = datetime.fromisoformat(next_attempt_str)
        
        # Check if we're past the scheduled time
        return datetime.now() >= next_attempt
        
    except Exception as e:
        logger.error(f"Error checking refresh schedule: {str(e)}")
        return True  # Default to allowing refresh on error

def is_account_due_for_refresh(account_name):
    """Check if an account is due for refresh based on its last processing time"""
    # Path to the account state file
    account_state_file = os.path.join(DATA_DIR, "logs", "account_states.json")
    
    # Default: account is due for refresh
    if not os.path.exists(account_state_file):
        os.makedirs(os.path.dirname(account_state_file), exist_ok=True)
        return True
    
    try:
        # Load account states
        with open(account_state_file, 'r') as f:
            account_states = json.load(f)
        
        # Check if account exists in states
        if account_name not in account_states:
            return True
        
        account_state = account_states[account_name]
        last_processed = account_state.get('last_processed')
        
        # If no last processed time, account is due for refresh
        if not last_processed:
            return True
        
        # Check if account is in cooldown
        cooldown_until = account_state.get('cooldown_until')
        if cooldown_until:
            cooldown_time = datetime.fromisoformat(cooldown_until)
            if datetime.now() < cooldown_time:
                logger.info(f"Account {account_name} is in cooldown until {cooldown_until}")
                return False
        
        # Check if enough time has passed since last processing
        last_processed_time = datetime.fromisoformat(last_processed)
        refresh_interval = account_state.get('refresh_interval', 24)  # Default: 24 hours
        
        next_refresh_time = last_processed_time + timedelta(hours=refresh_interval)
        
        if datetime.now() < next_refresh_time:
            logger.info(f"Account {account_name} not due for refresh until {next_refresh_time.isoformat()}")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Error checking refresh status for {account_name}: {str(e)}")
        # Default to allowing refresh on error
        return True

def mark_account_processed(account_name, success=True):
    """Mark an account as processed and update its refresh schedule"""
    # Path to the account state file
    account_state_file = os.path.join(DATA_DIR, "logs", "account_states.json")
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(account_state_file), exist_ok=True)
        
        # Load existing account states or create new dict
        if os.path.exists(account_state_file):
            with open(account_state_file, 'r') as f:
                account_states = json.load(f)
        else:
            account_states = {}
        
        # Get or create account state
        if account_name not in account_states:
            account_states[account_name] = {}
        
        # Update account state
        account_states[account_name]['last_processed'] = datetime.now().isoformat()
        
        # Update success/failure count
        if success:
            account_states[account_name]['consecutive_failures'] = 0
            # Reset refresh interval to default on success
            account_states[account_name]['refresh_interval'] = 24  # Default: 24 hours
        else:
            # Increment failure count
            failures = account_states[account_name].get('consecutive_failures', 0) + 1
            account_states[account_name]['consecutive_failures'] = failures
            
            # Implement exponential backoff for failures
            backoff_hours = min(24 * (2 ** (failures - 1)), 168)  # Max 1 week
            account_states[account_name]['refresh_interval'] = backoff_hours
            
            # Set cooldown period for repeated failures
            if failures > 2:
                cooldown_minutes = ACCOUNT_COOLDOWN_MINUTES * (2 ** (failures - 3))  # Exponential cooldown
                cooldown_until = (datetime.now() + timedelta(minutes=cooldown_minutes)).isoformat()
                account_states[account_name]['cooldown_until'] = cooldown_until
                logger.warning(f"Account {account_name} in cooldown until {cooldown_until} after {failures} failures")
        
        # Save updated account states
        with open(account_state_file, 'w') as f:
            json.dump(account_states, f, indent=4)
            
    except Exception as e:
        logger.error(f"Error marking account {account_name} as processed: {str(e)}")

def mark_account_cooldown(username, cooldown_minutes=ACCOUNT_COOLDOWN_MINUTES):
    """Mark an account for cooldown after a failure"""
    # Path to the account state file
    account_state_file = os.path.join(DATA_DIR, "logs", "account_states.json")
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(account_state_file), exist_ok=True)
        
        # Load existing account states or create new dict
        if os.path.exists(account_state_file):
            with open(account_state_file, 'r') as f:
                account_states = json.load(f)
        else:
            account_states = {}
        
        # Get or create account state
        if username not in account_states:
            account_states[username] = {}
        
        # Set cooldown period
        cooldown_until = (datetime.now() + timedelta(minutes=cooldown_minutes)).isoformat()
        account_states[username]['cooldown_until'] = cooldown_until
        
        # Increment failure count
        failures = account_states[username].get('consecutive_failures', 0) + 1
        account_states[username]['consecutive_failures'] = failures
        
        logger.warning(f"Account {username} in cooldown until {cooldown_until}")
        
        # Save updated account states
        with open(account_state_file, 'w') as f:
            json.dump(account_states, f, indent=4)
            
    except Exception as e:
        logger.error(f"Error marking account {username} for cooldown: {str(e)}")

if __name__ == "__main__":
    download_from_instagram() 