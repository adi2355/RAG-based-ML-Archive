"""
Database setup script for the Knowledge Base
Creates the database schema with support for multiple content sources
"""
import os
import sqlite3
import logging
from config import DB_PATH, DATA_DIR

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('db_setup')

def setup_database():
    """Set up or migrate the database"""
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    # Connect to SQLite database (it will be created if it doesn't exist)
    logger.info(f"Setting up database at {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check if we need to run migrations (if certain tables already exist)
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='videos'")
    videos_table_exists = cursor.fetchone() is not None
    
    if videos_table_exists:
        logger.info("Existing schema detected, running migrations")
        migrate_schema(conn)
    else:
        logger.info("No existing schema detected, creating new schema")
        create_schema(conn)
    
    # Commit changes and close the connection
    conn.commit()
    conn.close()
    logger.info("Database setup complete")

def create_schema(conn):
    """Create the complete database schema"""
    cursor = conn.cursor()
    
    # Create content source types table
    cursor.execute("""
    CREATE TABLE source_types (
        id INTEGER PRIMARY KEY,
        name TEXT UNIQUE,
        description TEXT
    )
    """)
    
    # Insert initial source types
    source_types = [
        ('instagram', 'Instagram video content'),
        ('github', 'GitHub repository content'),
        ('paper', 'Research paper content')
    ]
    cursor.executemany("INSERT INTO source_types (name, description) VALUES (?, ?)", 
                      source_types)
    
    # Create videos table (for backward compatibility)
    cursor.execute("""
    CREATE TABLE videos (
        id INTEGER PRIMARY KEY,
        filepath TEXT,
        account TEXT,
        shortcode TEXT UNIQUE,
        upload_date TEXT,
        caption TEXT,
        url TEXT,
        processed INTEGER DEFAULT 0,
        transcribed INTEGER DEFAULT 0,
        summarized INTEGER DEFAULT 0,
        duration_seconds REAL,
        hashtags TEXT,
        transcript TEXT,
        summary TEXT,
        embedding TEXT,
        key_phrases TEXT,
        technical_level TEXT,
        research_citations TEXT,
        ai_concepts TEXT
    )
    """)
    
    # Create hashtags table
    cursor.execute("""
    CREATE TABLE hashtags (
        id INTEGER PRIMARY KEY,
        name TEXT UNIQUE
    )
    """)
    
    # Create video_hashtags junction table
    cursor.execute("""
    CREATE TABLE video_hashtags (
        video_id INTEGER,
        hashtag_id INTEGER,
        PRIMARY KEY (video_id, hashtag_id),
        FOREIGN KEY (video_id) REFERENCES videos (id),
        FOREIGN KEY (hashtag_id) REFERENCES hashtags (id)
    )
    """)
    
    # Create unified AI content table
    cursor.execute("""
    CREATE TABLE ai_content (
        id INTEGER PRIMARY KEY,
        source_type_id INTEGER,
        source_id TEXT,
        title TEXT,
        content TEXT,
        content_details TEXT,
        ai_concepts TEXT,
        technical_level TEXT,
        url TEXT,
        timestamp TEXT,
        citation_count INTEGER,
        star_count INTEGER,
        likes INTEGER,
        quality_score FLOAT,
        embedding TEXT,
        last_updated TEXT,
        FOREIGN KEY (source_type_id) REFERENCES source_types(id)
    )
    """)
    
    # Create index for efficient lookups
    cursor.execute("CREATE INDEX idx_ai_content_source ON ai_content(source_type_id, source_id)")
    
    # Create GitHub repositories table
    cursor.execute("""
    CREATE TABLE github_repos (
        id INTEGER PRIMARY KEY,
        name TEXT UNIQUE,
        full_name TEXT,
        description TEXT,
        url TEXT,
        stars INTEGER,
        watchers INTEGER,
        forks INTEGER,
        language TEXT,
        last_push TEXT,
        created_at TEXT,
        updated_at TEXT,
        topics TEXT,
        readme TEXT,
        last_crawled TEXT
    )
    """)
    
    # Create research papers table
    cursor.execute("""
    CREATE TABLE research_papers (
        id INTEGER PRIMARY KEY,
        arxiv_id TEXT UNIQUE,
        title TEXT,
        abstract TEXT,
        authors TEXT,
        published TEXT,
        updated TEXT,
        categories TEXT,
        url TEXT,
        pdf_url TEXT,
        sections TEXT,
        last_crawled TEXT
    )
    """)
    
    # Create virtual FTS table for searching across all content
    cursor.execute("""
    CREATE VIRTUAL TABLE ai_content_fts USING fts4(
        title,
        content,
        ai_concepts,
        content='ai_content',
        tokenize=porter
    )
    """)
    
    logger.info("Schema creation complete")

def migrate_schema(conn):
    """Migrate existing schema to new structure"""
    cursor = conn.cursor()
    
    # Check which tables need to be created
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='source_types'")
    source_types_exists = cursor.fetchone() is not None
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ai_content'")
    ai_content_exists = cursor.fetchone() is not None
    
    # Create source_types table if it doesn't exist
    if not source_types_exists:
        cursor.execute("""
        CREATE TABLE source_types (
            id INTEGER PRIMARY KEY,
            name TEXT UNIQUE,
            description TEXT
        )
        """)
        
        # Insert initial source types
        source_types = [
            ('instagram', 'Instagram video content'),
            ('github', 'GitHub repository content'),
            ('paper', 'Research paper content')
        ]
        cursor.executemany("INSERT INTO source_types (name, description) VALUES (?, ?)", 
                          source_types)
        logger.info("Created source_types table")
    
    # Create ai_content table if it doesn't exist
    if not ai_content_exists:
        cursor.execute("""
        CREATE TABLE ai_content (
            id INTEGER PRIMARY KEY,
            source_type_id INTEGER,
            source_id TEXT,
            title TEXT,
            content TEXT,
            content_details TEXT,
            ai_concepts TEXT,
            technical_level TEXT,
            url TEXT,
            timestamp TEXT,
            citation_count INTEGER,
            star_count INTEGER,
            likes INTEGER,
            quality_score FLOAT,
            embedding TEXT,
            last_updated TEXT,
            FOREIGN KEY (source_type_id) REFERENCES source_types(id)
        )
        """)
        cursor.execute("CREATE INDEX idx_ai_content_source ON ai_content(source_type_id, source_id)")
        logger.info("Created ai_content table")
    
    # Check for github_repos table
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='github_repos'")
    if cursor.fetchone() is None:
        cursor.execute("""
        CREATE TABLE github_repos (
            id INTEGER PRIMARY KEY,
            name TEXT UNIQUE,
            full_name TEXT,
            description TEXT,
            url TEXT,
            stars INTEGER,
            watchers INTEGER,
            forks INTEGER,
            language TEXT,
            last_push TEXT,
            created_at TEXT,
            updated_at TEXT,
            topics TEXT,
            readme TEXT,
            last_crawled TEXT
        )
        """)
        logger.info("Created github_repos table")
    
    # Check for research_papers table
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='research_papers'")
    if cursor.fetchone() is None:
        cursor.execute("""
        CREATE TABLE research_papers (
            id INTEGER PRIMARY KEY,
            arxiv_id TEXT UNIQUE,
            title TEXT,
            abstract TEXT,
            authors TEXT,
            published TEXT,
            updated TEXT,
            categories TEXT,
            url TEXT,
            pdf_url TEXT,
            sections TEXT,
            last_crawled TEXT
        )
        """)
        logger.info("Created research_papers table")
    
    # Create FTS table if it doesn't exist
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ai_content_fts'")
    if cursor.fetchone() is None:
        cursor.execute("""
        CREATE VIRTUAL TABLE ai_content_fts USING fts4(
            title,
            content,
            ai_concepts,
            content='ai_content',
            tokenize=porter
        )
        """)
        logger.info("Created ai_content_fts table")
    
    # Check if we need to add columns to the videos table
    try:
        cursor.execute("SELECT ai_concepts FROM videos LIMIT 1")
    except sqlite3.OperationalError:
        # ai_concepts column doesn't exist, add it and other new columns
        for column in ['ai_concepts', 'technical_level', 'research_citations', 'key_phrases', 'duration_seconds']:
            try:
                cursor.execute(f"ALTER TABLE videos ADD COLUMN {column} TEXT")
                logger.info(f"Added {column} column to videos table")
            except sqlite3.OperationalError:
                logger.info(f"Column {column} already exists in videos table")
    
    logger.info("Schema migration complete")

def migrate_instagram_data(conn):
    """Migrate existing Instagram data to the unified AI content table"""
    cursor = conn.cursor()
    
    # Check if data migration is needed
    cursor.execute("SELECT COUNT(*) FROM ai_content WHERE source_type_id = 1")  # Instagram source type
    if cursor.fetchone()[0] > 0:
        logger.info("Instagram data already migrated to ai_content table")
        return
    
    # Check if we have any videos data to migrate
    cursor.execute("SELECT COUNT(*) FROM videos")
    if cursor.fetchone()[0] == 0:
        logger.info("No videos data to migrate")
        return
    
    # Get Instagram source type ID
    cursor.execute("SELECT id FROM source_types WHERE name = 'instagram'")
    instagram_id = cursor.fetchone()[0]
    
    # Migrate data from videos table to ai_content
    cursor.execute("""
    INSERT INTO ai_content (
        source_type_id, source_id, title, content, url, timestamp, 
        likes, ai_concepts, technical_level, last_updated
    )
    SELECT 
        ?, shortcode, caption, transcript, url, upload_date,
        0, ai_concepts, technical_level, datetime('now')
    FROM videos
    WHERE transcript IS NOT NULL
    """, (instagram_id,))
    
    migrated_count = cursor.rowcount
    logger.info(f"Migrated {migrated_count} Instagram videos to the unified ai_content table")

if __name__ == "__main__":
    setup_database() 