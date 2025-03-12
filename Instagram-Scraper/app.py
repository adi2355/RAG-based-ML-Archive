"""
Web interface for the Instagram Knowledge Base
"""
import os
import re
import sqlite3
from functools import lru_cache
from flask import Flask, render_template, request, jsonify, g, send_from_directory, redirect, url_for

from config import (
    DB_PATH,
    WEB_PORT,
    DEBUG_MODE,
    DOWNLOAD_DIR,
    DATA_DIR
)

app = Flask(__name__)

def get_db():
    """Get database connection with row factory for easy access"""
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DB_PATH)
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_connection(exception):
    """Close database connection when app context ends"""
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

@app.route('/')
def index():
    """Home page with search interface"""
    return render_template('index.html')

@app.route('/search')
def search():
    """Search endpoint"""
    query = request.args.get('query', '')
    account_filter = request.args.get('account', '')
    tag_filter = request.args.get('tag', '')
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 10))
    
    # Calculate offset
    offset = (page - 1) * per_page
    
    db = get_db()
    cursor = db.cursor()
    
    # Get available accounts for filtering
    cursor.execute("SELECT DISTINCT account FROM videos ORDER BY account")
    accounts = [row['account'] for row in cursor.fetchall()]
    
    # Get popular tags for filtering
    cursor.execute('''
    SELECT tag, COUNT(*) as count 
    FROM tags 
    GROUP BY tag 
    ORDER BY count DESC 
    LIMIT 20
    ''')
    tags = [{'tag': row['tag'], 'count': row['count']} for row in cursor.fetchall()]
    
    results = []
    total_results = 0
    
    if query or account_filter or tag_filter:
        # Base query
        sql = '''
        SELECT v.id, v.shortcode, v.account, v.caption, v.transcript, 
               v.timestamp, v.url, v.likes, v.comments
        FROM videos v
        '''
        
        params = []
        where_clauses = []
        
        # Add tag filter if specified
        if tag_filter:
            sql += "JOIN tags t ON v.id = t.video_id "
            where_clauses.append("t.tag = ?")
            params.append(tag_filter)
        
        # Add account filter if specified
        if account_filter:
            where_clauses.append("v.account = ?")
            params.append(account_filter)
        
        # Add search query if specified
        if query:
            sql += "JOIN videos_fts fts ON v.id = fts.docid "  # Changed from rowid to docid for FTS4
            where_clauses.append("videos_fts MATCH ?")
            params.append(query)
            
            # FTS4 doesn't support the snippet function, so we'll remove this line
            # and handle highlighting in application code later
        
        # Combine where clauses
        if where_clauses:
            sql += "WHERE " + " AND ".join(where_clauses)
        
        # Get total results count
        count_sql = f"SELECT COUNT(*) as count FROM ({sql})"
        cursor.execute(count_sql, params)
        total_results = cursor.fetchone()['count']
        
        # Add order and limit
        if query:
            # FTS4 doesn't have built-in rank function
            sql += " ORDER BY v.timestamp DESC"
        else:
            sql += " ORDER BY v.timestamp DESC"
            
        sql += " LIMIT ? OFFSET ?"
        params.extend([per_page, offset])
        
        # Execute query
        cursor.execute(sql, params)
        results = cursor.fetchall()
    
    # Format results for template
    formatted_results = []
    for row in results:
        # Get tags for this video
        cursor.execute("SELECT tag FROM tags WHERE video_id = ?", (row['id'],))
        video_tags = [tag['tag'] for tag in cursor.fetchall()]
        
        # Format the date
        timestamp = row['timestamp'].split('T')[0] if row['timestamp'] else 'Unknown'
        
        # Get video path
        video_path = f"/video/{row['account']}/{row['shortcode']}"
        
        # Create our own snippet since snippet() isn't available in FTS4
        transcript = row['transcript'] or ""
        
        # Default snippet (first ~200 chars of transcript)
        if len(transcript) > 200:
            snippet = transcript[:200] + '...'
        else:
            snippet = transcript
        
        # If we have a search query, try to find a relevant part of the transcript
        if query and transcript:
            query_words = query.lower().split()
            transcript_lower = transcript.lower()
            
            # Find the first occurrence of any query word
            position = -1
            for word in query_words:
                pos = transcript_lower.find(word)
                if pos != -1:
                    position = pos
                    break
            
            # Extract a section around the match if found
            if position != -1:
                # Get 50 chars before and 150 after the match position
                start = max(0, position - 50)
                end = min(len(transcript), position + 150)
                
                # Get the relevant section
                context = transcript[start:end]
                
                # Add ellipsis if needed
                if start > 0:
                    context = '...' + context
                if end < len(transcript):
                    context += '...'
                
                snippet = context
        
        # Simple highlighting for matched terms
        if query and transcript:
            for word in query.lower().split():
                # Case-insensitive replace with HTML highlighting
                word_pattern = word.lower()
                start = 0
                while True:
                    start_pos = snippet.lower().find(word_pattern, start)
                    if start_pos == -1:
                        break
                    
                    end_pos = start_pos + len(word)
                    original_word = snippet[start_pos:end_pos]
                    snippet = snippet[:start_pos] + f'<mark>{original_word}</mark>' + snippet[end_pos:]
                    
                    # Move past this match
                    start = start_pos + len(f'<mark>{original_word}</mark>')
        
        formatted_results.append({
            'id': row['id'],
            'shortcode': row['shortcode'],
            'account': row['account'],
            'caption': row['caption'],
            'snippet': snippet,
            'timestamp': timestamp,
            'url': row['url'],
            'likes': row['likes'],
            'comments': row['comments'],
            'tags': video_tags,
            'video_path': video_path
        })
    
    # Calculate pagination info
    total_pages = (total_results + per_page - 1) // per_page if total_results > 0 else 1
    has_prev = page > 1
    has_next = page < total_pages
    
    return render_template(
        'search.html',
        query=query,
        account_filter=account_filter,
        tag_filter=tag_filter,
        results=formatted_results,
        accounts=accounts,
        tags=tags,
        total_results=total_results,
        page=page,
        per_page=per_page,
        total_pages=total_pages,
        has_prev=has_prev,
        has_next=has_next
    )

@app.route('/video/<account>/<shortcode>')
def video(account, shortcode):
    """Video detail page"""
    db = get_db()
    cursor = db.cursor()
    
    # Get video details
    cursor.execute('''
    SELECT v.* FROM videos v
    WHERE v.account = ? AND v.shortcode = ?
    ''', (account, shortcode))
    video = cursor.fetchone()
    
    if not video:
        return "Video not found", 404
    
    # Get tags
    cursor.execute("SELECT tag FROM tags WHERE video_id = ?", (video['id'],))
    tags = [tag['tag'] for tag in cursor.fetchall()]
    
    # Find video file
    video_filename = None
    account_dir = os.path.join(DOWNLOAD_DIR, account)
    if os.path.exists(account_dir):
        for filename in os.listdir(account_dir):
            if shortcode in filename and filename.endswith('.mp4'):
                video_filename = f"/media/{account}/{filename}"
                break
    
    return render_template(
        'video.html',
        video=video,
        tags=tags,
        video_filename=video_filename
    )

@app.route('/api/search')
def api_search():
    """API endpoint for search (for AJAX requests)"""
    query = request.args.get('query', '')
    account = request.args.get('account', '')
    tag = request.args.get('tag', '')
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 10))
    
    db = get_db()
    cursor = db.cursor()
    
    # Start building the query
    sql_select = """
        SELECT v.id, v.shortcode, v.account, v.caption, v.transcript, v.summary,
               v.timestamp, v.url, v.likes, v.comments, v.word_count, v.duration_seconds
    """
    
    sql_from = " FROM videos v"
    sql_where = ""
    sql_order = ""
    params = []
    
    # Add search condition if query provided
    if query:
        sql_from += " JOIN videos_fts fts ON v.id = fts.docid"
        sql_where = " WHERE videos_fts MATCH ?"
        params.append(query)
        
        # Custom snippets for transcript and summary
        sql_select += """,
            (SELECT substr(v.transcript, 
                max(0, instr(lower(v.transcript), lower(?)) - 50), 
                150)) AS transcript_snippet,
            (SELECT substr(v.summary, 
                max(0, instr(lower(v.summary), lower(?)) - 25), 
                100)) AS summary_snippet
        """
        params.extend([query, query])  # Add query params for snippets
        
        # Enhanced ordering that prioritizes matches in summary, then caption
        sql_order = """
        ORDER BY
            CASE 
                WHEN fts.summary MATCH ? THEN 1
                WHEN fts.caption MATCH ? THEN 2
                WHEN fts.transcript MATCH ? THEN 3
                ELSE 4
            END
        """
        params.extend([query, query, query])
    
    # Add account filter if specified
    if account:
        if sql_where:
            sql_where += " AND v.account = ?"
        else:
            sql_where = " WHERE v.account = ?"
        params.append(account)
    
    # Add tag filter if specified
    if tag:
        sql_from += " JOIN tags t ON v.id = t.video_id"
        if sql_where:
            sql_where += " AND t.tag = ?"
        else:
            sql_where = " WHERE t.tag = ?"
        params.append(tag)
    
    # Count total results for pagination
    count_sql = f"SELECT COUNT(*) as total_count {sql_from}{sql_where}"
    cursor.execute(count_sql, params)
    total_count = cursor.fetchone()['total_count']
    
    # Calculate total pages
    total_pages = (total_count + per_page - 1) // per_page
    
    # Add default sorting if no query or specific ordering
    if not sql_order:
        sql_order = " ORDER BY v.timestamp DESC"
    
    # Add pagination
    sql_limit = " LIMIT ? OFFSET ?"
    
    # Complete SQL query
    full_sql = f"{sql_select}{sql_from}{sql_where}{sql_order}{sql_limit}"
    
    # Execute with pagination parameters
    cursor.execute(full_sql, params + [per_page, (page - 1) * per_page])
    results = cursor.fetchall()
    
    # Convert to list of dicts with highlighted snippets
    result_list = []
    for row in results:
        result_dict = {key: row[key] for key in row.keys()}
        
        # Add highlighted snippets if search was performed
        if query:
            # Highlight the query term in snippets
            if 'transcript_snippet' in result_dict and result_dict['transcript_snippet']:
                result_dict['transcript_snippet'] = highlight_term(
                    result_dict['transcript_snippet'], query
                )
            
            if 'summary_snippet' in result_dict and result_dict['summary_snippet']:
                result_dict['summary_snippet'] = highlight_term(
                    result_dict['summary_snippet'], query
                )
        
        result_list.append(result_dict)
    
    return jsonify({
        'results': result_list,
        'total': total_count,
        'page': page,
        'per_page': per_page,
        'total_pages': total_pages,
        'query': query,
        'account': account,
        'tag': tag
    })

def highlight_term(text, term):
    """Highlight search term in text using HTML"""
    if not text or not term:
        return text
    
    # Simple case-insensitive replace
    # For more complex highlighting, consider using regex
    term_lower = term.lower()
    text_lower = text.lower()
    
    result = ""
    last_pos = 0
    
    # Find all occurrences of the term
    pos = text_lower.find(term_lower)
    while pos != -1:
        # Add text before the term
        result += text[last_pos:pos]
        # Add the highlighted term
        result += f"<mark>{text[pos:pos+len(term)]}</mark>"
        # Move past this occurrence
        last_pos = pos + len(term)
        # Find next occurrence
        pos = text_lower.find(term_lower, last_pos)
    
    # Add any remaining text
    result += text[last_pos:]
    
    return result

@app.route('/media/<path:path>')
def media(path):
    """Serve media files"""
    return send_from_directory(DOWNLOAD_DIR, path)

# Routes for static templates
@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.route('/stats')
def stats():
    """Statistics page"""
    db = get_db()
    cursor = db.cursor()
    
    # Get general stats
    cursor.execute("SELECT COUNT(*) as total FROM videos")
    total_videos = cursor.fetchone()['total']
    
    # Get account stats
    cursor.execute('''
    SELECT account, COUNT(*) as count 
    FROM videos 
    GROUP BY account 
    ORDER BY count DESC
    ''')
    accounts = cursor.fetchall()
    
    # Get tag stats
    cursor.execute('''
    SELECT tag, COUNT(*) as count 
    FROM tags 
    GROUP BY tag 
    ORDER BY count DESC 
    LIMIT 50
    ''')
    tags = cursor.fetchall()
    
    # Get timeline stats
    cursor.execute('''
    SELECT substr(timestamp, 1, 7) as month, COUNT(*) as count 
    FROM videos 
    WHERE timestamp IS NOT NULL
    GROUP BY month 
    ORDER BY month
    ''')
    timeline = cursor.fetchall()
    
    return render_template(
        'stats.html',
        total_videos=total_videos,
        accounts=accounts,
        tags=tags,
        timeline=timeline
    )

# Caching utilities
@lru_cache(maxsize=100)
def get_video_by_shortcode(shortcode):
    """Get video details by shortcode with caching"""
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT * FROM videos WHERE shortcode = ?", [shortcode])
    return cursor.fetchone()

@lru_cache(maxsize=30)
def get_recent_videos(limit=10, account=None):
    """Get recent videos with caching"""
    db = get_db()
    cursor = db.cursor()
    
    if account:
        cursor.execute(
            "SELECT * FROM videos WHERE account = ? ORDER BY timestamp DESC LIMIT ?", 
            [account, limit]
        )
    else:
        cursor.execute(
            "SELECT * FROM videos ORDER BY timestamp DESC LIMIT ?", 
            [limit]
        )
    
    return cursor.fetchall()

@lru_cache(maxsize=20)
def get_video_statistics():
    """Get video statistics with caching"""
    db = get_db()
    cursor = db.cursor()
    
    # Get total videos
    cursor.execute("SELECT COUNT(*) as video_count FROM videos")
    total_videos = cursor.fetchone()['video_count']
    
    # Get videos per account
    cursor.execute(
        "SELECT account, COUNT(*) as count FROM videos GROUP BY account ORDER BY count DESC"
    )
    accounts = cursor.fetchall()
    
    # Get total duration (if available)
    cursor.execute(
        "SELECT SUM(duration_seconds) as total_duration FROM videos WHERE duration_seconds IS NOT NULL"
    )
    total_duration = cursor.fetchone()['total_duration'] or 0
    
    # Get total word count
    cursor.execute(
        "SELECT SUM(word_count) as total_words FROM videos WHERE word_count IS NOT NULL"
    )
    total_words = cursor.fetchone()['total_words'] or 0
    
    return {
        'total_videos': total_videos,
        'accounts': accounts,
        'total_duration_seconds': total_duration,
        'total_words': total_words
    }

# Clear caches when data changes
def clear_caches():
    """Clear all LRU caches"""
    get_video_by_shortcode.cache_clear()
    get_recent_videos.cache_clear()
    get_video_statistics.cache_clear()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=WEB_PORT, debug=DEBUG_MODE) 