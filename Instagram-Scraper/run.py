"""
Main script to run the complete Instagram Knowledge Base system
"""
import os
import argparse
import logging
from time import time
import sqlite3

# Import our modules
from config import DATA_DIR
import config
import downloader
import transcriber
import indexer
import summarizer
from app import app

# Import the new modules
try:
    import db_migration
    import github_collector
    import arxiv_collector
    import concept_extractor
    import chunking
    import embeddings
    import generate_embeddings
    import vector_search
    import hybrid_search
    import context_builder
    import llm_integration
    has_additional_modules = True
    has_vector_search = True
    has_rag = True
except ImportError as e:
    has_additional_modules = False
    has_vector_search = False
    has_rag = False
    import_error = str(e)
    missing_module = str(e).split("No module named ")[-1].strip("'")

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
    os.makedirs('logs', exist_ok=True)

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

def run_db_migration():
    """Run database migration to support multiple content sources"""
    if not has_additional_modules:
        logger.error("Database migration module not available")
        return
    
    logger.info("Starting database migration")
    start_time = time()
    success = db_migration.migrate_database()
    
    if success:
        logger.info(f"Database migration completed in {time() - start_time:.2f} seconds")
    else:
        logger.error(f"Database migration failed after {time() - start_time:.2f} seconds")

def run_github_collector(max_repos=None):
    """Run GitHub repository collection"""
    if not has_additional_modules:
        logger.error("GitHub collector module not available")
        return
    
    logger.info("Starting GitHub repository collection")
    start_time = time()
    success_count = github_collector.collect_github_repos(max_repos=max_repos)
    logger.info(f"GitHub collection completed in {time() - start_time:.2f} seconds, processed {success_count} repositories")

def run_papers_collector(max_papers=None, force_update=False):
    """Run ArXiv research paper collection"""
    if not has_additional_modules:
        logger.error("ArXiv collector module not available")
        return
    
    logger.info("Starting ArXiv research paper collection")
    start_time = time()
    papers_added = arxiv_collector.collect_papers(max_papers=max_papers, force_update=force_update)
    logger.info(f"ArXiv collection completed in {time() - start_time:.2f} seconds, added {papers_added} new papers")

def run_concept_extractor(limit=None, source_type=None):
    """Run concept extraction on content"""
    if not has_additional_modules:
        logger.error("Concept extractor module not available")
        return
    
    logger.info("Starting concept extraction")
    start_time = time()
    
    if source_type:
        logger.info(f"Processing {limit or 'all'} items from source type: {source_type}")
        processed = concept_extractor.process_unprocessed_content(limit=limit or 5, source_type=source_type)
        logger.info(f"Processed {processed} items from {source_type}")
    else:
        # Process some content from each source type
        total_processed = 0
        for src_type in ["research_paper", "github", "instagram"]:
            logger.info(f"Processing source type: {src_type}")
            processed = concept_extractor.process_unprocessed_content(limit=limit or 3, source_type=src_type)
            logger.info(f"Processed {processed} items from {src_type}")
            total_processed += processed
        
        logger.info(f"Concept extraction completed in {time() - start_time:.2f} seconds, processed {total_processed} items")

def run_embedding_generation(source_type=None, limit=None, batch_size=50, 
                           chunk_size=500, chunk_overlap=100, force=False):
    """Generate embeddings for content"""
    if not has_vector_search:
        logger.error(f"Vector search modules not available: {import_error}")
        return
    
    logger.info("Starting embedding generation")
    start_time = time()
    
    args = []
    if source_type:
        args.extend(["--source-type", source_type])
    
    if limit:
        args.extend(["--limit", str(limit)])
    
    if batch_size:
        args.extend(["--batch-size", str(batch_size)])
    
    if chunk_size:
        args.extend(["--chunk-size", str(chunk_size)])
    
    if chunk_overlap:
        args.extend(["--chunk-overlap", str(chunk_overlap)])
    
    if force:
        args.append("--force")
    
    # Run embedding generator script
    import sys
    old_args = sys.argv
    sys.argv = ["generate_embeddings.py"] + args
    try:
        generate_embeddings.main()
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
    finally:
        sys.argv = old_args
    
    logger.info(f"Embedding generation completed in {time() - start_time:.2f} seconds")

def run_vector_search(query, top_k=5, source_type=None, in_memory_index=False):
    """Run vector search for a query"""
    if not has_vector_search:
        logger.error(f"Vector search modules not available: {import_error}")
        return []
    
    logger.info(f"Running vector search for: {query}")
    start_time = time()
    
    try:
        if in_memory_index:
            # Create in-memory index for faster search
            index = vector_search.create_memory_index()
            
            # Create embedding generator
            embedding_generator = embeddings.EmbeddingGenerator()
            
            # Generate query embedding
            query_embedding = embedding_generator.generate_embedding(query)
            
            # Search using in-memory index
            results = vector_search.search_memory_index(query_embedding, index, top_k=top_k)
            
            # Fetch chunk text and enrich results
            results = vector_search.enrich_search_results(results)
        else:
            # Use standard search
            results = vector_search.debug_search(query, top_k=top_k)
        
        logger.info(f"Vector search completed in {time() - start_time:.2f} seconds with {len(results)} results")
        
        # Print results
        print(f"\nVector search results for: {query}")
        print("=" * 80)
        
        for i, result in enumerate(results):
            print(f"\nResult {i+1} - Similarity: {result.get('similarity', 0):.4f}")
            print(f"Title: {result.get('title', 'Unknown')}")
            print(f"Source: {result.get('source_type', 'Unknown')}")
            print(f"Content ID: {result.get('content_id', 'Unknown')}")
            print("-" * 40)
            if 'chunk_text' in result:
                text = result['chunk_text']
                print(text[:300] + "..." if len(text) > 300 else text)
            print("-" * 40)
            
            if 'concepts' in result:
                print("Concepts:")
                for concept in result['concepts'][:5]:
                    print(f"- {concept['name']} ({concept['category']}, {concept['importance']})")
            print()
        
        return results
    
    except Exception as e:
        logger.error(f"Error performing vector search: {str(e)}")
        return []

def run_hybrid_search(query, top_k=5, source_type=None, vector_weight=None, 
                     keyword_weight=None, adaptive=True):
    """Run hybrid search for a query"""
    if not has_vector_search:
        logger.error(f"Vector search modules not available: {import_error}")
        return []
    
    logger.info(f"Running hybrid search for: {query}")
    start_time = time()
    
    try:
        # Load weight history if using adaptive weighting
        weight_history = None
        if adaptive and (vector_weight is None or keyword_weight is None):
            weight_history = hybrid_search.load_weights_history()
            
            # Get query type
            query_type = hybrid_search.classify_query_type(query)
            
            # Use query-specific weights if available
            normalized_query = query.lower().strip()
            if normalized_query in weight_history['queries']:
                vector_weight = weight_history['queries'][normalized_query]['vector_weight']
                keyword_weight = weight_history['queries'][normalized_query]['keyword_weight']
                logger.info(f"Using query-specific weights: vector={vector_weight:.2f}, keyword={keyword_weight:.2f}")
            else:
                # Use query type defaults
                type_key = f"default_{query_type}_vector_weight"
                if type_key in weight_history:
                    vector_weight = weight_history[type_key]
                    keyword_weight = 1.0 - vector_weight
                    logger.info(f"Using {query_type} query type weights: vector={vector_weight:.2f}, keyword={keyword_weight:.2f}")
        
        # Perform hybrid search
        results = hybrid_search.hybrid_search(
            query=query,
            top_k=top_k,
            source_type=source_type,
            vector_weight=vector_weight,
            keyword_weight=keyword_weight
        )
        
        logger.info(f"Hybrid search completed in {time() - start_time:.2f} seconds with {len(results)} results")
        
        # Get actual weights used for display
        if vector_weight is None or keyword_weight is None:
            vector_weight, keyword_weight = hybrid_search.determine_weights(query)
        
        # Print results
        print(f"\nHybrid search results for: {query}")
        print(f"Weights: vector={vector_weight:.2f}, keyword={keyword_weight:.2f}")
        print("=" * 80)
        
        for i, result in enumerate(results):
            print(f"\nResult {i+1} - Combined Score: {result.get('combined_score', 0):.4f}")
            print(f"Vector Score: {result.get('vector_score', 0):.4f}, Keyword Score: {result.get('keyword_score', 0):.4f}")
            print(f"Title: {result.get('title', 'Unknown')}")
            print(f"Source: {result.get('source_type', 'Unknown')}")
            print(f"Content ID: {result['content_id']}")
            print("-" * 40)
            
            if 'snippet' in result and result['snippet']:
                print(f"Keyword Match: {result['snippet']}")
            
            if 'chunk_text' in result and result['chunk_text']:
                text = result['chunk_text']
                print(text[:300] + "..." if len(text) > 300 else text)
            
            print("-" * 40)
            
            if 'concepts' in result:
                print("Concepts:")
                for concept in result['concepts'][:5]:
                    print(f"- {concept['name']} ({concept['category']}, {concept['importance']})")
            
            print()
        
        return results
    
    except Exception as e:
        logger.error(f"Error performing hybrid search: {str(e)}")
        return []

def run_rag_query(query, search_type='hybrid', source_type=None, top_k=5, 
                 vector_weight=None, keyword_weight=None, 
                 max_tokens_context=4000, max_tokens_answer=1000,
                 temperature=0.5, model=None, stream=False):
    """Run a RAG query and get a response from the LLM"""
    if not has_rag:
        logger.error(f"RAG modules not available: {import_error}")
        return
    
    logger.info(f"Running RAG query: {query}")
    start_time = time()
    
    try:
        # First check if we have any content in the database with embeddings
        conn = sqlite3.connect(config.DB_PATH)
        cursor = conn.cursor()
        
        # Check if content_embeddings table exists and has data
        cursor.execute("""
            SELECT COUNT(*) FROM sqlite_master 
            WHERE type='table' AND name='content_embeddings'
        """)
        table_exists = cursor.fetchone()[0] > 0
        
        if table_exists:
            cursor.execute("SELECT COUNT(*) FROM content_embeddings")
            embedding_count = cursor.fetchone()[0]
            
            if embedding_count == 0:
                logger.warning("No embeddings found in the database. Run --generate-embeddings first.")
                print("No content embeddings found in the database.")
                print("Please run the following command to generate embeddings first:")
                print("  python run.py --generate-embeddings")
                return
        else:
            logger.warning("Content embeddings table does not exist. Run --migrate and --generate-embeddings first.")
            print("Content embeddings table does not exist.")
            print("Please run the following commands to set up the database and generate embeddings:")
            print("  1. python run.py --migrate")
            print("  2. python run.py --generate-embeddings")
            return
        
        conn.close()
        
        # Create LLM provider
        llm_provider = llm_integration.ClaudeProvider(model=model or "claude-3-sonnet-20240229")
        
        # Create context builder
        ctx_builder = context_builder.ContextBuilder(max_tokens=max_tokens_context)
        
        # Create RAG assistant
        rag = llm_integration.RAGAssistant(
            llm_provider=llm_provider,
            context_builder=ctx_builder,
            max_tokens_answer=max_tokens_answer,
            max_tokens_context=max_tokens_context,
            temperature=temperature
        )
        
        # Answer query
        if stream:
            # Define callback for streaming
            def print_chunk(chunk):
                print(chunk, end="", flush=True)
                
            response = rag.answer_query_streaming(
                query=query,
                callback=print_chunk,
                search_type=search_type,
                vector_weight=vector_weight,
                keyword_weight=keyword_weight,
                source_type=source_type,
                top_k=top_k
            )
            print("\n")  # Add newline after streaming
        else:
            response = rag.answer_query(
                query=query,
                search_type=search_type,
                vector_weight=vector_weight,
                keyword_weight=keyword_weight,
                source_type=source_type,
                top_k=top_k
            )
            print(response["answer"])
            print("")
            
        # Print source information
        print("\nSources:")
        for idx, source in enumerate(response["sources"]):
            title = source.get("title", "Untitled")
            source_type = source.get("source_type", "Unknown")
            print(f"[{idx+1}] {title} ({source_type})")
            
        # Save response
        timestamp = response["metadata"]["timestamp"].split("T")[0]
        filepath = rag.save_response(response)
        print(f"\nResponse saved to: {filepath}")
        
        logger.info(f"RAG query completed in {time() - start_time:.2f} seconds")
        
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(f"Error running RAG query: {str(e)}\n{error_traceback}")
        print(f"Error: {str(e)}")
        print(f"Traceback:\n{error_traceback}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Instagram Knowledge Base')
    parser.add_argument('--download', action='store_true', help='Run the downloader module')
    parser.add_argument('--transcribe', action='store_true', help='Run the transcription module')
    parser.add_argument('--summarize', action='store_true', help='Run the summarization module')
    parser.add_argument('--index', action='store_true', help='Run the indexer module')
    parser.add_argument('--web', action='store_true', help='Run the web interface')
    parser.add_argument('--all', action='store_true', help='Run the complete pipeline')
    
    # Add new arguments for additional modules
    if has_additional_modules:
        parser.add_argument('--migrate', action='store_true', help='Run database migration')
        parser.add_argument('--github', action='store_true', help='Run GitHub repository collection')
        parser.add_argument('--github-max', type=int, help='Maximum number of GitHub repositories to collect')
        parser.add_argument('--papers', action='store_true', help='Run ArXiv research paper collection')
        parser.add_argument('--papers-max', type=int, help='Maximum number of papers to collect')
        parser.add_argument('--force-update', action='store_true', help='Force update of existing content')
        parser.add_argument('--concepts', action='store_true', help='Run AI concept extraction')
        parser.add_argument('--concepts-limit', type=int, help='Maximum number of items to process for concept extraction')
        parser.add_argument('--concepts-source', choices=['research_paper', 'github', 'instagram'], 
                            help='Only extract concepts from this source type')
    
    # Add vector search related arguments
    if has_vector_search:
        parser.add_argument('--generate-embeddings', action='store_true', help='Generate vector embeddings for content')
        parser.add_argument('--embeddings-source', choices=['research_paper', 'github', 'instagram'],
                            help='Generate embeddings only for this source type')
        parser.add_argument('--embeddings-limit', type=int, help='Maximum number of items to process for embedding generation')
        parser.add_argument('--embeddings-batch', type=int, default=50, help='Batch size for embedding generation')
        parser.add_argument('--embeddings-chunk', type=int, default=500, help='Chunk size for embedding generation')
        parser.add_argument('--embeddings-overlap', type=int, default=100, help='Chunk overlap for embedding generation')
        parser.add_argument('--embeddings-force', action='store_true', help='Force regeneration of existing embeddings')
        
        parser.add_argument('--vector-search', help='Run vector search with the provided query')
        parser.add_argument('--hybrid-search', help='Run hybrid search with the provided query')
        parser.add_argument('--search-top-k', type=int, default=5, help='Number of search results to return')
        parser.add_argument('--search-source', choices=['research_paper', 'github', 'instagram'],
                            help='Search only in this source type')
        parser.add_argument('--vector-weight', type=float, help='Weight for vector search (0-1)')
        parser.add_argument('--keyword-weight', type=float, help='Weight for keyword search (0-1)')
        parser.add_argument('--adaptive-weights', action='store_true', help='Use adaptive weights based on query type')
        parser.add_argument('--in-memory-index', action='store_true', help='Use in-memory index for vector search (faster)')
    
    # Add RAG arguments
    if has_rag:
        parser.add_argument('--rag-query', help='Run RAG query and get response from LLM')
        parser.add_argument('--rag-search-type', choices=['vector', 'hybrid'], default='hybrid',
                           help='Search type to use for RAG')
        parser.add_argument('--rag-max-tokens-context', type=int, default=4000,
                           help='Maximum tokens for RAG context')
        parser.add_argument('--rag-max-tokens-answer', type=int, default=1000,
                           help='Maximum tokens for RAG answer')
        parser.add_argument('--rag-temperature', type=float, default=0.5,
                           help='Temperature for LLM generation in RAG')
        parser.add_argument('--rag-model', help='LLM model to use for RAG')
        parser.add_argument('--rag-stream', action='store_true',
                           help='Stream the RAG response')
    
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
    
    # Run new modules if available
    if has_additional_modules:
        if args.all or args.migrate:
            run_db_migration()
        
        if args.all or args.github:
            max_repos = args.github_max if hasattr(args, 'github_max') else None
            run_github_collector(max_repos=max_repos)
            
        if args.all or args.papers:
            max_papers = args.papers_max if hasattr(args, 'papers_max') else None
            force_update = args.force_update if hasattr(args, 'force_update') else False
            run_papers_collector(max_papers=max_papers, force_update=force_update)
            
        if args.all or args.concepts:
            limit = args.concepts_limit if hasattr(args, 'concepts_limit') else None
            source_type = args.concepts_source if hasattr(args, 'concepts_source') else None
            run_concept_extractor(limit=limit, source_type=source_type)
    
    # Run vector search modules if available
    if has_vector_search:
        if args.generate_embeddings:
            source_type = args.embeddings_source if hasattr(args, 'embeddings_source') else None
            limit = args.embeddings_limit if hasattr(args, 'embeddings_limit') else None
            batch_size = args.embeddings_batch if hasattr(args, 'embeddings_batch') else 50
            chunk_size = args.embeddings_chunk if hasattr(args, 'embeddings_chunk') else 500
            chunk_overlap = args.embeddings_overlap if hasattr(args, 'embeddings_overlap') else 100
            force = args.embeddings_force if hasattr(args, 'embeddings_force') else False
            
            run_embedding_generation(
                source_type=source_type, 
                limit=limit,
                batch_size=batch_size,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                force=force
            )
        
        if args.vector_search:
            top_k = args.search_top_k if hasattr(args, 'search_top_k') else 5
            source_type = args.search_source if hasattr(args, 'search_source') else None
            in_memory_index = args.in_memory_index if hasattr(args, 'in_memory_index') else False
            
            run_vector_search(
                query=args.vector_search,
                top_k=top_k,
                source_type=source_type,
                in_memory_index=in_memory_index
            )
        
        if args.hybrid_search:
            top_k = args.search_top_k if hasattr(args, 'search_top_k') else 5
            source_type = args.search_source if hasattr(args, 'search_source') else None
            vector_weight = args.vector_weight if hasattr(args, 'vector_weight') else None
            keyword_weight = args.keyword_weight if hasattr(args, 'keyword_weight') else None
            adaptive = args.adaptive_weights if hasattr(args, 'adaptive_weights') else True
            
            run_hybrid_search(
                query=args.hybrid_search,
                top_k=top_k,
                source_type=source_type,
                vector_weight=vector_weight,
                keyword_weight=keyword_weight,
                adaptive=adaptive
            )
    
    # Run RAG modules if available
    if has_rag and args.rag_query:
        top_k = args.search_top_k if hasattr(args, 'search_top_k') else 5
        source_type = args.search_source if hasattr(args, 'search_source') else None
        vector_weight = args.vector_weight if hasattr(args, 'vector_weight') else None
        keyword_weight = args.keyword_weight if hasattr(args, 'keyword_weight') else None
        search_type = args.rag_search_type if hasattr(args, 'rag_search_type') else 'hybrid'
        max_tokens_context = args.rag_max_tokens_context if hasattr(args, 'rag_max_tokens_context') else 4000
        max_tokens_answer = args.rag_max_tokens_answer if hasattr(args, 'rag_max_tokens_answer') else 1000
        temperature = args.rag_temperature if hasattr(args, 'rag_temperature') else 0.5
        model = args.rag_model if hasattr(args, 'rag_model') else None
        stream = args.rag_stream if hasattr(args, 'rag_stream') else False
        
        run_rag_query(
            query=args.rag_query,
            search_type=search_type,
            source_type=source_type,
            top_k=top_k,
            vector_weight=vector_weight,
            keyword_weight=keyword_weight,
            max_tokens_context=max_tokens_context,
            max_tokens_answer=max_tokens_answer,
            temperature=temperature,
            model=model,
            stream=stream
        )
    
    if args.all or args.web:
        run_web_interface()
    
    # If no arguments provided, show help
    no_args = not (args.download or args.transcribe or args.summarize or 
                  args.index or args.web or args.all)
    
    # Check for new arguments if modules are available
    if has_additional_modules:
        no_args = no_args and not (args.migrate or args.github or args.papers or args.concepts)
    
    # Check for vector search arguments
    if has_vector_search:
        no_args = no_args and not (args.generate_embeddings or args.vector_search or args.hybrid_search)
    
    if no_args:
        parser.print_help()

if __name__ == "__main__":
    main() 