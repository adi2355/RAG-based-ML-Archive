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
    has_db_migration = True
except ImportError:
    has_db_migration = False

try:
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

# Try to import knowledge graph module
try:
    import knowledge_graph
    has_knowledge_graph = True
except ImportError:
    has_knowledge_graph = False

# Try to import evaluation modules
try:
    from evaluation import dashboard
    from evaluation.test_runner import RAGTestRunner
    has_evaluation = True
except ImportError:
    has_evaluation = False

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

def run_downloader(force_refresh=False, use_auth=True):
    """Run the Instagram downloader"""
    logger.info("Starting Instagram content download")
    start_time = time()
    downloader.download_from_instagram(force_refresh=force_refresh, use_auth=use_auth)
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
    if args.no_batch:
        logger.info("Batch processing disabled, using sequential processing")
        summarizer.summarize_transcripts(use_batch_api=False)
    else:
        logger.info("Using batch processing with Claude API for cost savings (50% cheaper)")
        summarizer.summarize_transcripts(use_batch_api=True)
    logger.info(f"Summarization completed in {time() - start_time:.2f} seconds")

def run_indexer():
    """Run the knowledge base indexer"""
    logger.info("Starting indexing of transcripts")
    start_time = time()
    indexer.index_transcripts()
    logger.info(f"Indexing completed in {time() - start_time:.2f} seconds")

def run_web_interface(port=5000, debug=False):
    """Run the web interface with API endpoints"""
    from app import app
    
    # The app module already registers all available blueprints when imported
    # So we don't need to register them again, just run the app
    logger.info(f"Starting web interface on port {port}, debug={debug}")
    logger.info(f"Registered blueprints: {list(app.blueprints.keys())}")
    
    # Run the app
    app.run(host='0.0.0.0', port=port, debug=debug)

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

def run_concept_extractor(limit=None, source_type=None, batch=False, batch_size=5, force=False):
    """Run concept extraction on content"""
    if not has_additional_modules:
        logger.error("Concept extractor module not available")
        return
    
    logger.info("Starting concept extraction")
    start_time = time()
    
    if batch:
        logger.info(f"Processing content in batch mode with batch size {batch_size}")
        processed = concept_extractor.process_in_batches(batch_size=batch_size, force=force)
        logger.info(f"Batch processing completed. Processed {processed} items.")
        return processed
    
    if source_type:
        logger.info(f"Processing {limit or 'all'} items from source type: {source_type}")
        processed = concept_extractor.process_unprocessed_content(limit=limit or 5, source_type=source_type, force=force)
        logger.info(f"Processed {processed} items from {source_type}")
    else:
        # Process some content from each source type
        total_processed = 0
        for src_type in ["research_paper", "github", "instagram"]:
            logger.info(f"Processing source type: {src_type}")
            processed = concept_extractor.process_unprocessed_content(limit=limit or 3, source_type=src_type, force=force)
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

def run_evaluation_dashboard(port=5050, debug=False):
    """Run the evaluation dashboard"""
    try:
        from evaluation import evaluation_bp
        from app import app
        
        # Register the evaluation blueprint
        app.register_blueprint(evaluation_bp, url_prefix='/evaluation')
        
        # Add a redirect from root to evaluation dashboard
        @app.route('/')
        def redirect_to_evaluation():
            return redirect(url_for('evaluation_bp.evaluation_dashboard'))
            
        # Run the app
        app.run(host='0.0.0.0', port=port, debug=debug)
    except ImportError:
        logger.error("Evaluation module not available")
        
def run_create_test_dataset(concept_queries=15, content_queries=10):
    """Create a test dataset for RAG evaluation"""
    if not has_evaluation:
        logger.error("Evaluation module not available")
        return False
    
    logger.info("Creating test dataset for RAG evaluation")
    start_time = time()
    
    from evaluation.test_queries import TestQueryGenerator
    generator = TestQueryGenerator()
    dataset_id = generator.create_test_dataset(
        concept_queries=concept_queries,
        content_queries=content_queries
    )
    
    logger.info(f"Test dataset creation completed in {time() - start_time:.2f} seconds")
    logger.info(f"Dataset ID: {dataset_id}")
    return True

def run_evaluation_tests(dataset_id, search_type='hybrid', top_k=10, vector_weight=0.7, keyword_weight=0.3):
    """Run retrieval tests on a test dataset"""
    if not has_evaluation:
        logger.error("Evaluation module not available")
        return False
    
    logger.info(f"Running retrieval tests on dataset {dataset_id}")
    start_time = time()
    
    test_runner = RAGTestRunner()
    results = test_runner.run_retrieval_tests(
        dataset_id=dataset_id,
        search_types=[search_type],
        top_k=[top_k],
        vector_weights=[vector_weight],
        keyword_weights=[keyword_weight]
    )
    
    logger.info(f"Retrieval tests completed in {time() - start_time:.2f} seconds")
    logger.info(f"Results: {results}")
    return True

def run_answer_tests(dataset_id, search_type='hybrid', top_k=5, vector_weight=0.7, keyword_weight=0.3, max_queries=10):
    """Run answer quality tests on a test dataset"""
    if not has_evaluation:
        logger.error("Evaluation module not available")
        return False
    
    logger.info(f"Running answer quality tests on dataset {dataset_id}")
    start_time = time()
    
    test_runner = RAGTestRunner()
    results = test_runner.run_answer_tests(
        dataset_id=dataset_id,
        search_type=search_type,
        top_k=top_k,
        vector_weight=vector_weight,
        keyword_weight=keyword_weight,
        max_queries=max_queries
    )
    
    logger.info(f"Answer quality tests completed in {time() - start_time:.2f} seconds")
    logger.info(f"Results: {results}")
    return True

def main():
    """Main function to parse arguments and run the system"""
    parser = argparse.ArgumentParser(description='Instagram Knowledge Base System')

    # Basic arguments
    parser.add_argument('--download', action='store_true', help='Download content from Instagram')
    parser.add_argument('--refresh-force', action='store_true', help='Force download regardless of refresh schedule')
    parser.add_argument('--no-auth', action='store_true', help='Run Instagram download without authentication (reduces API limits but may help with some errors)')
    parser.add_argument('--transcribe', action='store_true', help='Extract and transcribe audio from videos')
    parser.add_argument('--summarize', action='store_true', help='Summarize transcripts using Claude')
    parser.add_argument('--no-batch', action='store_true', help='Disable batch processing when summarizing (higher cost but potentially faster)')
    parser.add_argument('--index', action='store_true', help='Index transcripts in the knowledge base')
    parser.add_argument('--web', action='store_true', help='Run the web interface')
    parser.add_argument('--port', type=int, default=5000, help='Port for the web interface (default: 5000)')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--all', action='store_true', help='Run all steps')

    # Additional modules arguments
    parser.add_argument('--migrate', action='store_true', help='Migrate database to support multiple content sources')
    parser.add_argument('--github', action='store_true', help='Collect content from GitHub repositories')
    parser.add_argument('--github-max', type=int, help='Maximum number of GitHub repositories to process')
    parser.add_argument('--papers', action='store_true', help='Collect content from research papers')
    parser.add_argument('--papers-max', type=int, help='Maximum number of papers to process')
    parser.add_argument('--papers-force', action='store_true', help='Force update of existing papers')

    # Concept extraction arguments
    parser.add_argument('--concepts', action='store_true', help='Extract AI/ML concepts from content')
    parser.add_argument('--concepts-limit', type=int, help='Maximum number of content items to process for concept extraction')
    parser.add_argument('--concepts-source', help='Source type to process (instagram, github, research_paper)')
    parser.add_argument('--concepts-batch', action='store_true', help='Process in batches')
    parser.add_argument('--concepts-batch-size', type=int, default=5, help='Batch size for concept extraction')
    parser.add_argument('--concepts-force', action='store_true', help='Force re-extraction of concepts')
    
    # Knowledge graph arguments
    parser.add_argument('--kg-analyze', action='store_true', help='Analyze knowledge graph for statistics')
    parser.add_argument('--kg-visualize', action='store_true', help='Visualize knowledge graph')
    parser.add_argument('--kg-concept', type=int, help='Analyze a specific concept by ID')
    parser.add_argument('--kg-search', help='Search for concepts matching term')
    parser.add_argument('--kg-output-dir', help='Output directory for visualizations')
    
    # Embedding generation arguments
    parser.add_argument('--embeddings', action='store_true', help='Generate embeddings for content')
    parser.add_argument('--embeddings-source', help='Source type to process (instagram, github, research_paper)')
    parser.add_argument('--embeddings-limit', type=int, help='Maximum number of content items to process for embeddings')
    parser.add_argument('--embeddings-batch-size', type=int, default=50, help='Batch size for embedding generation')
    parser.add_argument('--embeddings-chunk-size', type=int, default=500, help='Chunk size for embedding generation')
    parser.add_argument('--embeddings-overlap', type=int, default=100, help='Chunk overlap for embedding generation')
    parser.add_argument('--embeddings-force', action='store_true', help='Force re-generation of embeddings')
    
    # Search arguments
    parser.add_argument('--search', help='Perform a vector search with the specified query')
    parser.add_argument('--search-type', choices=['vector', 'hybrid', 'keyword'], default='hybrid', help='Type of search to perform')
    parser.add_argument('--search-source', help='Source type to search (instagram, github, research_paper)')
    parser.add_argument('--top-k', type=int, default=5, help='Number of results to return')
    parser.add_argument('--vector-weight', type=float, default=0.7, help='Weight for vector search (0-1)')
    parser.add_argument('--keyword-weight', type=float, default=0.3, help='Weight for keyword search (0-1)')
    parser.add_argument('--in-memory', action='store_true', help='Use in-memory index for vector search')
    
    # RAG arguments
    parser.add_argument('--rag-query', help='Use RAG to answer the specified query')
    parser.add_argument('--rag-model', help='Model to use for RAG (default is claude-3-sonnet-20240229)')
    parser.add_argument('--rag-tokens-context', type=int, default=4000, help='Maximum tokens to include in context')
    parser.add_argument('--rag-tokens-answer', type=int, default=1000, help='Maximum tokens for answer generation')
    parser.add_argument('--rag-temperature', type=float, default=0.5, help='Temperature for answer generation')
    parser.add_argument('--rag-stream', action='store_true', help='Stream the response')
    
    # Evaluation arguments
    parser.add_argument('--evaluation-dashboard', action='store_true', help='Run the evaluation dashboard')
    parser.add_argument('--evaluation-port', type=int, default=5050, help='Port for the evaluation dashboard')
    parser.add_argument('--create-test-dataset', action='store_true', help='Create a test dataset for RAG evaluation')
    parser.add_argument('--concept-queries', type=int, default=15, help='Number of concept queries to generate')
    parser.add_argument('--content-queries', type=int, default=10, help='Number of content queries to generate')
    parser.add_argument('--evaluation-tests', type=int, help='Run retrieval tests on the specified dataset ID')
    parser.add_argument('--answer-tests', type=int, help='Run answer quality tests on the specified dataset ID')
    parser.add_argument('--max-queries', type=int, default=10, help='Maximum number of queries to test for answer quality')
    
    # API and web interface arguments
    parser.add_argument('--api', action='store_true', help='Run only the API server')
    parser.add_argument('--api-port', type=int, default=5000, help='Port for the API server')
    
    args = parser.parse_args()
    
    setup()
    
    # First run migrations if needed
    if args.migrate or args.all:
        run_db_migration()

    # Then collect content if requested
    if args.download or args.all:
        run_downloader(force_refresh=args.refresh_force, use_auth=not args.no_auth)
    
    if args.github or args.all:
        run_github_collector(args.github_max)
        
    if args.papers or args.all:
        run_papers_collector(args.papers_max, args.papers_force)

    # Then process content
    if args.transcribe or args.all:
        run_transcriber()
    
    if args.summarize or args.all:
        run_summarizer()
        
    if args.concepts or args.all:
        run_concept_extractor(
            limit=args.concepts_limit, 
            source_type=args.concepts_source,
            batch=args.concepts_batch,
            batch_size=args.concepts_batch_size,
            force=args.concepts_force
        )
        
    if args.kg_analyze:
        if has_knowledge_graph:
            run_knowledge_graph(analyze=True)
        else:
            logger.error("Knowledge graph module not available")
    
    if args.kg_visualize:
        if has_knowledge_graph:
            run_knowledge_graph(visualize=True, output_dir=args.kg_output_dir)
        else:
            logger.error("Knowledge graph module not available")
            
    if args.kg_concept:
        if has_knowledge_graph:
            run_knowledge_graph(concept_id=args.kg_concept, output_dir=args.kg_output_dir)
        else:
            logger.error("Knowledge graph module not available")
            
    if args.kg_search:
        if has_knowledge_graph:
            run_knowledge_graph(search_term=args.kg_search, output_dir=args.kg_output_dir)
        else:
            logger.error("Knowledge graph module not available")

    if args.embeddings or args.all:
        run_embedding_generation(
            source_type=args.embeddings_source, 
            limit=args.embeddings_limit, 
            batch_size=args.embeddings_batch_size,
            chunk_size=args.embeddings_chunk_size,
            chunk_overlap=args.embeddings_overlap,
            force=args.embeddings_force
        )

    if args.index or args.all:
        run_indexer()
        
    # Handle search-related arguments
    if args.search:
        if args.search_type == 'vector':
            results = run_vector_search(
                query=args.search, 
                top_k=args.top_k, 
                source_type=args.search_source,
                in_memory_index=args.in_memory
            )
            print(f"\nTop {args.top_k} results for vector search: '{args.search}'")
        else:
            results = run_hybrid_search(
                query=args.search, 
                top_k=args.top_k, 
                source_type=args.search_source,
                vector_weight=args.vector_weight,
                keyword_weight=args.keyword_weight
            )
            print(f"\nTop {args.top_k} results for {args.search_type} search: '{args.search}'")
        
        print("\n" + "-"*80)
        for i, result in enumerate(results):
            print(f"\n{i+1}. {result.get('title', 'No title')} (Score: {result.get('similarity', 0):.4f})")
            print(f"   Source: {result.get('source_type', 'Unknown')}, ID: {result.get('content_id', 'Unknown')}")
            print(f"   {result.get('snippet', '')[:200]}...")
        print("\n" + "-"*80)
    
    # Handle RAG query
    if args.rag_query:
        run_rag_query(
            query=args.rag_query,
            search_type=args.search_type,
            source_type=args.search_source,
            top_k=args.top_k,
            vector_weight=args.vector_weight,
            keyword_weight=args.keyword_weight,
            max_tokens_context=args.rag_tokens_context,
            max_tokens_answer=args.rag_tokens_answer,
            temperature=args.rag_temperature,
            model=args.rag_model,
            stream=args.rag_stream
        )
    
    # Handle evaluation arguments
    if args.evaluation_dashboard:
        run_evaluation_dashboard(port=args.evaluation_port, debug=args.debug)
        
    if args.create_test_dataset:
        run_create_test_dataset(
            concept_queries=args.concept_queries,
            content_queries=args.content_queries
        )
        
    if args.evaluation_tests:
        run_evaluation_tests(
            dataset_id=args.evaluation_tests,
            search_type=args.search_type,
            top_k=args.top_k,
            vector_weight=args.vector_weight,
            keyword_weight=args.keyword_weight
        )
        
    if args.answer_tests:
        run_answer_tests(
            dataset_id=args.answer_tests,
            search_type=args.search_type,
            top_k=args.top_k,
            vector_weight=args.vector_weight,
            keyword_weight=args.keyword_weight,
            max_queries=args.max_queries
        )
        
    # Run the web interface if requested
    if args.api:
        run_web_interface(port=args.api_port, debug=args.debug)
        
    if args.web or args.all:
        run_web_interface(port=args.port, debug=args.debug)

if __name__ == "__main__":
    main() 