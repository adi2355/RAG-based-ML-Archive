# Project Context File

Created on 2025-03-15


## 2025-03-15 - Claude Batch Processing Implementation

We implemented batch processing capability in summarizer.py using Claude's Message Batches API. This implementation significantly improves efficiency and reduces costs.

### Key Features

- Uses the Message Batches API from Anthropic to process up to 100 transcripts at once
- Implements efficient transcript grouping and processing
- Provides robust error handling with detailed logging
- Includes automatic retry logic for failed items

### Key Decisions

- Selected Claude 3 Haiku model for cost efficiency
- Added a command-line flag (--no-batch) to allow users to choose between batch and sequential processing
- Implemented a 50% cost reduction compared to sequential processing
- Designed the system to handle both small and large transcript volumes efficiently

### Technical Implementation

- Created a batch_id tracking system using UUID for request identification
- Implemented asynchronous processing with polling for completion status
- Added caching to prevent reprocessing of already summarized content
- Modified run.py to support the new batch processing option as the default behavior


## 2025-03-15 - Comprehensive Project Analysis

The Instagram-Scraper project is a sophisticated AI-powered content collection and analysis system with multiple integrated components.

### Core Components

#### Content Collection

- **Instagram Downloader**: Uses instaloader to fetch videos from configured Instagram accounts, with built-in rate limiting protection through proxy rotation and session management
- **GitHub Repository Collector**: Gathers ML/AI repositories based on a curated list and topic search, storing code and documentation
- **ArXiv Research Paper Collector**: Fetches AI research papers from ArXiv, extracting full text and metadata

#### Processing Pipeline

- **Audio Extraction & Transcription**: Extracts audio from videos and transcribes using Whisper
- **Batch Summarization**: Processes transcripts with Claude API using batch processing for 50% cost reduction
- **Concept Extraction**: Uses Claude to identify AI/ML concepts, entities and their relationships
- **Embedding Generation**: Creates vector embeddings for semantic search using sentence-transformers
- **Text Chunking**: Splits content into optimal chunks for vector embedding and retrieval

#### Knowledge Representation

- **SQLite Database**: Core storage with tables for videos, content sources, embeddings, and concepts
- **Full-Text Search**: FTS4 virtual tables for keyword-based content retrieval
- **Vector Database**: Stores embeddings for semantic similarity search
- **Knowledge Graph**: Maintains concept relationships with network analysis capabilities

#### Retrieval & Analysis

- **Hybrid Search**: Combines vector similarity and keyword matching for optimal results
- **RAG Implementation**: Uses retrieved content to generate context-aware responses
- **Knowledge Graph Analysis**: Provides concept relationship visualization and exploration
- **Evaluation Framework**: Tests and benchmarks retrieval and generation quality

### Technical Architecture

- **Modular Design**: Components are separated into distinct modules with clear interfaces
- **Graceful Degradation**: Core functions work even when optional components are missing
- **API Integration**: Connects with Anthropic Claude, Whisper, and GitHub APIs
- **Web Interface**: Flask-based UI for searching and exploring content

### Key Files

- **run.py**: Main entry point with command-line interface for all functionalities
- **config.py**: Central configuration for accounts, API keys, and system settings
- **downloader.py**: Instagram content collection with rate limit handling
- **transcriber.py**: Audio extraction and transcription using Whisper
- **summarizer.py**: Claude-based transcript summarization with batch processing
- **vector_search.py/hybrid_search.py**: Semantic and keyword search implementation
- **knowledge_graph.py**: AI concept extraction and relationship management
- **embeddings.py**: Vector representation generation for semantic search
- **concept_extractor.py**: Claude-powered AI concept extraction
- **llm_integration.py**: RAG implementation with Claude

### Database Schema

- **videos**: Stores Instagram video metadata, transcripts, and summaries
- **ai_content**: Unified content store for multiple sources (Instagram, GitHub, papers)
- **content_embeddings**: Vector embeddings for semantic search
- **concepts**: AI/ML concepts extracted from content
- **concept_relationships**: Connections between AI concepts
- **tags**: Hashtags and categorization

### Key Technologies

- **Claude API**: For summarization, concept extraction, and RAG responses
- **Whisper**: For audio transcription
- **Sentence Transformers**: For generating vector embeddings
- **NetworkX**: For knowledge graph analysis
- **Plotly/Matplotlib**: For visualizations
- **SQLite**: For database storage
- **Flask**: For web interface

### Deployment Strategy

- Self-contained Python application with comprehensive CLI
- Simple SQLite-based storage requiring no separate database server
- Configurable through single config file
- Dependencies managed through requirements.txt

### Extensibility Points

- Addition of new content sources (configured in CONTENT_SOURCES)
- New search and retrieval methods
- Alternative embedding models
- Additional visualization and analysis tools


## 2025-03-15 - Chat Command Auto-Tracker Implementation

We've implemented an automated system for tracking commands mentioned in chat conversations.

### Key Features

- Automatically detects when commands, scripts, or utilities are mentioned in conversations
- Extracts command names, descriptions, and implementation code from natural language
- Logs everything to a structured commands.md file without requiring explicit requests
- Provides a companion command_logger.py utility for viewing and searching tracked commands

### How It Works

1. The system monitors conversations for patterns indicating command creation
2. When detected, it extracts the command name using several strategies:
   - From natural language (e.g., "I created a data_processor script")
   - From code blocks (e.g., shebang lines or function definitions)
   - From filename patterns or generates a timestamped name if needed
3. It then captures surrounding context and code implementation
4. Everything is timestamped and logged to the commands.md file

### Command Logger Utility

The companion command_logger.py script provides easy access to tracked commands:

- List all tracked commands: ./command_logger.py list
- Search for commands: ./command_logger.py search "keyword"
- View command details: ./command_logger.py view "command_name"
- Show statistics: ./command_logger.py stats

### Benefits

- Automatic documentation of all commands discussed in conversations
- No manual tracking or copying required
- Easy retrieval of previously discussed commands
- Command history is preserved across conversations

