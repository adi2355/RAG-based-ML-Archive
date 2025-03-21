
# System Architecture Overview


![System Architecture]

1. **Data Collection Layer**
2. **Data Processing Layer**
3. **Knowledge Extraction Layer**
4. **Vector Embedding & Indexing Layer**
5. **Retrieval Layer**
6. **LLM Integration Layer**
7. **User Interface Layer**

## Data Flow Analysis

### 1. Data Collection

Data enters your system from three primary sources:

- **Instagram Videos** (`downloader.py`)
  - Uses proxies and account rotation to avoid rate limiting
  - Stores downloaded videos and metadata in `data/downloads`

- **GitHub Repositories** (`github_collector.py`)
  - Collects repositories via GitHub API based on topics/stars
  - Extracts READMEs and repository metadata

- **Research Papers** (`arxiv_collector.py`)
  - Downloads papers from ArXiv and custom URLs
  - Stores PDFs in `data/papers/pdf`

### 2. Data Processing

Each content type undergoes specialized processing:

- **Instagram Videos**:
  ```
  Video → Audio Extraction → Whisper Transcription → Claude Summarization
  ```
  - `transcriber.py` extracts audio and transcribes using Whisper
  - `summarizer.py` generates structured summaries with Claude
  - Uses batch API for cost efficiency (50% cheaper)

- **Research Papers**:
  ```
  PDF → Mistral OCR → Text Extraction → Structured Storage
  ```
  - `mistral_ocr.py` extracts text with superior quality
  - Handles large PDFs through chunking
  - Falls back to PyPDF2 if needed

- **GitHub Repos**:
  ```
  Repository → README Extraction → Metadata Analysis → Storage
  ```

### 3. Database Storage

All processed content flows into a SQLite database (`knowledge_base.db`) with a unified schema:

```
Collection → Processing → ai_content table → Embedding → Indexing
```

Key tables include:
- `ai_content`: Central table for all content types
- `videos`, `research_papers`, `github_repos`: Type-specific tables
- `content_embeddings`: Stores vector embeddings for chunks
- `concepts`, `concept_relationships`: Knowledge graph structure

### 4. Knowledge Extraction

Your system builds a knowledge graph through:

```
Content → Claude Analysis → Concept Extraction → Relationship Mapping
```

- `concept_extractor.py` uses Claude to identify AI/ML concepts
- Concepts are categorized and linked to form a knowledge graph
- Relationships between concepts are stored with confidence scores

### 5. Vector Embedding & Indexing

Text content is prepared for semantic search:

```
Content → Chunking → Embedding Generation → Vector Storage
```

- `chunking.py` splits content into optimal overlapping chunks
- `embeddings.py` generates vector embeddings for each chunk
- Both in-memory and database-based indexes are supported

### 6. Retrieval System

When a query arrives:

```
Query → Hybrid Search → Context Selection → LLM Prompt Construction
```

- `hybrid_search.py` combines vector and keyword search
- Adaptive weighting based on query characteristics
- `context_builder.py` selects diverse, relevant context
- Content is formatted with source citations

### 7. LLM Integration

The final response generation:

```
Context + Query → Claude API → Structured Response → User
```

- `llm_integration.py` handles communication with Claude API
- Supports streaming responses
- Includes citation of sources in responses

## Key Strengths of Your System

1. **Multi-Source Integration**: Unified approach to diverse content types
2. **Advanced OCR**: Superior text extraction with Mistral OCR
3. **Hybrid Search**: Combines semantic and keyword search adaptively
4. **Knowledge Graph**: Goes beyond simple retrieval to concept relationships
5. **Batch Processing**: Cost-efficient API usage
6. **Evaluation Framework**: Built-in metrics for retrieval quality

## Data Transformation Examples

When a research paper is processed:
1. PDF is downloaded → `data/papers/pdf/[paper_id].pdf`
2. Text is extracted using Mistral OCR
3. Content stored in `ai_content` with `source_type_id = 2`
4. Concepts extracted and stored in `concepts` table
5. Text is chunked and embeddings generated
6. Paper becomes searchable through vector and keyword search

When a user query is processed:
1. Query analyzed for characteristics (factual, conversational, etc.)
2. Hybrid search performed with appropriate weights
3. Top results selected with diversity consideration
4. Context formatted with source citations
5. Claude generates response based on provided context
6. Response returned with source attribution

This comprehensive RAG system efficiently moves data from collection through processing, enrichment, indexing, and finally to retrieval and response generation.
