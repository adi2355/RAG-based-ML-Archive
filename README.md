# MistralOCR GitHub Repository Collector

A powerful tool for collecting, processing, and storing AI/ML repositories from GitHub for knowledge bases and research purposes.

## Overview

The GitHub Repository Collector is designed to intelligently gather high-quality AI/ML repositories from GitHub. It goes beyond simple data collection by prioritizing repositories based on quality indicators, selectively processing the most valuable content, and organizing the information for efficient retrieval and analysis.

## Features

- **Intelligent Repository Selection**: Automatically evaluates repositories based on stars, activity, documentation, and maturity
- **Smart Content Prioritization**: Identifies and collects the most valuable files from each repository
- **Specialized Content Processing**:
  - Extracts key insights from Jupyter notebooks
  - Processes Python files to focus on docstrings and important code structures
  - Preserves formatting and structure in markdown documentation
- **Efficient Content Chunking**: Splits content into optimized chunks for vector databases and semantic search
- **Robust Rate Limiting**: Implements adaptive sleep strategies based on GitHub API rate limits
- **Database Integration**: Stores repositories, files, and content chunks in SQLite database
- **Local Content Storage**: Maintains a local file-based archive of all collected GitHub content
- **Configurable Search Criteria**: Supports flexible search queries to target specific repositories or topics

## Installation

### Prerequisites

- Python 3.7+
- SQLite3
- Git

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/MistralOCR.git
   cd MistralOCR
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up GitHub API token (optional but recommended to avoid rate limiting):
   - Create a [GitHub Personal Access Token](https://github.com/settings/tokens)
   - Set it as an environment variable:
     ```bash
     export GITHUB_TOKEN=your_token_here
     ```
   - Or add it to your config.py file:
     ```python
     GITHUB_CONFIG = {
         'api_token': 'your_token_here',
         # other config options
     }
     ```

## Usage

### Basic Usage

1. Run the collector with default settings:
   ```bash
   python github_collector.py
   ```

2. Limit the number of repositories to collect:
   ```bash
   python github_collector.py --max-repos 10
   ```

### Using the Test Script

The test script provides an easy way to test different collection scenarios:

1. Collect a specific repository:
   ```bash
   ./test_github_collector.py --repo tensorflow/tensorflow
   ```

2. Use a custom search query:
   ```bash
   ./test_github_collector.py --query "natural language processing language:python stars:>500"
   ```

3. Set maximum number of repositories:
   ```bash
   ./test_github_collector.py --max-repos 5
   ```

### Configuration

You can customize the collector's behavior by modifying the `config.py` file:

```python
GITHUB_CONFIG = {
    'api_token': 'your_github_token',
    'search_query': 'machine learning language:python stars:>100',
    'sort_by': 'stars',
    'sort_order': 'desc',
    'max_updates_per_run': 50,
    'readme_max_length': 100000
}
```

## How It Works

1. **Repository Discovery**: Uses GitHub's search API to find repositories matching the search criteria
2. **Repository Assessment**: Evaluates repositories based on quality indicators
3. **Content Collection**: Fetches valuable files from selected repositories
4. **Content Processing**: Processes different file types with specialized extractors
5. **Content Chunking**: Splits processed content into optimized chunks for storage
6. **Storage**: 
   - Stores metadata and content in SQLite database for efficient querying
   - Archives all content to local file system for backup and direct access
7. **Retrieval**: Provides methods to access content via database queries or direct file access

## Storage System

The collector implements a dual storage approach to maximize both performance and accessibility:

### Database Storage

The collector uses a SQLite database with the following tables:

- **github_repos**: Stores repository metadata and README content
- **github_repo_files**: Stores individual file contents from repositories
- **github_content_chunks**: Stores processed content chunks for semantic search

### File System Storage

In addition to the database, all collected content is organized and stored in the local file system at:

```
/home/adi235/MistralOCR/Instagram-Scraper/data/github
```

The file structure follows this organization:

```
data/
└── github/
    ├── tensorflow_tensorflow/
    │   ├── metadata.json        # Repository metadata
    │   ├── README.md            # Repository readme
    │   ├── docs/                # Documentation files
    │   ├── examples/            # Example files
    │   └── processed/           # Processed content and chunks
    │
    ├── huggingface_transformers/
    │   ├── metadata.json
    │   └── ...
    │
    └── ...
```

This approach provides several benefits:
- Direct file access for external tools and scripts
- Easy backup and transfer capabilities
- Fallback access if database becomes corrupted
- Human-readable organization for browsing content

## Advanced Usage

### Custom Repository Priority

You can define your own set of repositories to prioritize by modifying the `GITHUB_REPOS` list in `github_collector.py`.

### Custom File Processors

You can extend the collector by adding custom processors for additional file types:

1. Create a new processor function in `github_collector.py`
2. Update the `process_file_content` function to use your processor for specific file extensions

## Troubleshooting

- **Rate Limiting Issues**: If you encounter rate limiting issues, ensure you've set up a GitHub API token
- **Memory Errors**: Adjust the `max_files` parameter in the `collect_valuable_files` function to collect fewer files
- **Database Errors**: Check that you have write permissions to the database file location
- **Storage Space**: Ensure adequate disk space is available in the data directory

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
