import os
import re
import datetime
from typing import Optional, List, Tuple

class ContextFileHandler:
    """
    Handler for managing the project context file.
    
    This class provides functionality to read, write, search, and summarize 
    the content of a project context file, which stores important discussions
    and decisions for future reference.
    """
    
    def __init__(self, context_file_path: str = "context.md"):
        """
        Initialize the context file handler.
        
        Args:
            context_file_path: Path to the context file
        """
        self.context_file_path = context_file_path
        self._ensure_context_file_exists()
    
    def _ensure_context_file_exists(self) -> None:
        """Create the context file if it doesn't exist."""
        if not os.path.exists(self.context_file_path):
            with open(self.context_file_path, "w") as f:
                f.write(f"# Project Context File\n\nCreated on {datetime.date.today()}\n\n")
    
    def read_context(self, topic: Optional[str] = None) -> str:
        """
        Read the context file or a specific topic section.
        
        Args:
            topic: Optional topic to filter by
            
        Returns:
            The content of the context file or specified section
        """
        try:
            with open(self.context_file_path, "r") as f:
                content = f.read()
            
            if topic:
                # Find the topic section
                topic_pattern = re.compile(f"## .*?{re.escape(topic)}.*?\n(.*?)(?=\n## |$)", re.DOTALL)
                match = topic_pattern.search(content)
                
                if match:
                    return f"## Topic: {topic}\n\n{match.group(1).strip()}"
                else:
                    return f"Topic '{topic}' not found in context file."
            
            return content
        except Exception as e:
            return f"Error reading context file: {str(e)}"
    
    def add_context(self, topic: str, content: str) -> str:
        """
        Add a new context section to the file.
        
        Args:
            topic: Topic of the new section
            content: Content to add
            
        Returns:
            Confirmation message
        """
        try:
            date_str = datetime.date.today().strftime("%Y-%m-%d")
            
            with open(self.context_file_path, "a") as f:
                f.write(f"\n## {date_str} - {topic}\n\n{content}\n")
            
            return f"Successfully added new context section: {topic}"
        except Exception as e:
            return f"Error adding to context file: {str(e)}"
    
    def search_context(self, term: str) -> str:
        """
        Search for a term in the context file.
        
        Args:
            term: Search term
            
        Returns:
            Search results as formatted string
        """
        try:
            with open(self.context_file_path, "r") as f:
                content = f.read()
            
            lines = content.split("\n")
            results = []
            
            for i, line in enumerate(lines):
                if term.lower() in line.lower():
                    # Get some context around the match
                    start = max(0, i - 1)
                    end = min(len(lines), i + 2)
                    context_lines = lines[start:end]
                    results.append(f"Line {i+1}: {'...' if start > 0 else ''}")
                    results.extend([f"  {line}" for line in context_lines])
                    results.append("  ...")
            
            if results:
                return f"Search results for '{term}':\n" + "\n".join(results)
            else:
                return f"No results found for '{term}' in context file."
        except Exception as e:
            return f"Error searching context file: {str(e)}"
    
    def summarize_context(self) -> str:
        """
        Provide a summary of all topics in the context file.
        
        Returns:
            Summary of topics with dates
        """
        try:
            with open(self.context_file_path, "r") as f:
                content = f.read()
            
            # Find all topic sections
            topic_pattern = re.compile(r"## ([\d-]+) - (.*?)\n", re.MULTILINE)
            matches = topic_pattern.findall(content)
            
            if not matches:
                return "No topics found in context file."
            
            summary = ["# Context File Summary", ""]
            
            for date, topic in matches:
                summary.append(f"- {date}: {topic}")
            
            return "\n".join(summary)
        except Exception as e:
            return f"Error summarizing context file: {str(e)}"
    
    def extract_decisions(self) -> List[Tuple[str, str, List[str]]]:
        """
        Extract all decisions from the context file.
        
        Returns:
            List of (date, topic, decisions) tuples
        """
        try:
            with open(self.context_file_path, "r") as f:
                content = f.read()
            
            results = []
            
            # Find topic sections
            sections = re.split(r"(?=## [\d-]+ - )", content)
            
            for section in sections:
                if not section.startswith("## "):
                    continue
                
                # Extract the date and topic
                header_match = re.match(r"## ([\d-]+) - (.*?)\n", section)
                if not header_match:
                    continue
                
                date = header_match.group(1)
                topic = header_match.group(2)
                
                # Look for decision sections
                decision_section = re.search(r"### Key Decisions\s*\n((?:- .*?\n)+)", section)
                
                if decision_section:
                    # Extract individual decisions
                    decision_text = decision_section.group(1)
                    decisions = [d.strip('- \n') for d in decision_text.split('\n') if d.strip().startswith('-')]
                    
                    results.append((date, topic, decisions))
            
            return results
        except Exception as e:
            print(f"Error extracting decisions: {str(e)}")
            return []

    def append_discussion(self, discussion: str) -> str:
        """
        Append a raw discussion log to the context file.
        
        Args:
            discussion: Discussion content to append
            
        Returns:
            Confirmation message
        """
        try:
            date_str = datetime.date.today().strftime("%Y-%m-%d")
            
            with open(self.context_file_path, "a") as f:
                f.write(f"\n## {date_str} - Discussion Log\n\n```\n{discussion}\n```\n")
            
            return "Successfully appended discussion to context file"
        except Exception as e:
            return f"Error appending discussion: {str(e)}"


# Example usage:
if __name__ == "__main__":
    context_handler = ContextFileHandler()
    
    # Create or ensure the context file exists
    context_handler._ensure_context_file_exists()
    
    # Add a new context section
    context_handler.add_context(
        "API Design", 
        "We discussed RESTful API design principles and decided to adopt them.\n\n"
        "### Key Decisions\n\n"
        "- Use plural nouns for resource endpoints\n"
        "- Implement proper HTTP status codes\n"
        "- Add comprehensive documentation\n\n"
        "### Action Items\n\n"
        "- [ ] Create API blueprint\n"
        "- [ ] Implement swagger documentation\n"
    )
    
    # Read the whole context
    print(context_handler.read_context())
    
    # Search for a term
    print(context_handler.search_context("API"))
    
    # Get a summary
    print(context_handler.summarize_context()) 