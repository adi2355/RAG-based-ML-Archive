#!/usr/bin/env python3
"""
Context File CLI - A tool for managing the project context file.

This script provides a command-line interface for interacting with the
project context file, which stores important discussions and decisions
for future reference.

Usage:
    ./context_cli.py read [topic]     - Read the context file or a specific topic
    ./context_cli.py add <topic>      - Add a new section (will prompt for content)
    ./context_cli.py search <term>    - Search for a term in the context file
    ./context_cli.py summarize        - Summarize all topics in the context file
    ./context_cli.py decisions        - List all key decisions from the context file
    ./context_cli.py save <filename>  - Save an external file to the context
"""

import os
import sys
import argparse
import tempfile
import subprocess
from typing import Optional, List

# Import the context handler
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from context_handler import ContextFileHandler
except ImportError:
    # If not installed, define the context handler inline
    
    import re
    import datetime
    from typing import Optional, List, Tuple

    class ContextFileHandler:
        """Handler for managing the project context file."""
        
        def __init__(self, context_file_path: str = "context.md"):
            """Initialize the context file handler."""
            self.context_file_path = context_file_path
            self._ensure_context_file_exists()
        
        def _ensure_context_file_exists(self) -> None:
            """Create the context file if it doesn't exist."""
            if not os.path.exists(self.context_file_path):
                with open(self.context_file_path, "w") as f:
                    f.write(f"# Project Context File\n\nCreated on {datetime.date.today()}\n\n")
        
        def read_context(self, topic: Optional[str] = None) -> str:
            """Read the context file or a specific topic section."""
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
            """Add a new context section to the file."""
            try:
                date_str = datetime.date.today().strftime("%Y-%m-%d")
                
                with open(self.context_file_path, "a") as f:
                    f.write(f"\n## {date_str} - {topic}\n\n{content}\n")
                
                return f"Successfully added new context section: {topic}"
            except Exception as e:
                return f"Error adding to context file: {str(e)}"
        
        def search_context(self, term: str) -> str:
            """Search for a term in the context file."""
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
            """Provide a summary of all topics in the context file."""
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
            """Extract all decisions from the context file."""
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
            """Append a raw discussion log to the context file."""
            try:
                date_str = datetime.date.today().strftime("%Y-%m-%d")
                
                with open(self.context_file_path, "a") as f:
                    f.write(f"\n## {date_str} - Discussion Log\n\n```\n{discussion}\n```\n")
                
                return "Successfully appended discussion to context file"
            except Exception as e:
                return f"Error appending discussion: {str(e)}"


def get_editor_input(initial_text: str = "") -> str:
    """
    Open a text editor to get multi-line input from the user.
    
    Args:
        initial_text: Initial text to put in the editor
        
    Returns:
        Text entered by the user
    """
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as temp:
        temp_filename = temp.name
        temp.write(initial_text.encode('utf-8'))
        
    # Determine which editor to use
    editor = os.environ.get('EDITOR', 'nano')
    
    # Open the editor
    try:
        subprocess.call([editor, temp_filename])
        
        # Read the edited content
        with open(temp_filename, 'r') as temp:
            content = temp.read()
            
        # Clean up
        os.unlink(temp_filename)
        
        return content
    except Exception as e:
        print(f"Error using editor: {str(e)}")
        os.unlink(temp_filename)
        return ""


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="Manage the project context file")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Read command
    read_parser = subparsers.add_parser("read", help="Read the context file")
    read_parser.add_argument("topic", nargs="?", help="Optional topic to read")
    
    # Add command
    add_parser = subparsers.add_parser("add", help="Add a new section to the context file")
    add_parser.add_argument("topic", help="Topic for the new section")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search the context file")
    search_parser.add_argument("term", help="Term to search for")
    
    # Summarize command
    subparsers.add_parser("summarize", help="Summarize all topics in the context file")
    
    # Decisions command
    subparsers.add_parser("decisions", help="List all key decisions from the context file")
    
    # Save command
    save_parser = subparsers.add_parser("save", help="Save an external file to the context")
    save_parser.add_argument("filename", help="File to save")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create the context handler
    context_file_path = os.environ.get("CONTEXT_FILE", "context.md")
    handler = ContextFileHandler(context_file_path)
    
    # Process commands
    if args.command == "read":
        print(handler.read_context(args.topic))
    
    elif args.command == "add":
        print(f"Adding new context section: {args.topic}")
        print("Enter content (press Ctrl+D when finished):")
        
        # Get content from editor
        editor_instructions = f"# {args.topic}\n\nEnter your content here...\n\n"
        editor_instructions += "## Structure Suggestions:\n\n"
        editor_instructions += "### Key Decisions\n\n- Decision 1\n- Decision 2\n\n"
        editor_instructions += "### Action Items\n\n- [ ] Action 1\n- [ ] Action 2\n"
        
        content = get_editor_input(editor_instructions)
        
        # Remove the instructions if they weren't edited
        content = content.replace(editor_instructions, "")
        
        if content.strip():
            print(handler.add_context(args.topic, content))
        else:
            print("No content entered, context not updated.")
    
    elif args.command == "search":
        print(handler.search_context(args.term))
    
    elif args.command == "summarize":
        print(handler.summarize_context())
    
    elif args.command == "decisions":
        decisions = handler.extract_decisions()
        
        if not decisions:
            print("No decisions found in context file.")
            return
        
        print("# Key Decisions")
        print()
        
        for date, topic, decision_list in decisions:
            print(f"## {date} - {topic}")
            
            for decision in decision_list:
                print(f"- {decision}")
            
            print()
    
    elif args.command == "save":
        if not os.path.exists(args.filename):
            print(f"Error: File {args.filename} not found.")
            return
        
        try:
            with open(args.filename, "r") as f:
                content = f.read()
            
            filename = os.path.basename(args.filename)
            print(handler.add_context(f"File: {filename}", f"Saved file content:\n\n```\n{content}\n```"))
        except Exception as e:
            print(f"Error saving file to context: {str(e)}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 