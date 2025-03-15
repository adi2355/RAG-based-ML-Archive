#!/usr/bin/env python3
"""
Command Logger - A utility for viewing and searching tracked commands.

This script provides a command-line interface for viewing and searching
commands that have been automatically tracked by the Chat Command Auto-Tracker.

Usage:
    ./command_logger.py list             - List all tracked commands
    ./command_logger.py search <term>    - Search for commands by keyword
    ./command_logger.py view <command>   - View details of a specific command
    ./command_logger.py stats            - Show statistics about tracked commands
"""

import os
import sys
import re
import argparse
from datetime import datetime
from typing import List, Dict, Optional, Tuple


def read_commands_file(file_path='commands.md') -> str:
    """Read the contents of the commands tracking file."""
    if not os.path.exists(file_path):
        print(f"Commands file not found: {file_path}")
        print("No commands have been tracked yet.")
        return ""
    
    with open(file_path, 'r') as f:
        return f.read()


def parse_commands(content: str) -> List[Dict[str, str]]:
    """Parse the commands file content into a structured format."""
    if not content:
        return []
    
    # Split by command sections (starting with ## )
    sections = re.split(r'\n## ', content)
    
    # First section is the header, skip it
    sections = sections[1:] if sections else []
    
    commands = []
    for section in sections:
        # Get command name (first line)
        lines = section.strip().split('\n')
        name = lines[0] if lines else "Unknown"
        
        # Extract detection timestamp
        detected_match = re.search(r'\*\*Detected:\*\* (.*)', section)
        detected = detected_match.group(1) if detected_match else "Unknown"
        
        # Extract description
        desc_start = section.find('### Description')
        if desc_start != -1:
            desc_end = section.find('###', desc_start + 1)
            description = section[desc_start + 16:desc_end].strip() if desc_end != -1 else section[desc_start + 16:].strip()
        else:
            description = "No description available"
        
        # Extract implementation
        impl_start = section.find('### Implementation')
        if impl_start != -1:
            impl_end = section.find('###', impl_start + 1)
            implementation = section[impl_start + 21:impl_end].strip() if impl_end != -1 else section[impl_start + 21:].strip()
        else:
            implementation = "No implementation available"
        
        commands.append({
            'name': name,
            'detected': detected,
            'description': description,
            'implementation': implementation,
            'full_content': section
        })
    
    return commands


def list_commands(commands: List[Dict[str, str]]) -> None:
    """Display a list of all tracked commands."""
    if not commands:
        print("No commands have been tracked yet.")
        return
    
    print(f"\n{'='*60}")
    print(f" {'COMMAND NAME':<25} | {'DETECTED':<20} | {'DESCRIPTION'}")
    print(f"{'-'*60}")
    
    for cmd in commands:
        # Truncate description if too long
        desc = cmd['description'].replace('\n', ' ')
        if len(desc) > 30:
            desc = desc[:27] + '...'
        
        print(f" {cmd['name']:<25} | {cmd['detected']:<20} | {desc}")
    
    print(f"{'='*60}")
    print(f"Total commands: {len(commands)}\n")


def search_commands(commands: List[Dict[str, str]], term: str) -> None:
    """Search for commands matching the given term."""
    if not commands:
        print("No commands have been tracked yet.")
        return
    
    term = term.lower()
    matches = []
    
    for cmd in commands:
        # Search in name, description, and implementation
        text = (
            cmd['name'].lower() + ' ' + 
            cmd['description'].lower() + ' ' + 
            cmd['implementation'].lower()
        )
        
        if term in text:
            matches.append(cmd)
    
    if not matches:
        print(f"No commands found matching '{term}'")
        return
    
    print(f"\n{'='*60}")
    print(f" SEARCH RESULTS FOR: '{term}'")
    print(f"{'-'*60}")
    
    for cmd in matches:
        # Truncate description if too long
        desc = cmd['description'].replace('\n', ' ')
        if len(desc) > 30:
            desc = desc[:27] + '...'
        
        print(f" {cmd['name']:<25} | {cmd['detected']:<20} | {desc}")
    
    print(f"{'='*60}")
    print(f"Found {len(matches)} matching commands\n")


def view_command(commands: List[Dict[str, str]], name: str) -> None:
    """View the details of a specific command."""
    if not commands:
        print("No commands have been tracked yet.")
        return
    
    # Find command by name (case insensitive)
    found = None
    for cmd in commands:
        if cmd['name'].lower() == name.lower():
            found = cmd
            break
    
    if not found:
        print(f"Command not found: {name}")
        return
    
    print(f"\n{'='*60}")
    print(f" COMMAND: {found['name']}")
    print(f"{'-'*60}")
    print(f"Detected: {found['detected']}")
    print(f"\nDescription:")
    print(f"{found['description']}")
    print(f"\nImplementation:")
    print(f"{found['implementation']}")
    print(f"{'='*60}\n")


def show_stats(commands: List[Dict[str, str]]) -> None:
    """Show statistics about tracked commands."""
    if not commands:
        print("No commands have been tracked yet.")
        return
    
    # Get date range
    dates = []
    for cmd in commands:
        try:
            date_str = cmd['detected'].split()[0]  # Extract date part only
            dates.append(datetime.strptime(date_str, "%Y-%m-%d"))
        except (ValueError, IndexError):
            continue
    
    date_range = f"{min(dates).strftime('%Y-%m-%d')} to {max(dates).strftime('%Y-%m-%d')}" if dates else "Unknown"
    
    # Count implementations
    with_impl = sum(1 for cmd in commands if "No implementation" not in cmd['implementation'])
    
    # Find longest and shortest
    longest = max(commands, key=lambda c: len(c['implementation']))
    shortest = min(commands, key=lambda c: len(c['implementation']))
    
    print(f"\n{'='*60}")
    print(f" COMMAND TRACKING STATISTICS")
    print(f"{'-'*60}")
    print(f"Total commands tracked: {len(commands)}")
    print(f"Date range: {date_range}")
    print(f"Commands with implementation: {with_impl} ({with_impl/len(commands)*100:.1f}%)")
    print(f"Commands without implementation: {len(commands) - with_impl}")
    print(f"Longest implementation: {longest['name']} ({len(longest['implementation'])} chars)")
    print(f"Shortest implementation: {shortest['name']} ({len(shortest['implementation'])} chars)")
    print(f"{'='*60}\n")


def main():
    """Main entry point for the command logger."""
    parser = argparse.ArgumentParser(description="View and search tracked commands")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # List command
    subparsers.add_parser("list", help="List all tracked commands")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search for commands by keyword")
    search_parser.add_argument("term", help="Search term")
    
    # View command
    view_parser = subparsers.add_parser("view", help="View details of a specific command")
    view_parser.add_argument("name", help="Command name to view")
    
    # Stats command
    subparsers.add_parser("stats", help="Show statistics about tracked commands")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Read and parse commands file
    content = read_commands_file()
    commands = parse_commands(content)
    
    # Process commands
    if args.command == "list":
        list_commands(commands)
    
    elif args.command == "search":
        search_commands(commands, args.term)
    
    elif args.command == "view":
        view_command(commands, args.name)
    
    elif args.command == "stats":
        show_stats(commands)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 