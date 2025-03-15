#!/usr/bin/env python3
"""
Context Extractor for AI Knowledge Base Codebase

This script extracts structured context from a Python/Flask codebase, focusing on:
- Function signatures and docstrings
- Class definitions and methods
- Flask routes and blueprints
- HTML template structure
- Command-line arguments

Usage:
  python context_extractor.py [output_file]
"""

import os
import re
import sys
import ast
import json
import inspect
from typing import List, Dict, Tuple, Any, Optional, Set
from collections import defaultdict
import argparse

# Configuration
DEFAULT_CONFIG = {
    "extensions": [".py", ".html", ".js", ".sql"],
    "exclude_dirs": [
        "venv", "__pycache__", "node_modules", 
        "Instagram-Scraper/logs", "Instagram-Scraper/responses", "Instagram-Scraper/visualisations", 
        "Instagram-Scraper/data", "Instagram-Scraper/static", "Instagram-Scraper/venv", 
        "Instagram-Scraper/templates", "site-packages"
    ],
    "exclude_files": [
        ".test.", ".spec.", ".min.js", ".map", "setup.py",
        ".gitignore", "package-lock.json", ".env"
    ],
    "include_dirs": [
        "Instagram-Scraper", "evaluation", "api"
    ],
    "max_file_size": 1024 * 1024,  # 1MB
    "max_lines": 15000,  # Target maximum lines for output
}

class CodeContextExtractor:
    """Extracts structured context from a Python/Flask codebase"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.blueprint_routes = defaultdict(list)
        self.flask_routes = []
        self.command_line_args = []
        self.template_hierarchy = {}
        self.file_summaries = []
        self.total_lines = 0
        # Add new tracking structures
        self.import_graph = defaultdict(set)  # Map of module -> imported modules
        self.function_calls = defaultdict(set)  # Map of function -> called functions
        self.module_functions = defaultdict(set)  # Map of module -> defined functions
        self.api_endpoints = []  # List of API endpoints with metadata
        self.db_tables = {}  # Database table schema
        self.db_relationships = []  # Database relationships
        self.complex_functions = []  # Track functions with high complexity
    
    def should_exclude_file(self, file_path: str) -> bool:
        """Check if a file should be excluded based on config"""
        normalized_path = os.path.normpath(file_path)
        
        # Check excluded patterns
        for pattern in self.config["exclude_files"]:
            if pattern in normalized_path:
                return True
        
        # Check file extension        
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in self.config["extensions"]:
            return True
            
        return False

    def should_exclude_dir(self, dir_path: str) -> bool:
        """Check if a directory should be excluded based on config"""
        normalized_path = os.path.normpath(dir_path)
        dir_name = os.path.basename(normalized_path)
        
        # Check if the directory name itself is in the exclude list
        if dir_name in self.config["exclude_dirs"]:
            print(f"Excluding directory (name match): {normalized_path}")
            return True
            
        # Check if the path contains any of the excluded directory patterns
        for exclude_dir in self.config["exclude_dirs"]:
            if exclude_dir in normalized_path:
                print(f"Excluding directory (path match): {normalized_path} - matched: {exclude_dir}")
                return True
        
        # If we have include_dirs, check if this is in it
        if self.config["include_dirs"]:
            for include_dir in self.config["include_dirs"]:
                if include_dir in normalized_path:
                    return False
            # If not in any include_dirs, exclude it
            return True
                
        return False

    def collect_files(self, root_dir: str) -> List[Tuple[str, str]]:
        """Collect all eligible files in the directory"""
        results = []
        print(f"Collecting files from: {root_dir}")
        print(f"Exclusion patterns: {self.config['exclude_dirs']}")
        
        for root, dirs, files in os.walk(root_dir):
            # Process directories
            dirs_before = len(dirs)
            dirs[:] = [d for d in dirs if not self.should_exclude_dir(os.path.join(root, d))]
            if dirs_before > len(dirs):
                print(f"  In {root}: Filtered out {dirs_before - len(dirs)} directories")
            
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, root_dir)
                
                # Check exclusion patterns
                if self.should_exclude_file(rel_path):
                    continue
                    
                # Check file size
                if os.path.getsize(file_path) > self.config["max_file_size"]:
                    continue
                
                # Check if the file is in an excluded directory
                file_dir = os.path.dirname(file_path)
                if self.should_exclude_dir(file_dir):
                    print(f"  Skipping file in excluded directory: {rel_path}")
                    continue
                    
                results.append((file_path, rel_path))
                
        print(f"Collected {len(results)} files for processing")
        return results

    def calculate_cyclomatic_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate the cyclomatic complexity of a function"""
        # Start with complexity of 1 (one path through the function)
        complexity = 1
        
        # Increment for each statement that branches the control flow
        for subnode in ast.walk(node):
            # If statements
            if isinstance(subnode, ast.If):
                complexity += 1
                
                # Handle complex boolean expressions with 'and'/'or'
                if isinstance(subnode.test, ast.BoolOp):
                    complexity += len(subnode.test.values) - 1
            
            # For loops
            elif isinstance(subnode, ast.For) or isinstance(subnode, ast.AsyncFor):
                complexity += 1
            
            # While loops
            elif isinstance(subnode, ast.While):
                complexity += 1
                
                # Handle complex boolean expressions with 'and'/'or'
                if isinstance(subnode.test, ast.BoolOp):
                    complexity += len(subnode.test.values) - 1
            
            # Try/except blocks (each except handler adds a branch)
            elif isinstance(subnode, ast.Try):
                complexity += len(subnode.handlers)
            
            # List/dict/set comprehensions and generator expressions
            elif isinstance(subnode, (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)):
                complexity += 1
                
            # Match/case statements (Python 3.10+)
            elif hasattr(ast, 'Match') and isinstance(subnode, ast.Match):
                complexity += len(subnode.cases)
                
            # Boolean operations with 'or' (short-circuiting can create branches)
            elif isinstance(subnode, ast.BoolOp) and isinstance(subnode.op, ast.Or):
                complexity += len(subnode.values) - 1
        
        return complexity

    def extract_python_function_info(self, node: ast.FunctionDef) -> Dict[str, Any]:
        """Extract information from a function definition"""
        function_info = {
            "name": node.name,
            "docstring": ast.get_docstring(node) or "",
            "params": []
        }
        
        # Calculate complexity
        complexity = self.calculate_cyclomatic_complexity(node)
        function_info["complexity"] = complexity
        
        # Track complex functions (complexity > 10)
        if complexity > 10:
            self.complex_functions.append({
                "name": node.name,
                "complexity": complexity,
                "line_number": node.lineno,
                "source_file": inspect.getfile(node) if hasattr(node, "_filename") else None
            })
        
        # Extract parameters
        for arg in node.args.args:
            param = {"name": arg.arg}
            if arg.annotation and isinstance(arg.annotation, ast.Name):
                param["type"] = arg.annotation.id
            function_info["params"].append(param)
        
        # Extract default values safely
        try:
            if node.args.defaults:
                default_offset = len(node.args.args) - len(node.args.defaults)
                for i, default in enumerate(node.args.defaults):
                    param_index = default_offset + i
                    if 0 <= param_index < len(function_info["params"]):
                        if isinstance(default, ast.Constant):
                            function_info["params"][param_index]["default"] = default.value
                        else:
                            # Handle non-constant default values
                            function_info["params"][param_index]["default"] = "..."
        except Exception as e:
            # Skip setting defaults if any error occurs
            pass
        
        return function_info

    def extract_class_info(self, node: ast.ClassDef) -> Dict[str, Any]:
        """Extract information from a class definition"""
        class_info = {
            "name": node.name,
            "docstring": ast.get_docstring(node) or "",
            "methods": [],
            "bases": []
        }
        
        # Extract base classes
        for base in node.bases:
            if isinstance(base, ast.Name):
                class_info["bases"].append(base.id)
        
        # Extract methods
        for body_item in node.body:
            if isinstance(body_item, ast.FunctionDef):
                method_info = self.extract_python_function_info(body_item)
                class_info["methods"].append(method_info)
        
        return class_info

    def extract_flask_routes(self, node: ast.FunctionDef, decorator_prefix: str = "") -> List[Dict[str, Any]]:
        """Extract Flask route information from a function with route decorators"""
        routes = []
        for decorator in node.decorator_list:
            route_info = {}
            
            # Extract @app.route or @blueprint.route decorators
            if isinstance(decorator, ast.Call) and hasattr(decorator.func, 'attr') and decorator.func.attr == 'route':
                if isinstance(decorator.func.value, ast.Name):
                    blueprint_name = decorator.func.value.id
                    route_info["blueprint"] = blueprint_name
                
                # Extract route path
                if decorator.args:
                    route_info["path"] = decorator.prefix + decorator.args[0].value if decorator_prefix else decorator.args[0].value
                
                # Extract HTTP methods
                for keyword in decorator.keywords:
                    if keyword.arg == 'methods' and isinstance(keyword.value, ast.List):
                        methods = []
                        for elt in keyword.value.elts:
                            if isinstance(elt, ast.Constant):
                                methods.append(elt.value)
                        route_info["methods"] = methods
                
                if route_info:
                    route_info["function"] = node.name
                    route_info["docstring"] = ast.get_docstring(node) or ""
                    routes.append(route_info)
        
        return routes

    def extract_blueprint_registration(self, content: str) -> List[Dict[str, Any]]:
        """Extract Blueprint registration from content using regex"""
        registrations = []
        
        # Pattern for app.register_blueprint(blueprint_name, url_prefix='/path')
        pattern = r'app\.register_blueprint\(([^,)]+)(?:,\s*(?:url_prefix=[\'"](.*?)[\'"]))?'
        matches = re.findall(pattern, content)
        
        for blueprint_name, url_prefix in matches:
            registrations.append({
                "blueprint": blueprint_name.strip(),
                "url_prefix": url_prefix if url_prefix else "/"
            })
        
        return registrations

    def extract_argparse_arguments(self, content: str) -> List[Dict[str, Any]]:
        """Extract command line arguments defined with argparse"""
        arguments = []
        
        # Pattern for parser.add_argument('--name', ...)
        pattern = r'parser\.add_argument\([\'"](-{1,2}[^\'"]+)[\'"](?:,\s*(?:action=[\'"]([^\'"]+)[\'"]|help=[\'"]([^\'"]+)[\'"]|type=([^,)]+)|default=([^,)]+)))*\)'
        matches = re.findall(pattern, content)
        
        for match in matches:
            arg_name, action, help_text, arg_type, default = match
            arguments.append({
                "name": arg_name,
                "action": action,
                "help": help_text,
                "type": arg_type,
                "default": default
            })
        
        return arguments

    def extract_jinja_template_structure(self, file_path: str) -> Dict[str, Any]:
        """Extract structure from a Jinja template"""
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        template_info = {
            "extends": None,
            "blocks": [],
            "includes": []
        }
        
        # Extract template inheritance
        extends_match = re.search(r'{%\s*extends\s+[\'"]([^\'"]+)[\'"]', content)
        if extends_match:
            template_info["extends"] = extends_match.group(1)
        
        # Extract blocks
        block_matches = re.findall(r'{%\s*block\s+([^\s%]+)[^%]*%}', content)
        template_info["blocks"] = list(set(block_matches))
        
        # Extract includes
        include_matches = re.findall(r'{%\s*include\s+[\'"]([^\'"]+)[\'"]', content)
        template_info["includes"] = list(set(include_matches))
        
        return template_info

    def process_python_file(self, file_path: str) -> Dict[str, Any]:
        """Process a Python file to extract its structure"""
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        # Get module name from file path
        rel_path = os.path.relpath(file_path)
        module_name = os.path.splitext(rel_path)[0].replace('/', '.')
        
        file_info = {
            "functions": [],
            "classes": [],
            "flask_routes": [],
            "imports": [],
            "blueprint_registrations": [],
            "function_calls": [],
            "api_endpoints": [],
            "orm_models": []
        }
        
        # Extract imports and build import graph
        import_lines = re.findall(r'^(?:from [^.]+(?:\.[^.]+)* )?import .+', content, re.MULTILINE)
        file_info["imports"] = import_lines
        self.extract_import_graph(content, module_name)
        
        # Extract command line arguments if found
        if "argparse" in content and "add_argument" in content:
            self.command_line_args.extend(self.extract_argparse_arguments(content))
        
        # Extract Blueprint registrations
        if "app.register_blueprint" in content:
            file_info["blueprint_registrations"] = self.extract_blueprint_registration(content)
        
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                # Extract functions
                if isinstance(node, ast.FunctionDef):
                    # Register function in module
                    self.module_functions[module_name].add(node.name)
                    
                    # Extract function calls
                    calls = self.extract_function_calls(node, module_name)
                    if calls:
                        file_info["function_calls"].append({
                            "function": node.name,
                            "calls": calls
                        })
                    
                    # Check if it's a route handler
                    routes = self.extract_flask_routes(node)
                    if routes:
                        file_info["flask_routes"].extend(routes)
                        self.flask_routes.extend(routes)
                        
                        # Extract API endpoint details for each route
                        for route in routes:
                            endpoint_info = self.extract_api_endpoint_details(route, rel_path)
                            file_info["api_endpoints"].append(endpoint_info)
                    else:
                        # Regular function
                        function_info = self.extract_python_function_info(node)
                        file_info["functions"].append(function_info)
                
                # Extract classes
                elif isinstance(node, ast.ClassDef):
                    class_info = self.extract_class_info(node)
                    file_info["classes"].append(class_info)
                    
                    # Check if this class might be an ORM model
                    base_classes = [base for base in node.bases if isinstance(base, ast.Name)]
                    orm_base_names = ['Model', 'Base', 'DeclarativeBase']
                    
                    is_orm_model = False
                    for base in base_classes:
                        if base.id in orm_base_names:
                            is_orm_model = True
                            break
                    
                    # Check for SQLAlchemy columns in the class body
                    if not is_orm_model:
                        for item in node.body:
                            if isinstance(item, ast.Assign) and isinstance(item.value, ast.Call):
                                if hasattr(item.value.func, 'id') and item.value.func.id == 'Column':
                                    is_orm_model = True
                                    break
                    
                    if is_orm_model:
                        orm_model = self.extract_orm_model(node)
                        if "orm_models" not in file_info:
                            file_info["orm_models"] = []
                        file_info["orm_models"].append(orm_model)
                
                # Extract variable assignments that might be blueprints
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and isinstance(node.value, ast.Call):
                            if (hasattr(node.value.func, 'id') and node.value.func.id == 'Blueprint'):
                                # Found a Blueprint definition
                                if len(node.value.args) >= 2:
                                    blueprint_name = target.id
                                    for route in self.flask_routes:
                                        if route.get('blueprint') == blueprint_name:
                                            self.blueprint_routes[blueprint_name].append(route)
        
        except SyntaxError:
            # Fallback for files with syntax errors
            print(f"Syntax error in {file_path}, using regex fallback")
            
            # Find functions with regex
            functions = re.findall(r'def\s+([^\(]+)\(([^\)]*)\)(?:\s*[-=]>\s*[^\:]+)?:', content)
            for func_name, params in functions:
                file_info["functions"].append({
                    "name": func_name.strip(),
                    "params": [{"name": p.strip()} for p in params.split(',') if p.strip()],
                    "docstring": "Extracted via regex. Docstring unavailable."
                })
            
            # Find classes with regex
            classes = re.findall(r'class\s+([^\(:]+)(?:\(([^\)]+)\))?:', content)
            for class_name, bases in classes:
                class_info = {
                    "name": class_name.strip(),
                    "bases": [b.strip() for b in bases.split(',') if b.strip()],
                    "methods": [],
                    "docstring": "Extracted via regex. Docstring unavailable."
                }
                file_info["classes"].append(class_info)
                
        return file_info

    def process_html_template(self, file_path: str) -> Dict[str, Any]:
        """Process an HTML template file"""
        template_info = self.extract_jinja_template_structure(file_path)
        
        # Store in template hierarchy
        template_name = os.path.basename(file_path)
        self.template_hierarchy[template_name] = template_info
        
        return template_info

    def format_function_signature(self, func_info: Dict[str, Any]) -> str:
        """Format a function signature with docstring"""
        params = []
        for param in func_info["params"]:
            param_str = param["name"]
            if "type" in param:
                param_str += f": {param['type']}"
            if "default" in param:
                param_str += f" = {param['default']}"
            params.append(param_str)
        
        signature = f"def {func_info['name']}({', '.join(params)})"
        
        # Add complexity if available
        if "complexity" in func_info:
            complexity = func_info["complexity"]
            complexity_label = ""
            if complexity <= 5:
                complexity_label = "low"
            elif complexity <= 10:
                complexity_label = "medium"
            else:
                complexity_label = "high"
            
            signature += f" [complexity: {complexity} - {complexity_label}]"
        
        # Format docstring - just first line for brevity
        docstring = func_info.get("docstring", "").split('\n')[0].strip()
        if docstring:
            signature += f": \"{docstring}\""
        
        return signature

    def format_class_info(self, class_info: Dict[str, Any]) -> List[str]:
        """Format a class definition with methods"""
        lines = []
        
        # Class definition with bases
        bases = ""
        if class_info["bases"]:
            bases = f"({', '.join(class_info['bases'])})"
        
        class_def = f"class {class_info['name']}{bases}"
        
        # First line of docstring
        docstring = class_info.get("docstring", "").split('\n')[0].strip()
        if docstring:
            class_def += f": \"{docstring}\""
        
        lines.append(class_def)
        
        # Add methods with indentation
        for method in class_info["methods"]:
            method_signature = self.format_function_signature(method)
            lines.append(f"    {method_signature}")
        
        return lines

    def format_flask_routes(self, routes: List[Dict[str, Any]]) -> List[str]:
        """Format Flask routes for output"""
        lines = []
        
        for route in routes:
            methods = route.get("methods", ["GET"])
            methods_str = ", ".join(methods)
            docstring = route.get("docstring", "").split('\n')[0].strip()
            
            if "blueprint" in route:
                route_str = f"@{route['blueprint']}.route('{route.get('path', '/')}', methods=[{methods_str}])"
            else:
                route_str = f"@app.route('{route.get('path', '/')}', methods=[{methods_str}])"
                
            lines.append(route_str)
            lines.append(f"def {route['function']}(): \"{docstring}\"")
        
        return lines

    def format_blueprint_registrations(self, registrations: List[Dict[str, Any]]) -> List[str]:
        """Format Blueprint registrations for output"""
        lines = []
        
        for reg in registrations:
            if reg["url_prefix"] and reg["url_prefix"] != "/":
                lines.append(f"app.register_blueprint({reg['blueprint']}, url_prefix='{reg['url_prefix']}')")
            else:
                lines.append(f"app.register_blueprint({reg['blueprint']})")
        
        return lines

    def format_template_structure(self, template_name: str, template_info: Dict[str, Any]) -> List[str]:
        """Format template structure for output"""
        lines = []
        
        if template_info["extends"]:
            lines.append(f"{{% extends '{template_info['extends']}' %}}")
        
        if template_info["blocks"]:
            for block in template_info["blocks"]:
                lines.append(f"{{% block {block} %}}...{{% endblock %}}")
        
        if template_info["includes"]:
            for include in template_info["includes"]:
                lines.append(f"{{% include '{include}' %}}")
        
        return lines

    def format_command_line_args(self) -> List[str]:
        """Format command line arguments for output"""
        lines = []
        
        for arg in self.command_line_args:
            arg_str = f"parser.add_argument('{arg['name']}'"
            
            if arg["action"]:
                arg_str += f", action='{arg['action']}'"
            
            if arg["help"]:
                arg_str += f", help='{arg['help']}'"
                
            if arg["type"]:
                arg_str += f", type={arg['type']}"
                
            if arg["default"]:
                arg_str += f", default={arg['default']}"
                
            arg_str += ")"
            lines.append(arg_str)
        
        return lines

    def process_file(self, file_path: str) -> Dict[str, Any]:
        """Process a file based on its type"""
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.py':
            return self.process_python_file(file_path)
        elif ext == '.html':
            return self.process_html_template(file_path)
        elif ext == '.sql':
            return {
                "type": "sql",
                "schema": self.extract_sql_schema(file_path)
            }
        else:
            # Just return basic info for other file types
            return {"type": ext[1:], "info": f"File summary not available for {ext} files"}

    def format_file_summary(self, file_path: str, rel_path: str, file_info: Dict[str, Any]) -> List[str]:
        """Format the summary of a file for the output"""
        ext = os.path.splitext(file_path)[1].lower()
        lines = []
        
        if ext == '.py':
            # Add imports summary
            if file_info["imports"]:
                lines.append("# Key imports:")
                lines.extend(["# " + imp for imp in file_info["imports"][:5]])
                if len(file_info["imports"]) > 5:
                    lines.append(f"# ... and {len(file_info['imports']) - 5} more imports")
                lines.append("")
            
            # Add blueprint registrations
            if file_info["blueprint_registrations"]:
                lines.append("# Blueprint registrations:")
                lines.extend(self.format_blueprint_registrations(file_info["blueprint_registrations"]))
                lines.append("")
            
            # Add Flask routes
            if file_info["flask_routes"]:
                lines.append("# Flask routes:")
                lines.extend(self.format_flask_routes(file_info["flask_routes"]))
                lines.append("")
            
            # Add functions
            if file_info["functions"]:
                lines.append("# Functions:")
                for func in file_info["functions"]:
                    lines.append(self.format_function_signature(func))
                lines.append("")
            
            # Add classes
            if file_info["classes"]:
                lines.append("# Classes:")
                for cls in file_info["classes"]:
                    lines.extend(self.format_class_info(cls))
                    lines.append("")
                    
            # Add ORM models if found
            if "orm_models" in file_info and file_info["orm_models"]:
                lines.append("# ORM Models:")
                for model in file_info["orm_models"]:
                    lines.append(f"class {model['name']} (table: {model['table_name']}):")
                    
                    if model["columns"]:
                        for column in model["columns"]:
                            col_attrs = []
                            if column.get("primary_key"):
                                col_attrs.append("primary_key")
                            if not column.get("nullable", True):
                                col_attrs.append("not null")
                            if column.get("unique"):
                                col_attrs.append("unique")
                            if column.get("foreign_key"):
                                fk = column["foreign_key"]
                                col_attrs.append(f"-> {fk['table']}.{fk['column']}")
                                
                            attrs_str = ", ".join(col_attrs)
                            if attrs_str:
                                attrs_str = f" ({attrs_str})"
                                
                            lines.append(f"    {column['name']}: {column.get('type', 'Unknown')}{attrs_str}")
                    
                    if model["relationships"]:
                        lines.append("    # Relationships:")
                        for rel in model["relationships"]:
                            rel_desc = f"{rel['name']} -> {rel['target']}"
                            if rel["back_populates"]:
                                rel_desc += f" (back_populates: {rel['back_populates']})"
                            lines.append(f"    {rel_desc}")
                    
                    lines.append("")
            
        elif ext == '.html':
            # Template structure
            template_name = os.path.basename(file_path)
            lines.append("# Template structure:")
            lines.extend(self.format_template_structure(template_name, file_info))
            lines.append("")
            
        elif ext == '.sql':
            # Format SQL schema information
            schema = file_info.get("schema", {})
            
            if "tables" in schema and schema["tables"]:
                lines.append("# Database Tables:")
                for table in schema["tables"]:
                    lines.append(f"CREATE TABLE {table['name']} (")
                    
                    for column in table["columns"]:
                        col_line = f"    {column['name']} {column['type']}"
                        if column.get("constraints"):
                            col_line += f" {column['constraints']}"
                        lines.append(col_line)
                    
                    lines.append(");")
                    lines.append("")
            
            if "indexes" in schema and schema["indexes"]:
                lines.append("# Indexes:")
                for index in schema["indexes"]:
                    lines.append(f"CREATE INDEX {index['name']} ON {index['table']} ({', '.join(index['columns'])});")
                lines.append("")
            
        else:
            # Basic info for other file types
            lines.append(f"# {file_info['type']} file - detailed extraction not supported")
            
        return lines

    def extract_function_calls(self, node: ast.FunctionDef, module_name: str) -> List[str]:
        """Extract function calls made within a function"""
        calls = []
        
        for subnode in ast.walk(node):
            if isinstance(subnode, ast.Call) and isinstance(subnode.func, ast.Name):
                calls.append(subnode.func.id)
            elif isinstance(subnode, ast.Call) and isinstance(subnode.func, ast.Attribute):
                if isinstance(subnode.func.value, ast.Name):
                    calls.append(f"{subnode.func.value.id}.{subnode.func.attr}")
        
        # Register these calls in our tracking structure
        func_id = f"{module_name}.{node.name}"
        self.function_calls[func_id].update(calls)
        
        return calls

    def extract_import_graph(self, content: str, module_name: str) -> List[str]:
        """Build a graph of import relationships"""
        imports = []
        
        # Extract regular imports
        import_matches = re.findall(r'^import\s+([^#\n]+)', content, re.MULTILINE)
        for match in import_matches:
            for module in match.split(','):
                module = module.strip()
                if ' as ' in module:
                    module = module.split(' as ')[0].strip()
                imports.append(module)
                self.import_graph[module_name].add(module)
        
        # Extract from imports
        from_matches = re.findall(r'^from\s+([^\s]+)\s+import\s+([^#\n]+)', content, re.MULTILINE)
        for base_module, submodules in from_matches:
            base_module = base_module.strip()
            for submodule in submodules.split(','):
                submodule = submodule.strip()
                if ' as ' in submodule:
                    submodule = submodule.split(' as ')[0].strip()
                full_module = f"{base_module}.{submodule}"
                imports.append(full_module)
                self.import_graph[module_name].add(full_module)
        
        return imports

    def extract_api_endpoint_details(self, route_info: Dict[str, Any], file_path: str) -> Dict[str, Any]:
        """Extract detailed API endpoint information including parameters and return types"""
        endpoint_info = {
            "path": route_info.get("path", "/"),
            "methods": route_info.get("methods", ["GET"]),
            "function": route_info.get("function", ""),
            "file_path": file_path,
            "docstring": route_info.get("docstring", ""),
            "parameters": [],
            "return_type": None
        }
        
        # Parse docstring for more information if available
        docstring = route_info.get("docstring", "")
        if docstring:
            # Extract parameters from docstring
            param_matches = re.findall(r'@param\s+(\w+)\s*:?\s*([^\n]*)', docstring)
            for param_name, param_desc in param_matches:
                endpoint_info["parameters"].append({
                    "name": param_name,
                    "description": param_desc.strip()
                })
                
            # Extract return type from docstring
            return_match = re.search(r'@return\s*:?\s*([^\n]*)', docstring)
            if return_match:
                endpoint_info["return_type"] = return_match.group(1).strip()
        
        self.api_endpoints.append(endpoint_info)
        return endpoint_info

    def extract_sql_schema(self, file_path: str) -> Dict[str, Any]:
        """Extract database schema from SQL files"""
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
            
        schema_info = {
            "tables": [],
            "indexes": [],
            "constraints": []
        }
        
        # Extract CREATE TABLE statements
        create_table_pattern = r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[`"]?(\w+)[`"]?\s*\((.*?)\);'
        table_matches = re.findall(create_table_pattern, content, re.DOTALL | re.IGNORECASE)
        
        for table_name, columns_text in table_matches:
            table_info = {
                "name": table_name,
                "columns": []
            }
            
            # Extract column definitions
            column_pattern = r'[`"]?(\w+)[`"]?\s+([^\s,]+)(?:\s+([^,]+))?'
            column_matches = re.findall(column_pattern, columns_text)
            
            for col_name, col_type, constraints in column_matches:
                column_info = {
                    "name": col_name,
                    "type": col_type,
                    "constraints": constraints.strip() if constraints else ""
                }
                
                # Extract primary key
                if "PRIMARY KEY" in constraints.upper():
                    column_info["primary_key"] = True
                
                # Extract foreign key
                fk_match = re.search(r'REFERENCES\s+[`"]?(\w+)[`"]?\s*\(\s*[`"]?(\w+)[`"]?\s*\)', constraints, re.IGNORECASE)
                if fk_match:
                    column_info["foreign_key"] = {
                        "table": fk_match.group(1),
                        "column": fk_match.group(2)
                    }
                    
                    # Add to relationships
                    self.db_relationships.append({
                        "from_table": table_name,
                        "from_column": col_name,
                        "to_table": fk_match.group(1),
                        "to_column": fk_match.group(2)
                    })
                
                table_info["columns"].append(column_info)
            
            # Add table to schema
            schema_info["tables"].append(table_info)
            self.db_tables[table_name] = table_info
            
        # Extract CREATE INDEX statements
        index_pattern = r'CREATE\s+(?:UNIQUE\s+)?INDEX\s+[`"]?(\w+)[`"]?\s+ON\s+[`"]?(\w+)[`"]?\s*\(([^)]+)\)'
        index_matches = re.findall(index_pattern, content, re.IGNORECASE)
        
        for index_name, table_name, columns in index_matches:
            index_info = {
                "name": index_name,
                "table": table_name,
                "columns": [col.strip('"` ') for col in columns.split(',')]
            }
            schema_info["indexes"].append(index_info)
            
        return schema_info

    def extract_orm_model(self, node: ast.ClassDef) -> Dict[str, Any]:
        """Extract ORM model definition from a class"""
        model_info = {
            "name": node.name,
            "table_name": None,
            "columns": [],
            "relationships": []
        }
        
        # Extract table name from __tablename__ attribute
        for item in node.body:
            if isinstance(item, ast.Assign) and len(item.targets) == 1:
                if isinstance(item.targets[0], ast.Name) and item.targets[0].id == "__tablename__":
                    if isinstance(item.value, ast.Constant):
                        model_info["table_name"] = item.value.value
        
        # If no explicit table name, use class name
        if not model_info["table_name"]:
            model_info["table_name"] = self.convert_camel_to_snake(node.name)
        
        # Extract columns and relationships
        for item in node.body:
            if isinstance(item, ast.Assign) and len(item.targets) == 1:
                if isinstance(item.targets[0], ast.Name):
                    column_name = item.targets[0].id
                    
                    # Skip special attributes
                    if column_name.startswith("__") and column_name.endswith("__"):
                        continue
                    
                    # Check if it's a Column definition
                    if isinstance(item.value, ast.Call):
                        if hasattr(item.value.func, 'id') and item.value.func.id == 'Column':
                            column_info = self.extract_column_definition(column_name, item.value)
                            model_info["columns"].append(column_info)
                            
                            # Update global schema
                            if model_info["table_name"] not in self.db_tables:
                                self.db_tables[model_info["table_name"]] = {
                                    "name": model_info["table_name"],
                                    "columns": []
                                }
                            self.db_tables[model_info["table_name"]]["columns"].append(column_info)
                            
                        # Check if it's a relationship
                        elif hasattr(item.value.func, 'id') and item.value.func.id == 'relationship':
                            rel_info = {
                                "name": column_name,
                                "target": None,
                                "back_populates": None,
                                "foreign_keys": None
                            }
                            
                            # Extract target model
                            if item.value.args:
                                if isinstance(item.value.args[0], ast.Constant):
                                    rel_info["target"] = item.value.args[0].value
                            
                            # Extract relationship options
                            for keyword in item.value.keywords:
                                if keyword.arg == 'back_populates' and isinstance(keyword.value, ast.Constant):
                                    rel_info["back_populates"] = keyword.value.value
                                elif keyword.arg == 'foreign_keys' and isinstance(keyword.value, ast.List):
                                    rel_info["foreign_keys"] = []
                                    for elt in keyword.value.elts:
                                        if isinstance(elt, ast.Attribute):
                                            if isinstance(elt.value, ast.Name):
                                                rel_info["foreign_keys"].append(f"{elt.value.id}.{elt.attr}")
                            
                            model_info["relationships"].append(rel_info)
                            
                            # Add to global relationships if we have enough info
                            if rel_info["target"] and model_info["table_name"]:
                                self.db_relationships.append({
                                    "from_table": model_info["table_name"],
                                    "relationship_name": column_name,
                                    "to_table": self.convert_camel_to_snake(rel_info["target"]),
                                    "back_populates": rel_info["back_populates"]
                                })
                                
        return model_info
        
    def extract_column_definition(self, column_name: str, node: ast.Call) -> Dict[str, Any]:
        """Extract column definition from a SQLAlchemy Column call"""
        column_info = {
            "name": column_name,
            "type": None,
            "primary_key": False,
            "nullable": True,
            "unique": False,
            "foreign_key": None
        }
        
        # Extract column type
        if node.args:
            arg0 = node.args[0]
            if isinstance(arg0, ast.Call):
                if hasattr(arg0.func, 'id'):
                    column_info["type"] = arg0.func.id
                elif hasattr(arg0.func, 'attr'):
                    column_info["type"] = arg0.func.attr
            elif isinstance(arg0, ast.Name):
                column_info["type"] = arg0.id
            elif isinstance(arg0, ast.Attribute):
                column_info["type"] = arg0.attr
        
        # Extract column constraints
        for arg in node.args[1:]:
            if isinstance(arg, ast.Call) and hasattr(arg.func, 'id') and arg.func.id == 'ForeignKey':
                if arg.args and isinstance(arg.args[0], ast.Constant):
                    fk_ref = arg.args[0].value
                    if '.' in fk_ref:
                        table_name, col_name = fk_ref.split('.')
                        column_info["foreign_key"] = {
                            "table": table_name,
                            "column": col_name
                        }
                        
                        # Add to relationships
                        table_name = self.convert_camel_to_snake(node.name) if hasattr(node, 'name') else None
                        if table_name:
                            self.db_relationships.append({
                                "from_table": table_name,
                                "from_column": column_name,
                                "to_table": table_name,
                                "to_column": col_name
                            })
        
        # Extract column options
        for keyword in node.keywords:
            if keyword.arg == 'primary_key' and isinstance(keyword.value, ast.Constant):
                column_info["primary_key"] = keyword.value.value
            elif keyword.arg == 'nullable' and isinstance(keyword.value, ast.Constant):
                column_info["nullable"] = keyword.value.value
            elif keyword.arg == 'unique' and isinstance(keyword.value, ast.Constant):
                column_info["unique"] = keyword.value.value
                
        return column_info

    def convert_camel_to_snake(self, name: str) -> str:
        """Convert CamelCase to snake_case"""
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

    def extract_context(self, root_dir: str, output_file: str) -> None:
        """Extract context from codebase and write to output file"""
        print(f"Processing project directory: {root_dir}")
        print(f"Using configuration:")
        print(f"  - Exclude directories: {self.config['exclude_dirs']}")
        print(f"  - Include directories: {self.config['include_dirs']}")
        print(f"  - Exclude files: {self.config['exclude_files']}")
        print(f"  - File extensions: {self.config['extensions']}")
        
        # Collect files
        try:
            files = self.collect_files(root_dir)
            print(f"Found {len(files)} eligible files")
            files.sort(key=lambda x: x[1])  # Sort by relative path
        except Exception as e:
            print(f"Error collecting files: {str(e)}")
            return
        
        # Process all files to gather relationships
        processed_count = 0
        excluded_count = 0
        error_count = 0
        
        for file_path, rel_path in files:
            # Skip processing if we've hit line limit
            if self.total_lines >= self.config["max_lines"]:
                print(f"Reached line limit of {self.config['max_lines']} lines. Stopping...")
                break
                
            try:
                # Process the file
                file_info = self.process_file(file_path)
                processed_count += 1
                
                # Format the file summary
                summary_lines = self.format_file_summary(file_path, rel_path, file_info)
                
                # Skip if no useful info extracted
                if not summary_lines:
                    excluded_count += 1
                    continue
                    
                # Check if adding this file would exceed our line limit
                if self.total_lines + len(summary_lines) + 3 > self.config["max_lines"]:
                    print(f"Reached target line limit ({self.config['max_lines']}). Stopping...")
                    break
                    
                # Add file info with header
                self.file_summaries.append({
                    "path": rel_path,
                    "lines": summary_lines
                })
                
                self.total_lines += len(summary_lines) + 3  # +3 for separator and file name lines
                
                # Periodically report progress
                if processed_count % 10 == 0:
                    print(f"Processed {processed_count} files, total lines: {self.total_lines}")
            
            except Exception as e:
                print(f"Error processing file {rel_path}: {str(e)}")
                error_count += 1
        
        print(f"Processing complete:")
        print(f"  - Processed: {processed_count} files")
        print(f"  - Excluded (no useful info): {excluded_count} files")
        print(f"  - Errors: {error_count} files")
        
        # Now write everything to file
        try:
            with open(output_file, 'w', encoding='utf-8') as out:
                # Write header
                out.write("# AI KNOWLEDGE BASE CODE CONTEXT\n")
                out.write(f"# Generated context from {root_dir}\n")
                out.write(f"# Total files processed: {len(self.file_summaries)}\n\n")
                
                # Write command line arguments summary if found
                if self.command_line_args:
                    out.write("\n\n" + "=" * 80 + "\n")
                    out.write("# COMMAND LINE INTERFACE\n")
                    out.write("=" * 80 + "\n\n")
                    out.write("\n".join(self.format_command_line_args()))
                    out.write("\n\n")
                
                # Write API endpoints summary if found
                if self.api_endpoints:
                    out.write("\n\n" + "=" * 80 + "\n")
                    out.write("# API ENDPOINTS\n")
                    out.write("=" * 80 + "\n\n")
                    
                    # Group endpoints by file
                    endpoints_by_file = defaultdict(list)
                    for endpoint in self.api_endpoints:
                        endpoints_by_file[endpoint["file_path"]].append(endpoint)
                    
                    for file_path, endpoints in endpoints_by_file.items():
                        out.write(f"## File: {file_path}\n\n")
                        
                        for endpoint in endpoints:
                            # Format method and path
                            methods = ", ".join(endpoint["methods"])
                            out.write(f"{methods} {endpoint['path']}\n")
                            
                            # Format function name
                            out.write(f"Handler: {endpoint['function']}\n")
                            
                            # Format docstring
                            if endpoint["docstring"]:
                                out.write(f"Description: {endpoint['docstring']}\n")
                            
                            # Format parameters
                            if endpoint["parameters"]:
                                out.write("Parameters:\n")
                                for param in endpoint["parameters"]:
                                    out.write(f"  - {param['name']}: {param['description']}\n")
                            
                            # Format return type
                            if endpoint["return_type"]:
                                out.write(f"Returns: {endpoint['return_type']}\n")
                            
                            out.write("\n")
                
                # Write blueprint routes summary
                if self.blueprint_routes:
                    out.write("\n\n" + "=" * 80 + "\n")
                    out.write("# BLUEPRINT ROUTES SUMMARY\n")
                    out.write("=" * 80 + "\n\n")
                    
                    for blueprint_name, routes in self.blueprint_routes.items():
                        out.write(f"## Blueprint: {blueprint_name}\n")
                        route_lines = self.format_flask_routes(routes)
                        out.write("\n".join(route_lines))
                        out.write("\n\n")
                
                # Write database schema summary if found
                if self.db_tables:
                    out.write("\n\n" + "=" * 80 + "\n")
                    out.write("# DATABASE SCHEMA\n")
                    out.write("=" * 80 + "\n\n")
                    
                    out.write("## Tables\n\n")
                    for table_name, table_info in self.db_tables.items():
                        out.write(f"### {table_name}\n\n")
                        
                        if "columns" in table_info:
                            out.write("| Column | Type | Constraints |\n")
                            out.write("|--------|------|-------------|\n")
                            
                            for column in table_info["columns"]:
                                constraints = []
                                if column.get("primary_key"):
                                    constraints.append("PRIMARY KEY")
                                if not column.get("nullable", True):
                                    constraints.append("NOT NULL")
                                if column.get("unique"):
                                    constraints.append("UNIQUE")
                                if column.get("foreign_key"):
                                    fk = column["foreign_key"]
                                    constraints.append(f"REFERENCES {fk['table']}({fk['column']})")
                                
                                constraints_str = ", ".join(constraints)
                                out.write(f"| {column['name']} | {column.get('type', 'Unknown')} | {constraints_str} |\n")
                            
                            out.write("\n")
                    
                    if self.db_relationships:
                        out.write("## Relationships\n\n")
                        out.write("| From Table | From Column | To Table | To Column |\n")
                        out.write("|------------|-------------|----------|----------|\n")
                        
                        # De-duplicate relationships
                        unique_rels = set()
                        for rel in self.db_relationships:
                            if "from_table" in rel and "to_table" in rel:
                                key = (rel["from_table"], rel.get("from_column", ""), rel["to_table"], rel.get("to_column", ""))
                                unique_rels.add(key)
                        
                        for from_table, from_col, to_table, to_col in sorted(unique_rels):
                            out.write(f"| {from_table} | {from_col} | {to_table} | {to_col} |\n")
                        
                        out.write("\n")
                
                # Write module dependency graph
                if self.import_graph:
                    out.write("\n\n" + "=" * 80 + "\n")
                    out.write("# MODULE DEPENDENCIES\n")
                    out.write("=" * 80 + "\n\n")
                    
                    # Find most important modules (most imported)
                    imported_count = defaultdict(int)
                    for imports in self.import_graph.values():
                        for imported in imports:
                            imported_count[imported] += 1
                    
                    important_modules = sorted(
                        [(module, count) for module, count in imported_count.items()],
                        key=lambda x: x[1], reverse=True
                    )[:20]  # Top 20 most imported modules
                    
                    out.write("## Most Important Modules\n\n")
                    out.write("| Module | Times Imported |\n")
                    out.write("|--------|---------------|\n")
                    
                    for module, count in important_modules:
                        out.write(f"| {module} | {count} |\n")
                    
                    out.write("\n## Module Import Graph\n\n")
                    
                    # Filter to just application modules (not stdlib)
                    app_modules = {
                        module for module in self.import_graph.keys()
                        if not module.startswith('__') and not any(
                            module.startswith(stdlib) 
                            for stdlib in ['os', 'sys', 're', 'json', 'time', 'datetime', 'logging']
                        )
                    }
                    
                    for module in sorted(app_modules):
                        app_imports = {
                            imported for imported in self.import_graph[module]
                            if not imported.startswith('__') and not any(
                                imported.startswith(stdlib) 
                                for stdlib in ['os', 'sys', 're', 'json', 'time', 'datetime', 'logging']
                            )
                        }
                        
                        if app_imports:
                            out.write(f"- {module}\n")
                            for imported in sorted(app_imports):
                                out.write(f"  - imports {imported}\n")
                            out.write("\n")
                
                # Write function call graph
                if self.function_calls:
                    out.write("\n\n" + "=" * 80 + "\n")
                    out.write("# FUNCTION CALLS\n")
                    out.write("=" * 80 + "\n\n")
                    
                    # Find important functions (called by many others)
                    call_counts = defaultdict(int)
                    for calls in self.function_calls.values():
                        for called in calls:
                            call_counts[called] += 1
                    
                    important_functions = sorted(
                        [(func, count) for func, count in call_counts.items()],
                        key=lambda x: x[1], reverse=True
                    )[:20]  # Top 20 most called functions
                    
                    out.write("## Most Called Functions\n\n")
                    out.write("| Function | Times Called |\n")
                    out.write("|----------|-------------|\n")
                    
                    for func, count in important_functions:
                        out.write(f"| {func} | {count} |\n")
                    
                    out.write("\n")
                
                # Write code complexity information
                if self.complex_functions:
                    out.write("\n\n" + "=" * 80 + "\n")
                    out.write("# CODE COMPLEXITY\n")
                    out.write("=" * 80 + "\n\n")
                    
                    # Sort by complexity
                    sorted_functions = sorted(
                        self.complex_functions,
                        key=lambda x: x["complexity"],
                        reverse=True
                    )
                    
                    out.write("## Most Complex Functions\n\n")
                    out.write("| Function | Complexity | File | Line |\n")
                    out.write("|----------|------------|------|------|\n")
                    
                    for func in sorted_functions[:20]:  # Top 20 most complex functions
                        file_path = func.get("source_file", "Unknown")
                        if file_path and file_path.startswith(root_dir):
                            file_path = os.path.relpath(file_path, root_dir)
                        
                        out.write(f"| {func['name']} | {func['complexity']} | {file_path} | {func.get('line_number', 'N/A')} |\n")
                    
                    # Add complexity guidelines
                    out.write("\n### Complexity Guidelines\n\n")
                    out.write("Cyclomatic complexity is a measure of the number of linearly independent paths through a program's source code:\n\n")
                    out.write("- 1-5: Low complexity - Simple, well-structured code\n")
                    out.write("- 6-10: Medium complexity - Moderately complex code, still maintainable\n")
                    out.write("- 11-20: High complexity - Complex code that may need refactoring\n")
                    out.write("- 21+: Very high complexity - Code that should be refactored\n\n")
                    
                    # Add complexity distribution
                    low_complexity = len([f for f in self.complex_functions if f["complexity"] <= 5])
                    medium_complexity = len([f for f in self.complex_functions if 5 < f["complexity"] <= 10])
                    high_complexity = len([f for f in self.complex_functions if 10 < f["complexity"] <= 20])
                    very_high_complexity = len([f for f in self.complex_functions if f["complexity"] > 20])
                    
                    out.write("### Complexity Distribution\n\n")
                    out.write(f"- Low complexity (1-5): {low_complexity} functions\n")
                    out.write(f"- Medium complexity (6-10): {medium_complexity} functions\n")
                    out.write(f"- High complexity (11-20): {high_complexity} functions\n")
                    out.write(f"- Very high complexity (21+): {very_high_complexity} functions\n\n")
                
                # Write template hierarchy if found
                if self.template_hierarchy:
                    out.write("\n\n" + "=" * 80 + "\n")
                    out.write("# TEMPLATE HIERARCHY\n")
                    out.write("=" * 80 + "\n\n")
                    
                    for template_name, template_info in self.template_hierarchy.items():
                        if template_info["extends"]:
                            out.write(f"{template_name} extends {template_info['extends']}\n")
                            if template_info["blocks"]:
                                out.write(f"  Blocks: {', '.join(template_info['blocks'])}\n")
                
                    out.write("\n\n")
                
                # Write file summaries
                for file_summary in self.file_summaries:
                    separator = "=" * 80
                    out.write(f"\n\n{separator}\n")
                    out.write(f"# FILE: {file_summary['path']}\n")
                    out.write(f"{separator}\n\n")
                    
                    for line in file_summary["lines"]:
                        out.write(f"{line}\n")
        
            print(f"Context extraction complete! Generated {self.total_lines} lines in {output_file}")
        except Exception as e:
            print(f"Error writing to output file: {str(e)}")

def main():
    """Main function to execute the script"""
    parser = argparse.ArgumentParser(description="Extract structured context from a Python/Flask codebase")
    parser.add_argument("output_file", nargs="?", default="code_context.txt", 
                      help="Output file path (default: code_context.txt)")
    parser.add_argument("--root-dir", "-d", default=os.getcwd(),
                      help="Root directory to process (default: current directory)")
    parser.add_argument("--exclude", "-e", action="append", default=[],
                      help="Additional directories to exclude (can be used multiple times)")
    parser.add_argument("--include", "-i", action="append", default=[],
                      help="Additional directories to include (can be used multiple times)")
    parser.add_argument("--max-lines", "-m", type=int, default=DEFAULT_CONFIG["max_lines"],
                      help=f"Maximum number of lines in output (default: {DEFAULT_CONFIG['max_lines']})")
    
    args = parser.parse_args()
    
    # Create a copy of the default config
    config = DEFAULT_CONFIG.copy()
    
    # Update config with command-line args
    config["exclude_dirs"].extend(args.exclude)
    config["include_dirs"].extend(args.include)
    config["max_lines"] = args.max_lines
    
    try:
        extractor = CodeContextExtractor(config)
        extractor.extract_context(args.root_dir, args.output_file)
    except KeyboardInterrupt:
        print("\nExtraction stopped by user.")
    except Exception as e:
        print(f"Error during extraction: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 