import os
import subprocess
import asyncio
import shlex
import json
from rich.prompt import Confirm
from rich.console import Console
from tavily import TavilyClient

# Import vector DB sync engine
from superagent.rag.memory_db import update_file_in_db, query_chat_memory

def truncate_output(text: str) -> str:
    """
    Limits any string to 128,000 characters for Online APIs.
    Keeps the first 20% and the last 80% to preserve context and error logs.
    """
    if len(text) <= 128000:
        return text
    
    first_part = text[:25600]
    last_part = text[-102400:]
    truncated_msg = f"\n\n... [TRUNCATED {len(text) - 128000} CHARACTERS TO PRESERVE ONLINE BUDGET] ...\n\n"
    return f"{first_part}{truncated_msg}{last_part}"


async def read_file(filepath: str) -> str:
    """
    This function safely reads a file and truncates large outputs to save LLM tokens.
    """
    try:
        normalized_path = os.path.abspath(filepath)
        if not os.path.exists(normalized_path):
            return f"Error: File does not exist at '{filepath}'."
        if os.path.isdir(normalized_path):
            return f"Error: '{filepath}' is a directory, not a file."
            
        def _read():
            ext = os.path.splitext(normalized_path)[1].lower()
            if ext in ['.pdf', '.docx', '.png', '.jpg', '.jpeg']:
                from superagent.rag.document_parser import process_document
                return process_document(normalized_path)
            with open(normalized_path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
                
        content = await asyncio.to_thread(_read)
        return truncate_output(content)
    except Exception as e:
        return f"Error reading file '{filepath}': {str(e)}"


async def read_file_chunk(filepath: str, start_line: int = 1, end_line: int = None) -> str:
    """
    Reads a specific range of line numbers from a file (1-indexed).
    Each line in the returned string is prefixed with its actual line number.
    """
    try:
        normalized_path = os.path.abspath(filepath)
        if not os.path.exists(normalized_path):
            return f"Error: File does not exist at '{filepath}'."
        if os.path.isdir(normalized_path):
            return f"Error: '{filepath}' is a directory, not a file."
            
        def _read_lines():
            with open(normalized_path, "r", encoding="utf-8", errors="replace") as f:
                return f.readlines()
                
        lines = await asyncio.to_thread(_read_lines)
            
        if end_line is None:
            end_line = len(lines)
            
        if start_line < 1 or end_line > len(lines) or start_line > end_line:
            return f"Error: Invalid line range {start_line}-{end_line} for file with {len(lines)} lines."
            
        selected_lines = lines[start_line-1:end_line]
        return "".join([f"{start_line+idx}: {line}" for idx, line in enumerate(selected_lines)])
    except Exception as e:
        return f"Error reading file chunk '{filepath}': {str(e)}"


async def write_file(filepath: str, content: str) -> str:
    """
    Writes content to a file, creating parent directories if they don't exist.
    Immediately updates the vector database index for the file.
    """
    try:
        normalized_path = os.path.abspath(filepath)
        
        # Git-Protection snapshot before modifications
        await git_checkpoint(f"Before modifying {os.path.basename(filepath)}")
        
        def _write():
            os.makedirs(os.path.dirname(normalized_path), exist_ok=True)
            with open(normalized_path, "w", encoding="utf-8") as f:
                f.write(content)
            update_file_in_db(normalized_path)
            
        await asyncio.to_thread(_write)
        return f"Successfully wrote {len(content)} characters to '{filepath}'."
    except Exception as e:
        return f"Error writing to file '{filepath}': {str(e)}"


async def run_terminal_command(command: str) -> str:
    """
    Executes a shell command on the user's terminal safely.
    Requires user confirmation before execution.
    Only allows specific commands for security.
    """
    try:
        def ask_confirm():
            return Confirm.ask(f"[bold yellow]⚠️  Execute command:[/] [bold cyan]{command}[/]")
            
        confirmed = await asyncio.to_thread(ask_confirm)
        if not confirmed:
            return "Command execution cancelled by user."
            
        parsed_command = shlex.split(command)
        if not parsed_command:
             return "Error: Empty command provided."
             
        allowlist = ['python', 'pip', 'git', 'ls', 'cat', 'pytest', 'npm', 'node', 'echo']
        if parsed_command[0] not in allowlist:
            return f"Security Violation: Command '{parsed_command[0]}' is not in the allowlist. Execution blocked."
            
        process = await asyncio.create_subprocess_exec(
            *parsed_command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=60.0)
            
            output = stdout.decode('utf-8', errors='replace')
            stderr_decoded = stderr.decode('utf-8', errors='replace')
            
            if stderr_decoded:
                output += f"\n[STDERR]: {stderr_decoded}"
                
            if process.returncode != 0:
                output += f"\n[EXIT CODE]: {process.returncode}"
                
            return truncate_output(output.strip())
        except asyncio.TimeoutError:
            process.kill()
            return "Error: Command timed out after 60 seconds."
            
    except Exception as e:
        return f"Error executing command: {str(e)}"


async def list_directory(dirpath: str = None) -> str:
    """
    Lists all files and subdirectories in the specified folder.
    """
    if dirpath is None:
        dirpath = os.getcwd()
    try:
        normalized_path = os.path.abspath(dirpath)
        if not os.path.exists(normalized_path):
            return f"Error: Directory does not exist at '{dirpath}'."
        if not os.path.isdir(normalized_path):
            return f"Error: '{dirpath}' is not a directory."
            
        def _list():
            return os.listdir(normalized_path)
            
        entries = await asyncio.to_thread(_list)
        
        output = []
        for entry in sorted(entries):
            full_path = os.path.join(normalized_path, entry)
            if os.path.isdir(full_path):
                output.append(f"📁 {entry}/")
            else:
                output.append(f"📄 {entry}")
                
        return "\n".join(output) if output else "Directory is empty."
    except Exception as e:
        return f"Error listing directory '{dirpath}': {str(e)}"


async def search_codebase(query: str) -> str:
    """
    Performs semantic vector-based code search across the entire repository.
    Includes Dependency-Aware Reranking using dynamic codebase import analysis.
    """
    try:
        from superagent.rag.memory_db import search_db, keyword_search_db
        
        def _search():
            semantic_results = search_db(query)
            lexical_results = keyword_search_db(query)
            
            # Fetch dependency graph using helper
            graph = get_workspace_dependency_graph()
            
            # Compute repository-wide dependency counts
            import_counts = {}
            for filepath, imports_list in graph.items():
                for imp in imports_list:
                    if imp not in import_counts:
                        import_counts[imp] = set()
                    import_counts[imp].add(filepath)
                    
            return semantic_results, lexical_results, import_counts
            
        semantic_results, lexical_results, import_counts = await asyncio.to_thread(_search)
        
        if not semantic_results and not lexical_results:
            return "No matching code vectors or files found. The codebase might not be indexed yet."
            
        merged_results = {}
        
        # Process Semantic Results
        if semantic_results and semantic_results.get("documents") and semantic_results.get("metadatas") and semantic_results.get("distances"):
            doc_list = semantic_results["documents"][0]
            meta_list = semantic_results["metadatas"][0] if (semantic_results["metadatas"] and semantic_results["metadatas"][0] is not None) else [{}] * len(doc_list)
            dist_list = semantic_results["distances"][0] if (semantic_results["distances"] and semantic_results["distances"][0] is not None) else [0.0] * len(doc_list)
            
            for doc, meta, dist in zip(doc_list, meta_list, dist_list):
                if not doc or not meta: continue
                filepath = meta.get("filepath", "Unknown")
                chunk_idx = meta.get("chunk_index", 0)
                key = (filepath, chunk_idx)
                
                sim_score = 1.0 - dist if dist is not None else 0.0
                merged_results[key] = {
                    "doc": doc,
                    "meta": meta,
                    "semantic_score": sim_score,
                    "lexical_score": 0.0
                }
                
        # Process Lexical Results
        if lexical_results and lexical_results.get("documents") and lexical_results.get("metadatas") and lexical_results.get("scores"):
            doc_list = lexical_results["documents"][0]
            meta_list = lexical_results["metadatas"][0] if (lexical_results["metadatas"] and lexical_results["metadatas"][0] is not None) else [{}] * len(doc_list)
            score_list = lexical_results["scores"][0] if (lexical_results["scores"] and lexical_results["scores"][0] is not None) else [0.0] * len(doc_list)
            
            max_lex = max(score_list) if score_list else 1.0
            if max_lex == 0: max_lex = 1.0
            
            for doc, meta, score in zip(doc_list, meta_list, score_list):
                if not doc or not meta: continue
                filepath = meta.get("filepath", "Unknown")
                chunk_idx = meta.get("chunk_index", 0)
                key = (filepath, chunk_idx)
                
                normalized_lex = score / max_lex
                if key in merged_results:
                    merged_results[key]["lexical_score"] = normalized_lex
                else:
                    merged_results[key] = {
                        "doc": doc,
                        "meta": meta,
                        "semantic_score": 0.0,
                        "lexical_score": normalized_lex
                    }
                    
        if not merged_results:
            return "No matching code vectors or files found. The codebase might not be indexed yet."
            
        # Calculate Hybrid Score with Dependency-Awareness Boost
        final_list = []
        for key, val in merged_results.items():
            filepath, _ = key
            
            # Extract dependency count for this file (how many other files import its module)
            dep_count = 0
            if filepath:
                module_name = os.path.splitext(os.path.basename(filepath))[0]
                # Filter out the file itself from the importers list to get only other files
                dep_count = len(import_counts.get(module_name, set()) - {filepath})
                
            val["dependency_count"] = dep_count
            # Apply dependency count boost to hybrid score
            val["hybrid_score"] = val["semantic_score"] + val["lexical_score"] + 0.15 * dep_count
            final_list.append(val)
            
        # Sort by boosted hybrid score descending
        final_list.sort(key=lambda x: x["hybrid_score"], reverse=True)
        
        # Take Top 3 uniquely combined chunks
        top_n = final_list[:3]
        
        output_parts = []
        for idx, item in enumerate(top_n):
            doc = item["doc"].strip()
            
            # Token optimization: Inline truncation
            if len(doc) > 1200:
                doc = doc[:600] + "\n\n... [TRUNCATED FOR CONTEXT LIMITS] ...\n\n" + doc[-600:]
 
            meta = item["meta"]
            filepath = meta.get("filepath", "Unknown")
            chunk_idx = meta.get("chunk_index", 0)
            score = item["hybrid_score"]
            dep_count = item.get("dependency_count", 0)
            
            # Parse imports to display in output metadata
            imports_raw = meta.get("imports", "[]")
            if isinstance(imports_raw, str):
                try:
                    imports_list = json.loads(imports_raw)
                except Exception:
                    imports_list = []
            elif isinstance(imports_raw, list):
                imports_list = imports_raw
            else:
                imports_list = []
                
            imports_str = ", ".join(imports_list) if imports_list else "None"
            
            output_parts.append(
                f"=== Relevance Match #{idx+1} (Score: {score:.4f}, Dependency Count: {dep_count}) ===\n"
                f"File: {filepath} (Chunk #{chunk_idx})\n"
                f"Imports: {imports_str}\n"
                f"----------------------------------------\n"
                f"{doc}\n"
                f"========================================\n"
            )
            
        if not output_parts:
            return "No matching code vectors or files found."
            
        # Token optimization: Wrapper safety net
        return truncate_output("\n".join(output_parts))
    except Exception as e:
        return f"Error searching codebase vector space: {str(e)}"


async def get_project_tree(dirpath: str = None, max_depth: int = 3) -> str:
    """
    Generates a visual directory tree structure.
    Ignores heavy/junk folders like .git, venv, and node_modules.
    """
    if dirpath is None:
        dirpath = os.getcwd()
    try:
        normalized_path = os.path.abspath(dirpath)
        if not os.path.exists(normalized_path):
            return f"Error: Directory '{dirpath}' does not exist."
            
        tree_str = "Project Structure:\n"
        for root, dirs, files in os.walk(normalized_path):
            # Exclude hidden directories (like .git, .chroma_db)
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            level = root.replace(normalized_path, '').count(os.sep)
            indent = ' ' * 4 * (level)
            tree_str += f"{indent}{os.path.basename(root)}/\n"
            subindent = ' ' * 4 * (level + 1)
            for f in files:
                if not f.startswith('.'):
                    tree_str += f"{subindent}{f}\n"
        return truncate_output(tree_str)
    except Exception as e:
        return f"Error generating project tree: {str(e)}"


async def replace_in_file(filepath: str, target_text: str, replacement_text: str) -> str:
    """
    Surgically replaces specific text in a file without overwriting the whole file.
    Great for small bug fixes in large files.
    """
    try:
        normalized_path = os.path.abspath(filepath)
        if not os.path.exists(normalized_path):
            return f"Error: File '{filepath}' not found."
            
        # Git-Protection snapshot before modifications
        await git_checkpoint(f"Before modifying {os.path.basename(filepath)}")
        
        def _read():
            with open(normalized_path, 'r', encoding='utf-8') as f:
                return f.read()
                
        content = await asyncio.to_thread(_read)
            
        if target_text not in content:
            return "Error: target_text not found in the file. Ensure you have the exact string match, including indentation and line breaks."
            
        updated_content = content.replace(target_text, replacement_text, 1)
        
        def _write():
            with open(normalized_path, 'w', encoding='utf-8') as f:
                f.write(updated_content)
            # Background RAG Sync
            try:
                from superagent.rag.memory_db import update_file_in_db
                update_file_in_db(normalized_path)
            except Exception:
                pass
                
        await asyncio.to_thread(_write)
        return f"Successfully replaced text in '{filepath}' and synced DB."
    except Exception as e:
        return f"Error replacing text: {str(e)}"


async def web_search(query: str, max_results: int = 3) -> str:
    """
    Performs a live web search using Tavily API (Optimized for AI Agents).
    Finds documentation, bug fixes, or syntax without getting rate-limited.
    """
    try:
        tavily_key = os.getenv("TAVILY_API_KEY")
        if not tavily_key:
            return "Error: TAVILY_API_KEY is missing in .env file. Please add it."
            
        def _search():
            client = TavilyClient(api_key=tavily_key)
            return client.search(query=query, max_results=max_results)
            
        response = await asyncio.to_thread(_search)
        
        if not response.get("results"):
            return f"No search results found for query: {query}"
            
        output = f"Web Search Results for '{query}':\n\n"
        for i, res in enumerate(response["results"]):
            output += f"--- Result {i+1} ---\n"
            output += f"Title: {res.get('title', 'N/A')}\n"
            output += f"Link: {res.get('url', 'N/A')}\n"
            output += f"Content: {res.get('content', 'N/A')}\n\n"
            
        return truncate_output(output)
        
    except Exception as e:
        return f"Error performing web search: {str(e)}"


def get_workspace_dependency_graph() -> dict:
    """
    Helper to fetch all metadatas in collection and return a mapping of { filepath: imports_list }
    """
    from superagent.rag.memory_db import collection
    graph = {}
    if collection:
        try:
            all_data = collection.get(include=["metadatas"])
            metadatas_all = all_data.get("metadatas", []) if all_data else []
            for meta in metadatas_all:
                if not meta: continue
                filepath = meta.get("filepath")
                if not filepath: continue
                
                imports_raw = meta.get("imports", "[]")
                if isinstance(imports_raw, str):
                    try:
                        imports_list = json.loads(imports_raw)
                    except Exception:
                        imports_list = []
                elif isinstance(imports_raw, list):
                    imports_list = imports_raw
                else:
                    imports_list = []
                if filepath not in graph:
                    graph[filepath] = set()
                graph[filepath].update(imports_list)
        except Exception:
            pass
    return {k: sorted(list(v)) for k, v in graph.items()}


async def get_dependency_graph() -> str:
    """
    Returns the repository's file dependency and import graph as a JSON string.
    """
    graph = await asyncio.to_thread(get_workspace_dependency_graph)
    return json.dumps(graph, indent=2)


async def git_checkpoint(message: str) -> str:
    """
    Creates a temporary git checkpoint (commit) of the current workspace state.
    If the project is not a git repository, it initializes one automatically.
    """
    try:
        def _run_git():
            # Check if .git directory exists
            if not os.path.exists(os.path.join(os.getcwd(), ".git")):
                subprocess.run(["git", "init"], capture_output=True, check=True)
                subprocess.run(["git", "config", "user.email", "agent@superagent.ai"], capture_output=True, check=True)
                subprocess.run(["git", "config", "user.name", "SuperAgent"], capture_output=True, check=True)
                subprocess.run(["git", "add", "."], capture_output=True, check=True)
                subprocess.run(["git", "commit", "-m", "Initial commit"], capture_output=True, check=True)
            
            # Create a checkpoint commit
            subprocess.run(["git", "add", "."], capture_output=True, check=True)
            res = subprocess.run(["git", "commit", "-m", f"Checkpoint: {message}", "--allow-empty"], capture_output=True, check=True)
            return res.stdout.decode('utf-8', errors='replace').strip()
            
        res_str = await asyncio.to_thread(_run_git)
        return f"Git Checkpoint created successfully: {message}"
    except Exception as e:
        return f"Error creating git checkpoint: {str(e)}"


async def git_revert() -> str:
    """
    Rolls back the workspace to the last working git checkpoint (HEAD~1).
    """
    try:
        def _run_revert():
            if not os.path.exists(os.path.join(os.getcwd(), ".git")):
                return "Error: No git repository exists. Cannot rollback."
            res = subprocess.run(["git", "reset", "--hard", "HEAD~1"], capture_output=True, check=True)
            return res.stdout.decode('utf-8', errors='replace').strip()
            
        res_str = await asyncio.to_thread(_run_revert)
        return f"Successfully rolled back to the last working checkpoint: {res_str}"
    except Exception as e:
        return f"Error executing git rollback: {str(e)}"


async def run_tests() -> str:
    """
    Detects python test framework (pytest, unittest) and executes the test suite.
    """
    try:
        # Detect the framework
        venv_pytest = os.path.join(os.getcwd(), ".venv", "bin", "pytest")
        if os.path.exists(venv_pytest):
            cmd = [venv_pytest]
        elif os.path.exists(os.path.join(os.getcwd(), ".venv", "Scripts", "pytest.exe")):
            cmd = [os.path.join(os.getcwd(), ".venv", "Scripts", "pytest.exe")]
        else:
            def _check_pytest():
                import shutil
                return shutil.which("pytest") is not None
            
            if await asyncio.to_thread(_check_pytest):
                cmd = ["pytest"]
            else:
                venv_python = os.path.join(os.getcwd(), ".venv", "bin", "python3")
                if not os.path.exists(venv_python):
                    venv_python = os.path.join(os.getcwd(), ".venv", "bin", "python")
                if not os.path.exists(venv_python):
                    venv_python = "python3"
                cmd = [venv_python, "-m", "unittest", "discover"]
                
        # Run command via asyncio subprocess
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        stdout_str = stdout.decode('utf-8', errors='replace').strip()
        stderr_str = stderr.decode('utf-8', errors='replace').strip()
        
        output = f"Test suite output:\n{stdout_str}"
        if stderr_str:
            output += f"\nTest suite errors:\n{stderr_str}"
            
        if process.returncode != 0:
            return f"Error: Tests failed with exit code {process.returncode}.\n\nOutput:\n{output}"
            
        return f"Tests passed successfully!\n\nOutput:\n{output}"
    except Exception as e:
        return f"Error running tests: {str(e)}"


async def async_query_chat_memory(**kwargs) -> str:
    """
    Asynchronous wrapper for the synchronous query_chat_memory tool.
    """
    return await asyncio.to_thread(query_chat_memory, **kwargs)


# Dictionary mapping tool names to functions
AVAILABLE_TOOLS = {
    "read_file": read_file,
    "read_file_chunk": read_file_chunk,
    "write_file": write_file,
    "run_terminal_command": run_terminal_command,
    "list_directory": list_directory,
    "search_codebase": search_codebase,
    "get_project_tree": get_project_tree,
    "replace_in_file": replace_in_file,
    "web_search": web_search,
    "git_checkpoint": git_checkpoint,
    "git_revert": git_revert,
    "run_tests": run_tests,
    "get_dependency_graph": get_dependency_graph,
    "query_chat_memory": async_query_chat_memory,
}

