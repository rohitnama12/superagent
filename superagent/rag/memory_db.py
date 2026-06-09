import os
import time
from datetime import datetime
import threading
import warnings

warnings.filterwarnings("ignore", message=".*pin_memory.*")
import chromadb
from chromadb.config import Settings
import re
import json
import hashlib

try:
    import tree_sitter
    HAS_TREE_SITTER = True
except ImportError:
    HAS_TREE_SITTER = False

try:
    import tree_sitter_languages
    HAS_TREE_SITTER_LANGUAGES = True
except ImportError:
    HAS_TREE_SITTER_LANGUAGES = False
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler, FileSystemEventHandler
from rich.console import Console

# Rich console for logging sync events
console = Console()

import shutil

# Global Lock to guarantee SQLite database thread safety during concurrent writes
db_lock = threading.Lock()

# Global debounce map to prevent watchdog loop re-indexing cycles
last_processed_time = {}
_active_observer = None

# 1. Initialize Persistent ChromaDB Client safely
db_path = os.path.abspath(os.path.join(os.getcwd(), ".chroma_db"))
db_client = None
collection = None
chat_collection = None

try:
    db_client = chromadb.PersistentClient(path=db_path, settings=Settings(anonymized_telemetry=False, allow_reset=True))
    collection = db_client.get_or_create_collection(
        name="codebase_rag",
        metadata={"hnsw:space": "cosine"}
    )
    chat_collection = db_client.get_or_create_collection(
        name="chat_memory_rag",
        metadata={"hnsw:space": "cosine"}
    )
except BaseException as e:
    console.print(f"[bold yellow]⚠️ ChromaDB Corruption detected, recreating database...[/bold yellow]")
    try:
        try:
            chromadb.api.client.SharedSystemClient.clear_system_cache()
        except BaseException:
            pass
        shutil.rmtree(db_path, ignore_errors=True)
        db_client = chromadb.PersistentClient(path=db_path, settings=Settings(anonymized_telemetry=False, allow_reset=True))
        collection = db_client.get_or_create_collection(
            name="codebase_rag",
            metadata={"hnsw:space": "cosine"}
        )
        chat_collection = db_client.get_or_create_collection(
            name="chat_memory_rag",
            metadata={"hnsw:space": "cosine"}
        )
    except BaseException as inner_e:
        console.print(f"[bold red]❌ Failed to initialize ChromaDB: {str(inner_e)}[/bold red]")

# 2. Local free embeddings using Hugging Face's all-MiniLM-L6-v2 (Lazy Loaded)
from sentence_transformers import CrossEncoder

def _is_network_available() -> bool:
    """DNS-resolution-level network check: tests if huggingface.co can actually be resolved.
    A raw TCP check to 8.8.8.8:53 returns True even when DNS is broken, so we use
    getaddrinfo which performs a real DNS lookup and fails the same way HF Hub would."""
    import socket
    try:
        socket.setdefaulttimeout(2)
        socket.getaddrinfo("huggingface.co", 443, socket.AF_INET, socket.SOCK_STREAM)
        return True
    except Exception:
        return False

def _ensure_offline_env():
    """Preemptively sets HF env vars to force strict use of local cache weights."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

EMBEDDING_MODEL = None
if not _is_network_available():
    _ensure_offline_env()
try:
    if not _is_network_available():
        reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', local_files_only=True)
    else:
        reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
except Exception as e:
    console.print(f"[bold red]⚠️ Reranker Initialization Error:[/] {str(e)}")
    reranker = None

def get_embedding_model():
    global EMBEDDING_MODEL
    if EMBEDDING_MODEL is None:
        # Preemptive offline guard: detect network state BEFORE touching the HF client.
        # The retry-after-failure approach is broken because HF's internal httpx client
        # closes permanently on first network failure and cannot be reopened in-process.
        if not _is_network_available():
            _ensure_offline_env()
        try:
            print("🧠 First-time token match fail or fresh file detected. Loading Local Embedding Model weights...")
            from langchain_huggingface import HuggingFaceEmbeddings
            import torch
            device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
            model_kwargs = {'device': device}
            if not _is_network_available():
                model_kwargs['local_files_only'] = True
            import os
            script_dir = os.path.dirname(os.path.abspath(__file__))
            models_dir = os.path.join(os.path.dirname(script_dir), "models")
            os.makedirs(models_dir, exist_ok=True)
            
            EMBEDDING_MODEL = HuggingFaceEmbeddings(
                model_name="BAAI/bge-small-en-v1.5",
                model_kwargs=model_kwargs,
                encode_kwargs={'normalize_embeddings': True},
                cache_folder=models_dir
            )
        except Exception as e:
            console.print(f"[bold yellow]⚠️ Network timeout or failure detected:[/] {str(e)}. Switching to offline mode...")
            import os
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            try:
                import os
                script_dir = os.path.dirname(os.path.abspath(__file__))
                models_dir = os.path.join(os.path.dirname(script_dir), "models")
                os.makedirs(models_dir, exist_ok=True)
                
                model_kwargs = {'device': device, 'local_files_only': True}
                EMBEDDING_MODEL = HuggingFaceEmbeddings(
                    model_name="BAAI/bge-small-en-v1.5",
                    model_kwargs=model_kwargs,
                    encode_kwargs={'normalize_embeddings': True},
                    cache_folder=models_dir
                )
            except Exception as inner_e:
                console.print(f"[bold red]⚠️ RAG Initialization Error:[/] Failed to load HuggingFaceEmbeddings offline: {str(inner_e)}")
                return None
    return EMBEDDING_MODEL

def rerank_results(query: str, chroma_results: dict):
    if not reranker or not chroma_results or not chroma_results.get("documents") or not chroma_results["documents"][0]:
        return chroma_results
        
    docs = chroma_results["documents"][0]
    metas = chroma_results["metadatas"][0] if chroma_results.get("metadatas") and chroma_results["metadatas"][0] else [{}] * len(docs)
    dists = chroma_results["distances"][0] if chroma_results.get("distances") and chroma_results["distances"][0] else [0.0] * len(docs)
    
    pairs = [[query, doc] for doc in docs]
    scores = reranker.predict(pairs)
    
    sorted_indices = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
    
    new_docs = [docs[i] for i in sorted_indices]
    new_metas = [metas[i] for i in sorted_indices]
    new_dists = [dists[i] for i in sorted_indices]
    
    chroma_results["documents"][0] = new_docs
    if chroma_results.get("metadatas"):
        chroma_results["metadatas"][0] = new_metas
    if chroma_results.get("distances"):
        chroma_results["distances"][0] = new_dists
        
    return chroma_results

def search_db(query: str, n_results: int = 5):
    """Searches the vector database for the query."""
    if not collection:
        return None
        
    embedder = get_embedding_model()
    if not embedder:
        return None
        
    try:
        query_vector = embedder.embed_query(query)
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=10
        )
        
        reranked_results = rerank_results(query, results)
        
        if reranked_results and reranked_results.get("documents") and reranked_results["documents"][0]:
            reranked_results["documents"][0] = reranked_results["documents"][0][:3]
            if reranked_results.get("metadatas") and reranked_results["metadatas"][0]:
                reranked_results["metadatas"][0] = reranked_results["metadatas"][0][:3]
            if reranked_results.get("distances") and reranked_results["distances"][0]:
                reranked_results["distances"][0] = reranked_results["distances"][0][:3]
                
        return reranked_results
    except Exception as e:
        console.print(f"[bold red]❌ Vector Search Error:[/] {str(e)}")
        return None


import uuid

def sanitize_tool_logs(text: str) -> str:
    """Safely removes tool-call specific JSON blocks and system execution logs from memory text."""
    if not text:
        return text
    import re
    # Remove markdown JSON blocks containing tool calls
    text = re.sub(r'```(?:json)?\s*\{.*?"name"\s*:.*?\}.*?```', '[TOOL EXECUTION HIDDEN]', text, flags=re.DOTALL)
    # Remove raw JSON tool calls
    text = re.sub(r'\{\s*"name"\s*:\s*".*?".*?\}', '[TOOL EXECUTION HIDDEN]', text, flags=re.DOTALL)
    # Remove system Nudge or tool execution logs
    text = re.sub(r'SYSTEM REPORT: The tool \'.*?\' successfully returned the following data:\n.*?(?=\n\n|\Z)', '[TOOL RESULT HIDDEN]', text, flags=re.DOTALL)
    return text

def save_chat_to_memory(user_msg: str, agent_response: str):
    """Combines query and response into a single text chunk, generates an ID and adds it to chat_collection."""
    if not chat_collection:
        return
        
    embedder = get_embedding_model()
    if not embedder:
        return
        
    try:
        agent_response = sanitize_tool_logs(agent_response)
        chunk_text = f"[Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] User Query: {user_msg}\nAgent Response: {agent_response}"
        vector = embedder.embed_documents([chunk_text])
        chunk_id = f"chat_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        with db_lock:
            chat_collection.add(
                embeddings=vector,
                documents=[chunk_text],
                metadatas=[{"timestamp": time.time(), "type": "chat_memory"}],
                ids=[chunk_id]
            )
    except Exception as e:
        console.print(f"[bold red]❌ Save Chat Memory Error:[/] {str(e)}")

def retrieve_past_context(query: str) -> str:
    """Queries chat_collection for top 5 results, reranks them using cross-encoder, and returns top 2 most relevant as formatted string."""
    if not chat_collection:
        return ""
        
    embedder = get_embedding_model()
    if not embedder:
        return ""
        
    try:
        query_vector = embedder.embed_query(query)
        results = chat_collection.query(
            query_embeddings=[query_vector],
            n_results=15
        )
        
        reranked_results = rerank_results(query, results)
        
        if not reranked_results or not reranked_results.get("documents") or not reranked_results["documents"][0]:
            return ""
            
        top_docs = reranked_results["documents"][0][:5]
        if not top_docs:
            return ""
            
        sanitized_docs = [sanitize_tool_logs(doc) for doc in top_docs]
            
        combined_text = "\n\n---\n\n".join(sanitized_docs)
        if len(combined_text) > 4000:
            truncated_docs = []
            for doc in sanitized_docs:
                if len(doc) > 500:
                    truncated_docs.append(doc[:497] + "...")
                else:
                    truncated_docs.append(doc)
            combined_text = "\n\n---\n\n".join(truncated_docs)
            if len(combined_text) > 5000:
                combined_text = combined_text[:4997] + "..."
                
        return combined_text
    except Exception as e:
        console.print(f"[bold red]❌ Retrieve Chat Memory Error:[/] {str(e)}")
        return ""


def keyword_search_db(query: str, n_results: int = 5):
    """Performs a basic lexical/keyword overlap search on all indexed chunks using TF (Term Frequency)."""
    if not collection:
        return None
        
    try:
        # Fetch all documents and metadata from ChromaDB
        all_data = collection.get(include=["documents", "metadatas"])
        documents = all_data.get("documents", [])
        metadatas = all_data.get("metadatas", [])
        
        if not documents:
            return None
            
        # Extract keywords from query
        words = re.findall(r'\b\w+\b', query.lower())
        query_words = set(w for w in words if len(w) > 2)
        if not query_words:
            query_words = set(words)
            
        if not query_words:
            return None
            
        pattern = re.compile(r'\b(' + '|'.join(map(re.escape, query_words)) + r')\b')
            
        scored_chunks = []
        for doc, meta in zip(documents, metadatas):
            matches = len(pattern.findall(doc.lower()))
            if matches > 0:
                scored_chunks.append((matches, doc, meta))
                
        # Sort by score descending
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        top_chunks = scored_chunks[:n_results]
        
        if not top_chunks:
            return None
            
        return {
            "documents": [[c[1] for c in top_chunks]],
            "metadatas": [[c[2] for c in top_chunks]],
            "scores": [[c[0] for c in top_chunks]]
        }
    except Exception as e:
        console.print(f"[bold red]❌ Lexical Search Error:[/] {str(e)}")
        return None


def extract_imports_regex(code: str, extension: str) -> list[str]:
    """
    Fallback parser using regex to find import patterns for multiple languages.
    """
    imports = set()
    ext = extension.lower().strip()
    
    if ext in ['.py']:
        for line in code.splitlines():
            line = line.strip()
            if line.startswith("import "):
                imp_line = line[7:].strip()
                for part in imp_line.split(','):
                    part = part.strip().split(' as ')[0].strip()
                    base = part.split('.')[0]
                    if base: imports.add(base)
            elif line.startswith("from "):
                from_line = line[5:].strip()
                parts = from_line.split(" import ")
                if parts:
                    base_module = parts[0].strip().split('.')[0]
                    if base_module: imports.add(base_module)
                    
    elif ext in ['.cpp', '.h', '.hpp', '.c', '.cc']:
        matches = re.findall(r'#include\s*[<"]([^>"]+)[>"]', code)
        for m in matches:
            base = os.path.basename(m).split('.')[0]
            if base: imports.add(base)
            
    elif ext in ['.cs']:
        matches = re.findall(r'using\s+([\w\.]+)\s*;', code)
        for m in matches:
            base = m.split('.')[0]
            if base: imports.add(base)
            
    elif ext in ['.java']:
        matches = re.findall(r'import\s+([\w\.\*]+)\s*;', code)
        for m in matches:
            base = m.split('.')[0]
            if base: imports.add(base)
            
    elif ext in ['.js', '.jsx', '.ts', '.tsx']:
        matches_import = re.findall(r'import\s+.*?\s+from\s*[\'"]([^\'"]+)[\'"]', code)
        matches_import_direct = re.findall(r'import\s*[\'"]([^\'"]+)[\'"]', code)
        matches_require = re.findall(r'require\s*\(\s*[\'"]([^\'"]+)[\'"]\s*\)', code)
        for m in matches_import + matches_import_direct + matches_require:
            base = m.split('/')[-1].split('.')[0]
            if base: imports.add(base)
            
    elif ext in ['.rs']:
        matches = re.findall(r'use\s+([\w\.\{\}\:\*]+)\s*;', code)
        for m in matches:
            base = m.split('::')[0].strip()
            if base: imports.add(base)
            
    elif ext in ['.go']:
        matches_direct = re.findall(r'import\s+[\'"]([^\'"]+)[\'"]', code)
        for m in matches_direct:
            base = m.split('/')[-1]
            if base: imports.add(base)
        block_match = re.search(r'import\s*\((.*?)\)', code, re.DOTALL)
        if block_match:
            block_content = block_match.group(1)
            matches_block = re.findall(r'[\'"]([^\'"]+)[\'"]', block_content)
            for m in matches_block:
                base = m.split('/')[-1]
                if base: imports.add(base)
                
    elif ext in ['.rb']:
        matches = re.findall(r'(?:require|require_relative)\s*[\'"]([^\'"]+)[\'"]', code)
        for m in matches:
            base = os.path.basename(m).split('.')[0]
            if base: imports.add(base)
            
    return sorted(list(imports))


class LanguageManager:
    """
    Modular manager mapping file extensions to tree-sitter languages
    and executing queries for structural symbol and import extraction.
    """
    def __init__(self):
        self.extension_map = {
            '.py': 'python',
            '.cpp': 'cpp',
            '.cc': 'cpp',
            '.h': 'cpp',
            '.hpp': 'cpp',
            '.cs': 'c_sharp',
            '.java': 'java',
            '.js': 'javascript',
            '.jsx': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.rs': 'rust',
            '.go': 'go',
            '.rb': 'ruby'
        }
        
    def get_language_name(self, extension: str) -> str:
        return self.extension_map.get(extension.lower().strip())
        
    def extract_imports(self, code: str, filepath: str) -> list[str]:
        """
        Structural extraction of imports/includes using tree-sitter if available.
        Falls back to robust regex matching.
        """
        ext = os.path.splitext(filepath)[1].lower()
        lang_name = self.get_language_name(ext)
        
        if HAS_TREE_SITTER and HAS_TREE_SITTER_LANGUAGES and lang_name:
            try:
                lang = tree_sitter_languages.get_language(lang_name)
                parser = tree_sitter_languages.get_parser(lang_name)
                tree = parser.parse(bytes(code, "utf8"))
                
                imports = set()
                query_str = ""
                if lang_name == 'python':
                    query_str = """
                    (import_statement) @import
                    (import_from_statement) @import
                    """
                elif lang_name == 'cpp':
                    query_str = """
                    (preproc_include) @import
                    """
                elif lang_name == 'java':
                    query_str = """
                    (import_declaration) @import
                    """
                elif lang_name in ['javascript', 'typescript']:
                    query_str = """
                    (import_statement) @import
                    (call_expression
                      function: (identifier) @func (#eq? @func "require")
                      arguments: (arguments (string) @import))
                    """
                elif lang_name == 'go':
                    query_str = """
                    (import_spec) @import
                    """
                elif lang_name == 'rust':
                    query_str = """
                    (use_declaration) @import
                    """
                elif lang_name == 'ruby':
                    query_str = """
                    (call
                      method: (identifier) @method (#match? @method "^require")
                      arguments: (argument_list (string) @import))
                    """
                elif lang_name == 'c_sharp':
                    query_str = """
                    (using_directive) @import
                    """
                    
                if query_str:
                    query = lang.query(query_str)
                    captures = query.captures(tree.root_node)
                    for node, _ in captures:
                        text = code[node.start_byte:node.end_byte].strip()
                        if lang_name == 'python':
                            if text.startswith('from'):
                                base = text.split()[1].split('.')[0]
                            else:
                                base = text.split()[1].split('.')[0]
                            imports.add(base)
                        elif lang_name == 'cpp':
                            m = re.search(r'[<"]([^>"]+)[>"]', text)
                            if m:
                                imports.add(os.path.basename(m.group(1)).split('.')[0])
                        elif lang_name == 'java':
                            parts = text.replace('import', '').replace(';', '').strip().split('.')
                            if parts: imports.add(parts[0])
                        elif lang_name in ['javascript', 'typescript']:
                            m = re.search(r'[\'"]([^\'"]+)[\'"]', text)
                            if m:
                                imports.add(m.group(1).split('/')[-1].split('.')[0])
                        elif lang_name == 'go':
                            m = re.search(r'[\'"]([^\'"]+)[\'"]', text)
                            if m:
                                imports.add(m.group(1).split('/')[-1])
                        elif lang_name == 'rust':
                            parts = text.replace('use', '').replace(';', '').strip().split('::')
                            if parts: imports.add(parts[0].strip())
                        elif lang_name == 'ruby':
                            m = re.search(r'[\'"]([^\'"]+)[\'"]', text)
                            if m:
                                imports.add(os.path.basename(m.group(1)).split('.')[0])
                        elif lang_name == 'c_sharp':
                            parts = text.replace('using', '').replace(';', '').strip().split('.')
                            if parts: imports.add(parts[0])
                            
                if imports:
                    return sorted(list(imports))
            except Exception:
                pass
                
        return extract_imports_regex(code, ext)


# Instantiate LanguageManager
language_manager = LanguageManager()
SUPPORTED_EXTENSIONS = ['.py', '.cpp', '.h', '.hpp', '.cs', '.java', '.js', '.jsx', '.ts', '.tsx', '.rs', '.go', '.rb', '.pdf', '.docx', '.png', '.jpg', '.jpeg']

def get_document_chunks(text: str, chunk_size: int = 1500, chunk_overlap: int = 200) -> list[str]:
    """
    Custom document chunker that ensures Markdown tables are kept intact.
    """
    import re
    table_pattern = re.compile(r'(?:^\|.*\|[ \t]*\n?)+', re.MULTILINE)
    
    parts = []
    last_end = 0
    for match in table_pattern.finditer(text):
        start, end = match.span()
        if start > last_end:
            parts.append(text[last_end:start])
        parts.append(match.group(0))
        last_end = end
    if last_end < len(text):
        parts.append(text[last_end:])
        
    chunks = []
    current_chunk = ""
    
    for part in parts:
        is_table = bool(table_pattern.fullmatch(part))
        if is_table:
            if len(current_chunk) + len(part) > chunk_size and current_chunk.strip():
                chunks.append(current_chunk.strip())
                current_chunk = part
            else:
                current_chunk += "\n" + part if current_chunk else part
        else:
            paragraphs = part.split('\n\n')
            for p in paragraphs:
                if len(current_chunk) + len(p) > chunk_size and current_chunk.strip():
                    chunks.append(current_chunk.strip())
                    current_chunk = p
                else:
                    current_chunk += "\n\n" + p if current_chunk else p
                    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
        
    return chunks


def get_universal_chunks(code: str, filepath: str) -> list[str]:
    """
    Universal code chunker using tree-sitter if available, or falling back to
    RecursiveCharacterTextSplitter from langchain.
    """
    ext = os.path.splitext(filepath)[1].lower()
    
    if HAS_TREE_SITTER and HAS_TREE_SITTER_LANGUAGES:
        try:
            lang_name = language_manager.get_language_name(ext)
            if lang_name:
                lang = tree_sitter_languages.get_language(lang_name)
                parser = tree_sitter_languages.get_parser(lang_name)
                tree = parser.parse(bytes(code, "utf8"))
                root_node = tree.root_node
                
                chunks = []
                module_nodes = []
                
                for child in root_node.children:
                    if child.type in ['class_definition', 'function_definition', 'method_definition', 'struct_specifier', 'class_declaration', 'function_declarator']:
                        chunk_text = code[child.start_byte:child.end_byte]
                        if chunk_text.strip():
                            chunks.append(chunk_text)
                    else:
                        module_nodes.append(code[child.start_byte:child.end_byte])
                        
                if module_nodes:
                    module_text = "\n".join(module_nodes)
                    if module_text.strip():
                        chunks.append(module_text)
                        
                if chunks:
                    return [c for c in chunks if c.strip()]
        except Exception:
            pass
            
    # Fallback: Use standard language-specific character text splitter
    try:
        lang_map = {
            '.py': Language.PYTHON,
            '.cpp': Language.CPP,
            '.cc': Language.CPP,
            '.h': Language.CPP,
            '.hpp': Language.CPP,
            '.js': Language.JS,
            '.jsx': Language.JS,
            '.ts': Language.TS,
            '.tsx': Language.TS,
            '.java': Language.JAVA,
            '.rs': Language.RUST,
            '.go': Language.GO,
            '.rb': Language.RUBY
        }
        language = lang_map.get(ext, Language.PYTHON)
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=language,
            chunk_size=1500,
            chunk_overlap=200
        )
        return splitter.split_text(code)
    except Exception:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200
        )
        return splitter.split_text(code)

def should_ignore(filepath: str) -> bool:
    """
    Checks if the filepath should be ignored by the RAG sync loop.
    Excludes hidden files, virtual environments, caches, git or vector DB folders, and agent metadata JSONs.
    """
    abs_filepath = os.path.abspath(filepath)
    workspace_dir = os.getcwd()
    try:
        rel_filepath = os.path.relpath(abs_filepath, workspace_dir)
    except ValueError:
        rel_filepath = filepath
        
    parts = rel_filepath.split(os.sep)
    if any(p in parts for p in [".git", ".venv", ".chroma_db", "__pycache__", "node_modules", "dist"]) or rel_filepath.endswith(".json") or any(p.startswith(".") for p in parts if p and p not in [".", ".."]):
        return True
    return False

def update_file_in_db(filepath: str):
    """
    Deletes existing vector chunks for a specific file and re-indexes it.
    Safely handles deletion if the file was deleted on disk.
    Guarantees thread-safe SQLite operations via db_lock.
    """
    try:
        if should_ignore(filepath):
            return

        abs_filepath = os.path.abspath(filepath)
        # Generate a relative path within the workspace for nice presentation
        workspace_dir = os.getcwd()
        rel_filepath = os.path.relpath(abs_filepath, workspace_dir)

        # 2-second debounce window check
        now = time.time()
        if now - last_processed_time.get(abs_filepath, 0) < 2.0:
            return
        last_processed_time[abs_filepath] = now

        with db_lock:
            if not collection:
                return

            # If the file was deleted or is not a supported file type, remove from index and stop
            ext = os.path.splitext(filepath)[1].lower()
            if not os.path.exists(abs_filepath) or ext not in SUPPORTED_EXTENSIONS:
                try:
                    collection.delete(where={"filepath": rel_filepath})
                except Exception:
                    pass
                return

            with open(abs_filepath, "rb") as f:
                raw_bytes = f.read()
            file_hash = hashlib.sha256(raw_bytes).hexdigest()

            # Compare with the stored file_hash in ChromaDB metadata
            try:
                existing = collection.get(where={"filepath": rel_filepath}, include=["metadatas"])
                if existing and existing.get("metadatas"):
                    stored_hash = existing["metadatas"][0].get("file_hash")
                    if stored_hash == file_hash:
                        # Content hash has not changed, skip re-indexing to avoid infinite loops and save compute
                        return
            except Exception:
                pass

            if ext in ['.pdf', '.docx', '.png', '.jpg', '.jpeg']:
                from superagent.rag.document_parser import process_document
                try:
                    code = process_document(abs_filepath)
                except Exception as e:
                    console.print(f"[bold red]❌ Document Parse Error:[/] {str(e)}")
                    return
            else:
                code = raw_bytes.decode("utf-8", errors="replace")

            if not code.strip():
                try:
                    collection.delete(where={"filepath": rel_filepath})
                except Exception:
                    pass
                return  # Empty file, nothing to index

            # Hash changed, proceed with deletion and update
            try:
                collection.delete(where={"filepath": rel_filepath})
            except Exception:
                pass

            if ext in ['.pdf', '.docx', '.png', '.jpg', '.jpeg']:
                chunks = get_document_chunks(code)
                if not chunks:
                    return
                metadatas = [
                    {
                        "filepath": rel_filepath,
                        "chunk_index": i,
                        "type": "document",
                        "file_hash": file_hash
                    }
                    for i in range(len(chunks))
                ]
            else:
                # Split text using universal chunker
                chunks = get_universal_chunks(code, filepath)
                
                if not chunks:
                    return

                imports = language_manager.extract_imports(code, filepath)
                metadatas = [
                    {
                        "filepath": rel_filepath,
                        "chunk_index": i,
                        "imports": json.dumps(imports),
                        "file_hash": file_hash
                    }
                    for i in range(len(chunks))
                ]
            ids = [f"{rel_filepath}_chunk_{i}" for i in range(len(chunks))]
            
            # Embed and insert chunks
            embedder = get_embedding_model()
            if not embedder:
                return
                
            vectors = embedder.embed_documents(chunks)
            collection.add(
                embeddings=vectors,
                documents=chunks,
                metadatas=metadatas,
                ids=ids
            )
    except Exception as e:
        console.print(f"[bold yellow]⚠️  RAG Sync Warning (update):[/] Failed to sync '{filepath}': {str(e)}")



def index_full_codebase(directory: str = None):
    """
    Recursively scans the directory for supported source files and fully builds the vector database index.
    Guarantees thread-safe SQLite operations via db_lock.
    """
    with db_lock:
        if directory is None:
            directory = os.getcwd()
        abs_dir = os.path.abspath(directory)
        
        all_chunks = []
        all_metadatas = []
        all_ids = []
        
        for root, dirs, files in os.walk(abs_dir):
            # Ignore special directories in-place to optimize traversal
            dirs[:] = [d for d in dirs if d not in [".venv", ".git", ".chroma_db", "__pycache__", "node_modules", "dist"]]
            
            for file in files:
                filepath = os.path.join(root, file)
                if should_ignore(filepath):
                    continue
                ext = os.path.splitext(file)[1].lower()
                if ext in SUPPORTED_EXTENSIONS:
                    rel_filepath = os.path.relpath(filepath, abs_dir)
                    
                    try:
                        # 1. First hash the raw file on disk to determine if we can skip expensive parsing
                        with open(filepath, "rb") as f:
                            raw_bytes = f.read()
                        file_hash = hashlib.sha256(raw_bytes).hexdigest()

                        # 2. Check if the file's hash in db is already matching
                        skip_parsing = False
                        try:
                            existing = collection.get(where={"filepath": rel_filepath}, include=["metadatas"])
                            if existing and existing.get("metadatas"):
                                stored_hash = existing["metadatas"][0].get("file_hash")
                                if stored_hash == file_hash:
                                    # Already indexed and matching!
                                    console.print(f"[Storage] '{file}' is up to date. Skipping parsing.")
                                    skip_parsing = True
                        except Exception:
                            pass
                            
                        if skip_parsing:
                            continue

                        # 3. Only parse the file if it has changed
                        if ext in ['.pdf', '.docx', '.png', '.jpg', '.jpeg']:
                            from superagent.rag.document_parser import process_document
                            code = process_document(filepath)
                        else:
                            # For regular text files, we can just decode the raw bytes
                            code = raw_bytes.decode("utf-8", errors="replace")
                            
                        if not code.strip():
                            continue

                        if ext in ['.pdf', '.docx', '.png', '.jpg', '.jpeg']:
                            chunks = get_document_chunks(code)
                        else:
                            chunks = get_universal_chunks(code, filepath)
                            
                        if not chunks:
                            continue
                            
                        # Pre-emptively delete existing index for this file to prevent duplication
                        try:
                            collection.delete(where={"filepath": rel_filepath})
                        except Exception:
                            pass
                            
                        if ext in ['.pdf', '.docx', '.png', '.jpg', '.jpeg']:
                            for i, chunk in enumerate(chunks):
                                all_chunks.append(chunk)
                                all_metadatas.append({
                                    "filepath": rel_filepath,
                                    "chunk_index": i,
                                    "type": "document",
                                    "file_hash": file_hash
                                })
                                all_ids.append(f"{rel_filepath}_chunk_{i}")
                        else:
                            imports = language_manager.extract_imports(code, filepath)
                            for i, chunk in enumerate(chunks):
                                all_chunks.append(chunk)
                                all_metadatas.append({
                                    "filepath": rel_filepath,
                                    "chunk_index": i,
                                    "imports": json.dumps(imports),
                                    "file_hash": file_hash
                                })
                                all_ids.append(f"{rel_filepath}_chunk_{i}")
                            
                    except Exception as e:
                        console.print(f"[bold red]⚠️  RAG Index Warning:[/] Skip indexing '{rel_filepath}': {str(e)}")

        if all_chunks and collection:
            embedder = get_embedding_model()
            if embedder:
                try:
                    vectors = embedder.embed_documents(all_chunks)
                    collection.add(
                        embeddings=vectors,
                        documents=all_chunks,
                        metadatas=all_metadatas,
                        ids=all_ids
                    )
                except Exception as e:
                    console.print(f"[bold red]❌ RAG Indexing Error:[/] Failed to save vector index: {str(e)}")


# 3. Watchdog Handler to observe files in real-time
class PyCodeFileWatcherHandler(PatternMatchingEventHandler):
    def __init__(self):
        patterns = [f"*{ext}" for ext in SUPPORTED_EXTENSIONS]
        ignore_patterns = ["*/.chroma_db/*", "*.json", "*/.git/*", "*/.venv/*", "*/node_modules/*", "*/dist/*"]
        super().__init__(
            patterns=patterns,
            ignore_patterns=ignore_patterns,
            ignore_directories=True,
            case_sensitive=False
        )

    def on_modified(self, event):
        if event.is_directory:
            return
        if should_ignore(event.src_path):
            return
        rel = os.path.relpath(event.src_path, os.getcwd())
        # Print a neat micro-log showing instant background sync
        console.print(f"[dim white]🔄 [RAG Sync] File modified background sync: {rel}[/]")
        update_file_in_db(event.src_path)

    def on_created(self, event):
        if event.is_directory:
            return
        if should_ignore(event.src_path):
            return
        rel = os.path.relpath(event.src_path, os.getcwd())
        console.print(f"[dim white]➕ [RAG Sync] File created background sync: {rel}[/]")
        update_file_in_db(event.src_path)


def start_watcher(directory: str = None):
    """
    Starts the watchdog folder observer to sync Python file changes instantly.
    Runs concurrently in background threads managed by watchdog.
    Forcefully stops any existing active observer to prevent duplicate or stuck watch loops.
    """
    global _active_observer
    if directory is None:
        directory = os.getcwd()
    try:
        # Interrupt/stop any currently running observer to clean up stuck processes
        if _active_observer is not None:
            try:
                _active_observer.stop()
                _active_observer.join(timeout=1.0)
            except Exception:
                pass
            _active_observer = None

        event_handler = PyCodeFileWatcherHandler()
        observer = Observer()
        observer.schedule(event_handler, path=os.path.abspath(directory), recursive=True)
        observer.start()
        _active_observer = observer
        return observer
    except Exception as e:
        console.print(f"[bold red]❌ Failed to start Watchdog Observer:[/] {str(e)}")
        return None

def query_chat_memory(query: str = "", recent_days: float = None, specific_day: float = None, limit: int = 5) -> str:
    """Advanced RAG tool to fetch specific past conversations based on time filters or semantic matches."""
    if not chat_collection: return "Error: Memory DB offline."
    try:
        conditions = [{"type": {"$eq": "chat_memory"}}]
        
        now = time.time()
        if specific_day is not None:
            start_time = now - ((specific_day + 0.5) * 86400)
            end_time = now - ((specific_day - 0.5) * 86400)
            conditions.append({"timestamp": {"$gte": start_time}})
            conditions.append({"timestamp": {"$lte": end_time}})
        elif recent_days is not None:
            cutoff_time = now - (recent_days * 86400)
            conditions.append({"timestamp": {"$gte": cutoff_time}})
            
        where_clause = {"$and": conditions} if len(conditions) > 1 else conditions[0]
            
        if query and query.strip():
            embedder = get_embedding_model()
            q_vec = embedder.embed_query(query)
            res = chat_collection.query(query_embeddings=[q_vec], n_results=limit*3, where=where_clause)
            res = rerank_results(query, res)
            
            docs = res.get("documents", [[]])[0]
            metas = res.get("metadatas", [[]])[0]
            
            if not docs: return "No matching history found."
            
            docs_with_time = list(zip(docs, metas))
            docs_with_time.sort(key=lambda x: x[1].get("timestamp", 0), reverse=True)
            
            top_docs = [d[0] for d in docs_with_time[:limit]]
            return "\n\n---\n\n".join(top_docs)
        else:
            res = chat_collection.get(where=where_clause, include=["documents", "metadatas"])
            if not res or not res.get("documents"): return "No recent history."
            docs_with_time = list(zip(res["documents"], res["metadatas"]))
            docs_with_time.sort(key=lambda x: x[1].get("timestamp", 0), reverse=True)
            top_docs = [d[0] for d in docs_with_time[:limit]]
            return "\n\n---\n\n".join(top_docs)
    except Exception as e:
        return f"Memory query failed: {str(e)}"

def debug_memory_count():
    if not chat_collection: return 0
    return chat_collection.count()

def debug_memory():
    if not chat_collection: return "DB Offline"
    all_data = chat_collection.get(include=["metadatas", "documents"])
    count = len(all_data.get('ids', []))
    return f"Total Memory Chunks: {count}. Last 3 IDs: {all_data.get('ids', [])[-3:]}"

