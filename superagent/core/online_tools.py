# Strictly defined tool schema compatible with OpenAI API function calling format
ONLINE_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Reads a file from disk. CRITICAL: NEVER use this for .pdf, .docx, .png, or image files—these are already indexed, so ALWAYS use `search_codebase` to query their contents. NEVER use this to read large source code files blindly. For large files, use `search_codebase` to find target line numbers, then use `read_file_chunk`.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "The relative or absolute file path to read from."
                    }
                },
                "required": ["filepath"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file_chunk",
            "description": "Reads a specific range of line numbers from a file (1-indexed) and prefixes each line with its actual line number. Helps read specific functions or classes efficiently. CRITICAL: This reads code LINES (1-indexed), NOT PDF PAGES. NEVER use this tool if the user asks to read a 'page' of a document. Use search_codebase or read_file instead.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "The relative or absolute file path to read."
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "The line number (1-indexed) to start reading from."
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "The line number to end reading at."
                    }
                },
                "required": ["filepath", "start_line", "end_line"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Creates a file or overwrites an existing one with new content. Automatically creates parent directories if they don't exist. Immediately indexes written content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "The relative or absolute file path to write to."
                    },
                    "content": {
                        "type": "string",
                        "description": "The complete and exact text content to write to the file."
                    }
                },
                "required": ["filepath", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_terminal_command",
            "description": "Executes a shell command on the user's terminal with a 60s timeout. Use only when necessary. Always prompts user for permission first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The precise terminal command to run."
                    }
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "Lists all files and subdirectories in the specified folder (excludes hidden files by default except .env). Useful for viewing files structure.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dirpath": {
                        "type": "string",
                        "description": "The directory path to list. Defaults to the current directory '.' if not provided."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_codebase",
            "description": "Performs semantic vector-based code search across the entire repository. Use this to find functions, classes, variables, or logic blocks when you do not know the exact file path or want to find cross-referenced code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The natural language query describing the function, logic, or code block you want to find."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_project_tree",
            "description": "Generates a visual directory tree structure of the repository. Ignores heavy/junk folders like .git, venv, and node_modules to preserve context space.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dirpath": {
                        "type": "string",
                        "description": "The directory path to generate the tree for. Defaults to current directory '.' if not provided."
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "The maximum recursion depth for scanning directories. Defaults to 3."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "replace_in_file",
            "description": "Surgically replaces a specific block of target_text in a file with replacement_text. Excellent for small bug fixes in large files without overwriting the whole file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "The relative or absolute file path to edit."
                    },
                    "target_text": {
                        "type": "string",
                        "description": "The exact string block to replace in the file (must match exactly, including spaces and indentation)."
                    },
                    "replacement_text": {
                        "type": "string",
                        "description": "The new string block to replace target_text with."
                    }
                },
                "required": ["filepath", "target_text", "replacement_text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Performs a live web search using the Tavily API to find documentation, bug fixes, or code syntax examples.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to execute."
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "The maximum number of search results to return. Defaults to 3."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "git_checkpoint",
            "description": "Creates a temporary git checkpoint (commit) of the current workspace state. If git is not initialized, it automatically initializes one.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "A description of the checkpoint (e.g., 'Before modifying main.py')."
                    }
                },
                "required": ["message"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "git_revert",
            "description": "Rolls back the workspace to the last working git checkpoint (HEAD~1). Use this to restore clean working state if tests fail or code corrupts.",
            "parameters": {
                "type": "object"
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "Detects the python test framework (pytest or unittest) and executes the workspace test suite.",
            "parameters": {
                "type": "object"
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_dependency_graph",
            "description": "Retrieves the repository's file dependency and import graph as a JSON string mapping files to their imports.",
            "parameters": {
                "type": "object"
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_chat_memory",
            "description": "Searches past chat history. Use this explicitly when the user asks about past conversations, previous tasks, or specific timeframes (e.g., 'last 15 chats', 'what did we do 4 days ago?', 'last chats').",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "The semantic concept to search for. Leave empty for general history." },
                    "recent_days": { "type": "number", "description": "Filter memory to include events from the last X days. e.g., 1 for last day, 4 for last 4 days." },
                    "specific_day": { "type": "number", "description": "Filter memory to ONLY include events from exactly X days ago (e.g., 4 for '4 days ago'). Do not use with recent_days." },
                    "limit": { "type": "integer", "description": "Number of past interactions to retrieve." }
                }
            }
        }
    }
]
