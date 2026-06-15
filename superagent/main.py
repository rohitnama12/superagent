import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import multiprocessing
import re
import requests
import json
from dotenv import load_dotenv
import asyncio
import httpx
from openai import AsyncOpenAI
from rich.live import Live
from rich.panel import Panel
from rich.markdown import Markdown
from rich import box
import time
from datetime import datetime
import traceback

from superagent.cli_ui import console, AGENT_COLOR, show_welcome_banner, get_user_input, show_spinner, print_system_msg, print_agent_response, print_plan, print_error, live_tool_execution_panel, print_execution_timeline, print_workspace_activity, print_session_footer
from superagent.metrics import SessionMetrics, TaskMetrics

# Import tooling engine
from superagent.core.tools import AVAILABLE_TOOLS
from superagent.core.online_tools import ONLINE_TOOLS_SCHEMA as TOOLS_SCHEMA
from superagent.rag.memory_db import index_full_codebase, start_watcher, save_chat_to_memory, retrieve_past_context

# Load environment variables
global_env = os.path.expanduser("~/.superagent.env")
if os.path.exists(global_env):
    load_dotenv(global_env)
else:
    load_dotenv()
print("API Keys Loaded Successfully")

# Check for API Key
api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    print_error("❌ Error: OPENROUTER_API_KEY is missing!\nPlease follow setup steps to configure .env.")
    input("\nPress Enter to exit...")
    sys.exit(1)

DEFAULT_MODEL = "openrouter/owl-alpha"
# DEFAULT_MODEL = "dummy/dummy"
MAX_RETRIES = 3
MAX_TOOL_ITERATIONS = 8  # Hard cap: prevents infinite tool-call loops in local/offline mode
_FAILED_MODELS = set()


# --- MODULE LEVEL MOCK CLASSES FOR TOOL/STREAM BUFFERING ---
class MockFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments
        
class MockToolCall:
    def __init__(self, id, type, function):
        self.id = id
        self.type = type
        self.function = MockFunction(**function) if isinstance(function, dict) else function

class MockMessage:
    def __init__(self, content, tool_calls_data):
        self.content = content
        self.role = "assistant"
        if tool_calls_data:
            self.tool_calls = [MockToolCall(**tc) if isinstance(tc, dict) else tc for tc in tool_calls_data]
        else:
            self.tool_calls = None

class MockChoice:
    def __init__(self, message):
        self.message = message

class MockResponse:
    def __init__(self, content, tool_calls_data):
        self.choices = [MockChoice(MockMessage(content, tool_calls_data))]

def calculate_model_score(model_dict: dict) -> float:
    """
    Dynamically evaluates how capable the model is for coding and tool calling based on its metadata.
    """
    score = 0.0
    
    model_id = str(model_dict.get("id") or "").lower()
    description = str(model_dict.get("description") or "").lower()
    
    # 1. Tool-Calling Capability Filter (Crucial)
    # Drop score if model lacks tools/function calling capabilities
    if "no tools" in description or "no function calling" in description or "does not support tool" in description:
        return 0.0
        
    # 2. Context Window & Pricing Check
    # Ensure it strictly validates that the model is indeed marked as free
    pricing = model_dict.get("pricing") or {}
    try:
        prompt_price = float(pricing.get("prompt") or -1.0)
        completion_price = float(pricing.get("completion") or -1.0)
    except (ValueError, TypeError):
        prompt_price, completion_price = -1.0, -1.0
        
    is_free = (prompt_price == 0.0 and completion_price == 0.0) or model_id.endswith(":free")
    if not is_free:
        return 0.0

    # Context Bonus for RAG
    try:
        context_length = int(model_dict.get("context_length") or 0)
    except (ValueError, TypeError):
        context_length = 0
        
    if context_length >= 32000:
        score += 20.0
        
    # 3. Brand/Quality-based Tiering System
    if any(keyword in model_id for keyword in ["google", "openai", "anthropic", "owl-alpha"]):
        score += 100.0
    elif any(keyword in model_id for keyword in ["nvidia", "liquid", "poolside", "moonshot"]):
        score += 80.0
    elif any(keyword in model_id for keyword in ["mistral", "cohere", "nous", "z.ai"]):
        score += 60.0
    elif any(keyword in model_id for keyword in ["meta", "llama"]):
        score += 40.0
    elif any(keyword in model_id for keyword in ["qwen", "coder", "deepseek"]):
        score -= 100.0
        
    # Keep the minor context scaling as a tie-breaker
    score += context_length / 100000.0
            
    return score

async def get_dynamic_free_models() -> list:
    """Dynamically fetches and scores all currently free models from OpenRouter asynchronously."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as async_client:
            response = await async_client.get("https://openrouter.ai/api/v1/models")
            if response.status_code == 200:
                models_data = response.json().get("data", [])
                scored_models = []
                
                for m in models_data:
                    # Ensure the model dictionary is valid
                    if not isinstance(m, dict):
                        continue
                        
                    # Upgraded Resiliency: Filter out music/audio generation models reported as free
                    architecture = m.get("architecture") or {}
                    output_modalities = architecture.get("output_modalities") or []
                    if "audio" in output_modalities:
                        continue
                        
                    # Check if both prompt and completion pricing are strictly "0"
                    pricing = m.get("pricing") or {}
                    if pricing.get("prompt") == "0" and pricing.get("completion") == "0":
                        score = calculate_model_score(m)
                        scored_models.append((m.get("id"), score))
                
                # Sort descending based on score
                scored_models.sort(key=lambda item: item[1], reverse=True)
                
                # Extract just the model IDs
                free_models = [item[0] for item in scored_models if item[0]]
                return free_models
    except Exception as e:
        print_system_msg(f"Warning: Could not fetch dynamic free models: {str(e)}")
    
    return []

async def run_online_engine(messages, tools, default_model="google/gemini-2.5-flash"):
    """
    Executes the LLM call with a 3-Tier Fallback Mechanism as an Async Generator supporting stream=True:
    Default -> Dynamic OpenRouter Free Models -> Local Ollama.
    """
    global DEFAULT_MODEL
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    
    # Cloud Client
    cloud_client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_key
    )

    models_to_try = [default_model]
    
    free_models = await get_dynamic_free_models()
    if default_model in free_models:
        free_models.remove(default_model)
    models_to_try.extend(free_models)
    
    # Filter out models that have already failed in this session
    models_to_try = [m for m in models_to_try if m not in _FAILED_MODELS]

    for idx, model in enumerate(models_to_try):
        try:
            if model != default_model:
                print_system_msg(f"🔄 Trying fallback model: {model}...")
                
            kwargs = {
                "model": model,
                "messages": messages,
                "stream": True,
            }
            if tools:
                kwargs["tools"] = tools

            response_stream = await cloud_client.chat.completions.create(**kwargs)
            
            tool_calls_buffer = {}
            is_tool_call = False
            
            async for chunk in response_stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                
                if getattr(delta, "tool_calls", None):
                    is_tool_call = True
                    for tc in delta.tool_calls:
                        tc_idx = tc.index
                        if tc_idx not in tool_calls_buffer:
                            tool_calls_buffer[tc_idx] = {
                                "id": tc.id,
                                "type": tc.type,
                                "name": tc.function.name if tc.function else "",
                                "arguments": ""
                            }
                        if tc.id:
                            tool_calls_buffer[tc_idx]["id"] = tc.id
                        if tc.type:
                            tool_calls_buffer[tc_idx]["type"] = tc.type
                        if tc.function:
                            if tc.function.name:
                                tool_calls_buffer[tc_idx]["name"] = tc.function.name
                            if tc.function.arguments:
                                tool_calls_buffer[tc_idx]["arguments"] += tc.function.arguments
                
                content = getattr(delta, "content", None)
                if content and not is_tool_call:
                    yield {"type": "text", "content": content}
            
            if is_tool_call:
                tc_list = []
                for idx_key, val in sorted(tool_calls_buffer.items()):
                    tc_list.append({
                        "id": val["id"],
                        "type": val["type"] or "function",
                        "function": {
                            "name": val["name"],
                            "arguments": val["arguments"]
                        }
                    })
                mock_resp = MockResponse(None, tc_list)
                DEFAULT_MODEL = model
                yield {"type": "tool_calls", "response": mock_resp}
                return
            else:
                DEFAULT_MODEL = model
                return
                
        except Exception as e:
            _FAILED_MODELS.add(model)
            if model == default_model:
                print_error(f"⚠️ Default model '{model}' failed: {str(e)}. Initiating Dynamic Fallback...")
            else:
                pass # Silent loop for dynamic free models
            continue

    print_error("❌ All Cloud Models Exhausted/Failed. Please type /offline to switch to Local Mode.")
    yield {"type": "error"}
    return

# Helper function to serialize messages to standard dicts
def message_to_dict(message):
    """
    Safely serializes an assistant or tool message to a pure dictionary structure.
    This guarantees compatibility with OpenRouter and all backend endpoint routers.
    """
    if isinstance(message, dict):
        return message
    m_dict = {"role": getattr(message, "role", "assistant")}
    content = getattr(message, "content", None)
    if content is not None:
        m_dict["content"] = content
    if getattr(message, "tool_calls", None):
        m_dict["tool_calls"] = [
            {
                "id": tc.id,
                "type": tc.type,
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments
                }
            } for tc in message.tool_calls
        ]
    return m_dict

async def compress_memory(history: list) -> list:
    if len(history) <= 14:
        return history
        
    system_prompt = history[0]
    
    older_messages = history[1:len(history)-6]
    prompt = "Summarize the following chat history concisely. Focus strictly on what actions were taken, what files were created/modified, and what system bugs or test failures were encountered. Keep it strictly factual under 150 words."
    summary_messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": json.dumps(older_messages)}
    ]
    
    try:
        async_gen = run_online_engine(
            messages=summary_messages,
            tools=None,
            default_model=DEFAULT_MODEL
        )
        full_summary = ""
        async for event in async_gen:
            if event["type"] == "text":
                full_summary += event["content"]
                
        if not full_summary:
            raise ValueError("No summary generated.")
            
        compressed_summary = {
            "role": "system",
            "content": f"🛡️ Compressed History Summary of older events: {full_summary}"
        }
        
        recent_messages = history[-6:]
        return [system_prompt, compressed_summary] + recent_messages
    except Exception as e:
        print_system_msg(f"⚠️ History compression failed: {str(e)}. Hard-truncating history to save tokens.")
        return [system_prompt] + history[-10:]

SYSTEM_PROMPT = """You are an intelligent code assistant. Use the provided tools (search_codebase, read_file, etc.) to perform tasks. Synthesize data from tools and conversation history accurately.

SMART FILE HANDLING PROTOCOL:
1. Documents (PDF/DOCX/Images): Do NOT use `read_file`. Use `search_codebase` to semantically extract information since they are already in the vector database.
2. Large Code Files: Do NOT read or overwrite entire files. Scout the logic using `search_codebase`, read specific lines using `read_file_chunk`, and edit surgically using `replace_in_file`.
"""


async def async_main():
    global DEFAULT_MODEL
    
    os.system("cls" if os.name == "nt" else "clear")
    
    watcher = None
    session_metrics = SessionMetrics()
    try:
        with show_spinner("Indexing codebase for Semantic Search (RAG)..."):
            try:
                index_full_codebase(os.getcwd())
                watcher = start_watcher(os.getcwd())
            except Exception as startup_err:
                print_error(f"⚠️  Semantic RAG indexing bypass or error: {str(startup_err)}")

        chat_history = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

        cwd = os.getcwd()
        show_welcome_banner(DEFAULT_MODEL, cwd)

        while True:
            try:
                user_input = get_user_input()
                
                if not user_input:
                    continue
                    
                if user_input.lower() in ["exit", "quit"]:
                    print_system_msg("\n👋 Exiting SuperAgent. Goodbye!")
                    break
                    
                task_metrics = TaskMetrics()
                execution_timeline = [(datetime.now(), "User Query Received")]
                    
                if user_input.startswith("/help"):
                    print_system_msg("\nAvailable Commands:\n• /clear : Clear history\n• /model <name> : Switch model\n• /help : Show menu\n• /exit or /quit: Exit")
                    continue



                if user_input.startswith("/clear"):
                    chat_history = [{"role": "system", "content": SYSTEM_PROMPT}]
                    print_system_msg("\n🧹 Conversation history cleared successfully.")
                    continue

                if user_input.startswith("/model"):
                    parts = user_input.split(maxsplit=1)
                    if len(parts) > 1:
                        new_model = parts[1].strip()
                        DEFAULT_MODEL = new_model
                        print_system_msg(f"\n🔄 Model switched dynamically to: {DEFAULT_MODEL}")
                    else:
                        print_system_msg(f"\nCurrent Active Model: {DEFAULT_MODEL}")
                    continue
                    
                chat_history = await compress_memory(chat_history)
                
                try:
                    past_context = retrieve_past_context(user_input)
                except Exception as e:
                    print_error(f"⚠️ Vector Memory Retrieval Failed: {str(e)}")
                    past_context = None

                # Append ONLY raw user_input to persistent chat history
                chat_history.append({"role": "user", "content": user_input})
                
                error_count = 0
                tool_iteration_count = 0  # Per-query tool call counter to prevent infinite loops
                called_tools = set()
                while True:
                    messages_for_llm = list(chat_history)

                    execution_timeline.append((datetime.now(), "LLM Planning Started"))
                    llm_start_time = time.time()
                    gen = run_online_engine(
                        messages=messages_for_llm,
                        tools=TOOLS_SCHEMA,
                        default_model=DEFAULT_MODEL
                    )

                    spinner_context = show_spinner("Agent is thinking...")
                    spinner_context.__enter__()
                    spinner_active = True

                    full_text = ""
                    response = None
                    is_first_chunk = True
                    live_context = None
                    text_already_streamed = False

                    try:
                        async for event in gen:
                            if event["type"] == "text":
                                content = event["content"]
                                if is_first_chunk:
                                    if spinner_active:
                                        spinner_context.__exit__(None, None, None)
                                        spinner_active = False
                                    is_first_chunk = False

                                full_text += content
                                text_already_streamed = True

                                if "<PLAN>" in full_text:
                                    if "</PLAN>" in full_text:
                                        plan_match = re.search(r"<PLAN>(.*?)</PLAN>", full_text, re.DOTALL)
                                        if plan_match:
                                            plan_text = plan_match.group(1).strip()
                                            print_plan(plan_text)
                                            full_text = full_text.replace(plan_match.group(0), "").strip()

                                            if full_text:
                                                live_context = Live(
                                                    Panel(
                                                        Markdown(full_text),
                                                        title=f"[bold {AGENT_COLOR}]🤖 Agent[/]",
                                                        border_style=AGENT_COLOR,
                                                        box=box.ROUNDED,
                                                        expand=False
                                                    ),
                                                    console=console,
                                                    refresh_per_second=15,
                                                    auto_refresh=False
                                                )
                                                live_context.start()
                                    else:
                                        # Buffering plan
                                        pass
                                else:
                                    if not live_context:
                                        console.print()
                                        live_context = Live(
                                            Panel(
                                                Markdown(full_text),
                                                title=f"[bold {AGENT_COLOR}]🤖 Agent[/]",
                                                border_style=AGENT_COLOR,
                                                box=box.ROUNDED,
                                                expand=False
                                            ),
                                            console=console,
                                            refresh_per_second=15,
                                            auto_refresh=False
                                        )
                                        live_context.start()
                                    else:
                                        live_context.update(Panel(
                                            Markdown(full_text),
                                            title=f"[bold {AGENT_COLOR}]🤖 Agent[/]",
                                            border_style=AGENT_COLOR,
                                            box=box.ROUNDED,
                                            expand=False
                                        ))
                                        live_context.refresh()

                            elif event["type"] == "tool_calls":
                                response = event["response"]

                            elif event["type"] == "error":
                                response = None
                                break

                        if live_context:
                            live_context.update(Panel(
                                Markdown(full_text),
                                title=f"[bold {AGENT_COLOR}]🤖 Agent[/]",
                                border_style=AGENT_COLOR,
                                box=box.ROUNDED,
                                expand=False
                            ))
                            live_context.refresh()
                            live_context.stop()

                        if spinner_active:
                            spinner_context.__exit__(None, None, None)
                            spinner_active = False

                    except Exception as stream_err:
                        if live_context:
                            live_context.stop()
                        if spinner_active:
                            spinner_context.__exit__(None, None, None)
                            spinner_active = False
                        print_error(f"\n❌ Error during stream: {str(stream_err)}")
                        response = None

                    if text_already_streamed and response is None:
                        response = MockResponse(full_text, None)

                    task_metrics.llm_latency_seconds += time.time() - llm_start_time

                    if response is None:
                        print_error("\n❌ Critical Error: All LLM models failed.")
                        break

                    if not hasattr(response, "choices") or not response.choices:
                        print_error("\n❌ API Error: Invalid response structure.")
                        if DEFAULT_MODEL != "openrouter/free":
                            print_system_msg("🔄 Attempting automatic fallback...")
                            DEFAULT_MODEL = "openrouter/free"
                            continue
                        break

                    choice = response.choices[0]
                    message = choice.message

                    message_content = getattr(message, "content", None)
                    if message_content and not text_already_streamed:
                        plan_match = re.search(r"<PLAN>(.*?)</PLAN>", message_content, re.DOTALL)
                        if plan_match:
                            plan_text = plan_match.group(1).strip()
                            print_plan(plan_text)
                            message_content = message_content.replace(plan_match.group(0), "").strip()

                        if message_content:
                            print_agent_response(message_content)

                    message_tool_calls = getattr(message, "tool_calls", None)
                    if message_tool_calls:
                        task_metrics.tool_calls += len(message_tool_calls)
                        tool_iteration_count += 1
                        if tool_iteration_count > MAX_TOOL_ITERATIONS:
                            print_error(f"\n🚨 Tool iteration limit ({MAX_TOOL_ITERATIONS}) reached. Halting agent loop to prevent infinite recursion.")
                            print_system_msg("ℹ️  The local model may be stuck in a tool-call loop. Try rephrasing your query.")
                            break
                        chat_history.append(message_to_dict(message))
                        has_error = False
                        retry_failed = False
                        tasks = []
                        valid_tool_calls = []

                        for tool_call in message_tool_calls:
                            tool_name = tool_call.function.name
                            tool_id = tool_call.id
                            
                            try:
                                tool_args = json.loads(tool_call.function.arguments)
                            except json.JSONDecodeError as je:
                                error_msg = f"Error: Failed to parse tool arguments as JSON: {str(je)}"
                                print_error(f"\n⚠️ Invalid tool arguments format for '{tool_name}'")
                                
                                chat_history.append({
                                    "role": "tool",
                                    "tool_call_id": tool_id,
                                    "name": tool_name,
                                    "content": error_msg
                                })
                                has_error = True
                                continue
                                
                            print_system_msg(f"\n⚙️  Running tool: {tool_name}")
                            for arg_key, arg_val in tool_args.items():
                                print_system_msg(f"   {arg_key}: {arg_val}")
                                
                            tool_signature = f"{tool_name}_{str(tool_args)}"
                            if tool_signature in called_tools:
                                print_error(f"\n⚠️ Critical System Override: Model attempted to call '{tool_name}' with the same arguments again.")
                                chat_history.append({
                                    "role": "tool",
                                    "tool_call_id": tool_id,
                                    "name": tool_name,
                                    "content": f"CRITICAL SYSTEM OVERRIDE: Tool '{tool_name}' duplicate call detected. Synthesize final response from existing data without mentioning tools."
                                })
                                has_error = True
                                continue
                                
                            called_tools.add(tool_signature)
                                
                            if tool_name in AVAILABLE_TOOLS:
                                tool_func = AVAILABLE_TOOLS[tool_name]
                                try:
                                    tasks.append(tool_func(**tool_args))
                                    valid_tool_calls.append((tool_id, tool_name))
                                except Exception as e:
                                    async def dummy_err(err=str(e)):
                                        return f"Error building tool call (Missing arguments?): {err}"
                                    tasks.append(dummy_err())
                                    valid_tool_calls.append((tool_id, tool_name))
                            else:
                                chat_history.append({
                                    "role": "tool",
                                    "tool_call_id": tool_id,
                                    "name": tool_name,
                                    "content": f"Error: Tool '{tool_name}' is not supported."
                                })
                                has_error = True

                        if tasks:
                            execution_timeline.append((datetime.now(), "Concurrent Tool Execution Started"))
                            tool_start_time = time.time()
                            with live_tool_execution_panel():
                                results = await asyncio.gather(*tasks, return_exceptions=True)
                            task_metrics.tool_execution_latency_seconds += time.time() - tool_start_time
                            
                            for (tool_id, tool_name), result in zip(valid_tool_calls, results):
                                if isinstance(result, Exception):
                                    tool_result_str = f"Error executing tool '{tool_name}': {str(result)}"
                                else:
                                    tool_result_str = str(result) if result is not None else ""
                                    
                                is_error = tool_result_str.startswith("Error") or "Security Violation" in tool_result_str
                                
                                if not is_error:
                                    if tool_name in ["read_file", "read_file_chunk", "search_codebase"]:
                                        task_metrics.files_read += 1
                                    elif tool_name in ["write_file", "replace_in_file"]:
                                        task_metrics.files_modified += 1
                                    elif tool_name in ["run_terminal_command", "git_checkpoint", "git_revert", "run_tests"]:
                                        task_metrics.commands_executed += 1
                                
                                chat_history.append({
                                    "role": "tool",
                                    "tool_call_id": tool_id,
                                    "name": tool_name,
                                    "content": tool_result_str
                                })
                                
                                if is_error:
                                    has_error = True
                                    print_error(f"\n⚠️ Tool '{tool_name}' failed: {tool_result_str}")

                        if has_error:
                            error_count += 1
                            alert_msg = (
                                f"The tool execution failed. Please analyze the error and try a different approach or tool. (Attempt {error_count} of {MAX_RETRIES})\n"
                                f"Start your response with your new thought process."
                            )
                            chat_history.append({
                                "role": "user",
                                "content": alert_msg
                            })
                            print_system_msg(f"🔄 Self-Correction Attempt {error_count}/{MAX_RETRIES} initiated...")
                            
                            if error_count >= MAX_RETRIES:
                                print_error(f"\n🚨 Critical: Failed to resolve after {MAX_RETRIES} attempts.")
                                retry_failed = True
                        else:
                            error_count = 0
                        
                        if retry_failed:
                            break
                        if has_error:
                            continue
                        
                        chat_history.append({
                            "role": "system",
                            "content": "SYSTEM NUDGE: The tool executed successfully. Summarize the results directly. Do NOT mention the tool execution or internal instructions."
                        })
                        continue
                    else:
                        chat_history.append(message_to_dict(message))
                        
                        final_text = getattr(message, "content", "")
                        if final_text:
                            save_chat_to_memory(user_input, final_text)
                            
                        print_execution_timeline(execution_timeline)
                        print_workspace_activity(task_metrics)
                        task_metrics.add_to_session(session_metrics)
                            
                        break

            except KeyboardInterrupt:
                print_system_msg("\n[System] Process interrupted. Graceful exit initiated. Goodbye. 🚀")
                break
            except EOFError:
                print_system_msg("\n[System] Process interrupted. Graceful exit initiated. Goodbye. 🚀")
                break
            except Exception as e:
                print_error(f"\n❌ Unexpected Error: {str(e)}")

    except KeyboardInterrupt:
        print_system_msg("\n[System] Process interrupted. Graceful exit initiated. Goodbye. 🚀")
        sys.exit(0)
    finally:
        print_session_footer(session_metrics)
        if watcher:
            watcher.stop()
            time.sleep(1.5)  # Wait for ChromaDB WAL sync
            watcher.join()

def main():
    try:
        asyncio.run(async_main())
    except (KeyboardInterrupt, asyncio.CancelledError):
        print_system_msg("\n[System] Process interrupted. Graceful exit initiated. Goodbye. 🚀")
        sys.exit(0)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
