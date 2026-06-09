from rich.console import Console, Group
from rich.panel import Panel
from rich.markdown import Markdown
from rich.rule import Rule
from rich.table import Table
from rich.status import Status
from rich.live import Live
from rich import box
from typing import List, Tuple
from datetime import datetime
from contextlib import contextmanager

from superagent.metrics import SessionMetrics, TaskMetrics

console = Console()

# Professional UI Design System Palette
WHITE = "#FFFFFF"       # Logo & Base text
GREEN = "#A8E6CF"       # Success & Smooth Operations
RED = "#FF4C4C"         # Errors & Alerts
ORANGE = "#D97757"      # Borders & User Inputs ("You")
SKY_BLUE = "#87CEEB"    # Agent System & Accents

# Legacy variables kept strictly for main.py import compatibility
AGENT_COLOR = SKY_BLUE
USER_COLOR = ORANGE
SYSTEM_COLOR = WHITE
ACCENT_COLOR = GREEN


def show_welcome_banner(model_name: str, working_dir: str):
    SAFE_LOGO = f"""[bold {WHITE}]
███████╗██╗   ██╗██████╗ ███████╗ ██████╗     █████╗  ██████╗ ███████╗███╗   ██╗████████╗
██╔════╝██║   ██║██╔══██╗██╔════╝██╔══██╗    ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝
███████╗██║   ██║██████╔╝█████╗  ██████╔╝    ███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║
╚════██║██║   ██║██╔═══╝ ██╔══╝  ██╔══██╗    ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║
███████║╚██████╔╝██║     ███████╗██║  ██║    ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║
╚══════╝ ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═╝    ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝             
[/]"""

    content = (
        f"{SAFE_LOGO}\n"
        f"[bold white]Autonomous Context Engine & Data Mover[/]\n\n"
        f"📍 [bold green]Dir:[/] {working_dir}\n"
        f"🤖 [bold green]Model:[/] {model_name}\n\n"
        f"[dim white]Type /help for commands, or exit to quit.[/]"
    )
    console.print(Panel(
        content,
        title=f"[bold {AGENT_COLOR}]👾 SuperAgent CLI[/]",
        border_style=AGENT_COLOR,
        box=box.ROUNDED,
        padding=(1, 2)
    ))

def get_user_input() -> str:
    console.print(Rule(style="dim #444444"))
    # console.input is better than Prompt.ask for direct hex color support
    return console.input(f"\n[bold {USER_COLOR}]❯ You:[/] ").strip()

def show_spinner(text: str = "Thinking..."):
    # Using 'point' or 'dots' spinner with purple style
    return console.status(f"[bold {AGENT_COLOR}]{text}[/]", spinner="dots", spinner_style=AGENT_COLOR)

def print_system_msg(text: str):
    console.print(f"[{SYSTEM_COLOR}]⚙️ {text}[/]")

def print_agent_response(markdown_text: str):
    console.print()
    console.print(Panel(
        Markdown(markdown_text),
        title=f"[bold {AGENT_COLOR}]🤖 Agent[/]",
        border_style=AGENT_COLOR,
        box=box.ROUNDED,
        expand=False
    ))

def print_plan(plan_text: str):
    console.print(Panel(
        Markdown(plan_text),
        title=f"[bold {ACCENT_COLOR}]🧠 Execution Plan[/]",
        border_style=ACCENT_COLOR,
        box=box.ROUNDED,
        expand=False
    ))

def print_error(text: str):
    console.print(f"[bold #F92672]{text}[/]") # Soft Red

@contextmanager
def live_tool_execution_panel():
    """Context manager to display a live panel during concurrent tool execution."""
    with console.status(f"[bold {ACCENT_COLOR}]⚙️ Executing tools concurrently...[/]", spinner="dots", spinner_style=ACCENT_COLOR) as status:
        yield status

def print_execution_timeline(events: List[Tuple[datetime, str]]):
    """Prints a timeline of system events deterministically."""
    if not events:
        return
        
    table = Table(box=box.SIMPLE, show_header=False, expand=False)
    table.add_column("Time", style="dim cyan")
    table.add_column("Event", style="white")
    
    for dt, event in events:
        table.add_row(f"[{dt.strftime('%H:%M:%S')}]", event)
        
    console.print()
    console.print(Panel(
        table,
        title=f"[bold {ACCENT_COLOR}]⏱️ Execution Timeline[/]",
        border_style=ACCENT_COLOR,
        box=box.ROUNDED,
        expand=False
    ))

def print_workspace_activity(task_metrics: TaskMetrics):
    """Prints a summary of task activity."""
    table = Table(box=box.SIMPLE, show_header=False, expand=False)
    table.add_column("Metric", style="dim white")
    table.add_column("Value", style="bold green")
    
    table.add_row("Files Read", str(task_metrics.files_read))
    table.add_row("Files Modified", str(task_metrics.files_modified))
    table.add_row("Commands Executed", str(task_metrics.commands_executed))
    table.add_row("LLM Latency", f"{task_metrics.llm_latency_seconds:.2f}s")
    table.add_row("Tool Execution Latency", f"{task_metrics.tool_execution_latency_seconds:.2f}s")
    
    console.print(Panel(
        table,
        title=f"[bold {AGENT_COLOR}]📊 Workspace Activity Summary[/]",
        border_style=AGENT_COLOR,
        box=box.ROUNDED,
        expand=False
    ))

def print_session_footer(session_metrics: SessionMetrics):
    """Prints the final aggregated session metrics."""
    table = Table(box=box.ROUNDED, expand=False, border_style=AGENT_COLOR)
    table.add_column("Session Metric", style="cyan", justify="left")
    table.add_column("Value", style="bold white", justify="right")
    
    table.add_row("Total Tasks Completed", str(session_metrics.total_tasks))
    table.add_row("Total Tool Calls", str(session_metrics.total_tool_calls))
    table.add_row("Total Files Read", str(session_metrics.files_read))
    table.add_row("Total Files Modified", str(session_metrics.files_modified))
    table.add_row("Total Commands Executed", str(session_metrics.commands_executed))
    table.add_row("Total LLM Wait Time", f"{session_metrics.llm_latency_seconds:.2f}s")
    table.add_row("Total Tool Execution Time", f"{session_metrics.tool_execution_latency_seconds:.2f}s")
    
    console.print()
    console.print(Panel(
        table,
        title=f"[bold {AGENT_COLOR}]📈 Session Footer[/]",
        border_style=AGENT_COLOR,
        box=box.DOUBLE_EDGE,
        expand=False
    ))














# from rich.console import Console, Group
# from rich.panel import Panel
# from rich.markdown import Markdown
# from rich.rule import Rule
# from rich.table import Table
# from rich.status import Status
# from rich.live import Live
# from rich import box
# from typing import List, Tuple
# from datetime import datetime
# from contextlib import contextmanager

# from superagent.metrics import SessionMetrics, TaskMetrics

# console = Console()

# # Professional UI Design System Palette
# WHITE = "#FFFFFF"       # Logo & Base text
# GREEN = "#A8E6CF"       # Success & Smooth Operations
# RED = "#FF4C4C"         # Errors & Alerts
# ORANGE = "#D97757"      # Borders & User Inputs ("You")
# SKY_BLUE = "#87CEEB"    # Agent System & Accents

# # Legacy variables kept strictly for main.py import compatibility
# AGENT_COLOR = SKY_BLUE
# USER_COLOR = ORANGE
# SYSTEM_COLOR = WHITE
# ACCENT_COLOR = GREEN

# def show_welcome_banner(model_name: str, working_dir: str):
#     # Logo stays purely White as requested
#     SAFE_LOGO = f"""[bold {WHITE}]
# ███████╗██╗   ██╗██████╗███████╗██████╗       █████╗  ██████╗ ███████╗███╗   ██╗████████╗
# ██╔════╝██║   ██║██╔══██╗██╔════╝██╔══██╗    ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝
# ███████╗██║   ██║██████╔╝█████╗  ██████╔╝    ███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║
# ╚════██║██║   ██║██╔═══╝ ██╔══╝  ██╔══██╗    ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║
# ███████║╚██████╔╝██║     ███████╗██║  ██║    ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║
# ╚══════╝ ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═╝    ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝             
# [/]"""

#     content = (
#         f"{SAFE_LOGO}\n"
#         f"[bold {WHITE}]Autonomous Context Engine & Data Mover[/]\n\n"
#         f"📍 [bold {GREEN}]Dir:[/] {working_dir}\n"
#         f"🤖 [bold {GREEN}]Model:[/] {model_name}\n\n"
#         f"[{WHITE}]Type /help for commands, or exit to quit.[/]"
#     )
#     console.print(Panel(
#         content,
#         title=f"[bold {SKY_BLUE}]👾 SuperAgent CLI[/]",
#         border_style=ORANGE, # Orange border theme
#         box=box.ROUNDED,
#         padding=(1, 2)
#     ))

# def get_user_input() -> str:
#     console.print(Rule(style=ORANGE))
#     return console.input(f"\n[bold {ORANGE}]❯ You:[/] ").strip()

# def show_spinner(text: str = "Thinking..."):
#     return console.status(f"[bold {SKY_BLUE}]{text}[/]", spinner="dots", spinner_style=SKY_BLUE)

# def print_system_msg(text: str):
#     console.print(f"[{SKY_BLUE}]⚙️ {text}[/]")

# def print_agent_response(markdown_text: str):
#     console.print()
#     console.print(Panel(
#         Markdown(markdown_text),
#         title=f"[bold {SKY_BLUE}]🤖 Agent[/]",
#         border_style=ORANGE, # Orange borders
#         box=box.ROUNDED,
#         expand=False
#     ))

# def print_plan(plan_text: str):
#     console.print(Panel(
#         Markdown(plan_text),
#         title=f"[bold {GREEN}]🧠 Execution Plan[/]",
#         border_style=ORANGE,
#         box=box.ROUNDED,
#         expand=False
#     ))

# def print_error(text: str):
#     # Errors strictly red for maximum visibility
#     console.print(f"[bold {RED}]{text}[/]")

# @contextmanager
# def live_tool_execution_panel():
#     """Context manager to display a live panel during concurrent tool execution."""
#     with console.status(f"[bold {GREEN}]⚙️ Executing tools concurrently...[/]", spinner="dots", spinner_style=GREEN) as status:
#         yield status

# def print_execution_timeline(events: List[Tuple[datetime, str]]):
#     """Prints a timeline of system events deterministically."""
#     if not events:
#         return
        
#     table = Table(box=box.SIMPLE, show_header=False, expand=False)
#     table.add_column("Time", style=SKY_BLUE)
#     table.add_column("Event", style=WHITE)
    
#     for dt, event in events:
#         table.add_row(f"[{dt.strftime('%H:%M:%S')}]", event)
        
#     console.print()
#     console.print(Panel(
#         table,
#         title=f"[bold {GREEN}]⏱️ Execution Timeline[/]",
#         border_style=ORANGE,
#         box=box.ROUNDED,
#         expand=False
#     ))

# def print_workspace_activity(task_metrics: TaskMetrics):
#     """Prints a summary of task activity."""
#     table = Table(box=box.SIMPLE, show_header=False, expand=False)
#     table.add_column("Metric", style=WHITE)
#     table.add_column("Value", style=f"bold {GREEN}") # Successful operations in Green
    
#     table.add_row("Files Read", str(task_metrics.files_read))
#     table.add_row("Files Modified", str(task_metrics.files_modified))
#     table.add_row("Commands Executed", str(task_metrics.commands_executed))
#     table.add_row("LLM Latency", f"{task_metrics.llm_latency_seconds:.2f}s")
#     table.add_row("Tool Execution Latency", f"{task_metrics.tool_execution_latency_seconds:.2f}s")
    
#     console.print(Panel(
#         table,
#         title=f"[bold {SKY_BLUE}]📊 Workspace Activity Summary[/]",
#         border_style=ORANGE,
#         box=box.ROUNDED,
#         expand=False
#     ))

# def print_session_footer(session_metrics: SessionMetrics):
#     """Prints the final aggregated session metrics."""
#     table = Table(box=box.ROUNDED, expand=False, border_style=ORANGE) # Table grid in Orange
#     table.add_column("Session Metric", style=SKY_BLUE, justify="left")
#     table.add_column("Value", style=f"bold {WHITE}", justify="right")
    
#     table.add_row("Total Tasks Completed", str(session_metrics.total_tasks))
#     table.add_row("Total Tool Calls", str(session_metrics.total_tool_calls))
#     table.add_row("Total Files Read", str(session_metrics.files_read))
#     table.add_row("Total Files Modified", str(session_metrics.files_modified))
#     table.add_row("Total Commands Executed", str(session_metrics.commands_executed))
#     table.add_row("Total LLM Wait Time", f"{session_metrics.llm_latency_seconds:.2f}s")
#     table.add_row("Total Tool Execution Time", f"{session_metrics.tool_execution_latency_seconds:.2f}s")
    
#     console.print()
#     console.print(Panel(
#         table,
#         title=f"[bold {SKY_BLUE}]📈 Session Footer[/]",
#         border_style=ORANGE,
#         box=box.DOUBLE_EDGE,
#         expand=False
#     ))