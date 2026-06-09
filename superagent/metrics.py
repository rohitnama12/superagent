from dataclasses import dataclass
from typing import List
import time

@dataclass
class SessionMetrics:
    files_read: int = 0
    files_modified: int = 0
    commands_executed: int = 0
    llm_latency_seconds: float = 0.0
    tool_execution_latency_seconds: float = 0.0
    total_tool_calls: int = 0
    total_tasks: int = 0

@dataclass
class TaskMetrics:
    files_read: int = 0
    files_modified: int = 0
    commands_executed: int = 0
    llm_latency_seconds: float = 0.0
    tool_execution_latency_seconds: float = 0.0
    tool_calls: int = 0

    def add_to_session(self, session_metrics: SessionMetrics):
        session_metrics.files_read += self.files_read
        session_metrics.files_modified += self.files_modified
        session_metrics.commands_executed += self.commands_executed
        session_metrics.llm_latency_seconds += self.llm_latency_seconds
        session_metrics.tool_execution_latency_seconds += self.tool_execution_latency_seconds
        session_metrics.total_tool_calls += self.tool_calls
        session_metrics.total_tasks += 1
