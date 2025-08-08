from __future__ import annotations
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import json
import platform
import subprocess
import threading
import time
import traceback
import textwrap
import copy
import ast
import re
import glob
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Set
from sentence_transformers import SentenceTransformer
import pickle
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
from nbclient import NotebookClient
from jupyter_client import KernelManager
from pydantic import BaseModel, ValidationError

# Fixed LangChain imports
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
try:
    from langchain_ollama import OllamaEmbeddings  # New import path
except ImportError:
    from langchain_community.embeddings import OllamaEmbeddings  # Fallback


# FAISS for efficient embedding storage
import faiss
import numpy as np

from ollama import chat
from loguru import logger

# ================================
# CONFIGURATION & PATHS
# ================================

CHAT_HISTORY_DIR = Path("chat_history")
NOTEBOOKS_DIR = Path("notebooks") 
AGENT_MEMORY_DIR = Path("agent_memory")
AGENT_MEMORY_DIR.mkdir(exist_ok=True)
NOTEBOOKS_DIR.mkdir(exist_ok=True)

MAX_CODE_RETRIES = 2  # Reduced for faster execution
MAX_CONTEXT_LENGTH = 20000  # Well under 32K limit
MAX_MEMORY_MESSAGES = 50

# ================================
# PYDANTIC MODELS
# ================================

class NotebookCellOutput(BaseModel):
    type: str
    content: Optional[str] = ""

class GeneratedCell(BaseModel):
    markdown: str
    code: str

class AgentOutput(BaseModel):
    cell: GeneratedCell
    status: str

class SessionMemory(BaseModel):
    session_id: str
    messages: List[Dict[str, Any]] = []
    notebook_context: Dict[str, Any] = {}
    last_updated: str = ""
def test_model_basic():
    """Test if model is responding at all"""
    try:
        response = chat(
            messages=[{"role": "user", "content": "Say hello"}],
            model="gemma3n:e2b",
            options={"num_ctx": 1000}
        )
        
        content = response.message.content
        logger.info(f"✅ Model test response: '{content}'")
        return len(content) > 0
        
    except Exception as e:
        logger.error(f"❌ Model test failed: {e}")
        return False

# ================================
# FAISS-BASED CODE KNOWLEDGE SYSTEM  
# ================================
class CodeKnowledgeBase:
    """
    Manages a knowledge base of code snippets for RAG-based error correction.
    Loads code, creates vector embeddings, and allows for similarity search.
    """
    def __init__(self, file_path="knowledge_base.json"):
        self.file_path = file_path
        self.knowledge_base = self._load_kb()
        # Use the specified embedding model
        self.embedding_model = OllamaEmbeddings(model="nomic-embed-text:v1.5")
        self.index, self.id_map = self._create_index()

    def _load_kb(self) -> List[Dict]:
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Could not load knowledge base: {e}")
            return []

    def _create_index(self):
        if not self.knowledge_base:
            return None, None
        
        descriptions = [item['description'] for item in self.knowledge_base]
        try:
            logger.info("Creating vector embeddings for knowledge base...")
            embeddings = self.embedding_model.embed_documents(descriptions)
            embeddings_np = np.array(embeddings, dtype='float32')
            
            index = faiss.IndexFlatL2(embeddings_np.shape[1])
            index.add(embeddings_np)
            
            id_map = {i: item['id'] for i, item in enumerate(self.knowledge_base)}
            logger.info("✅ Knowledge base indexed successfully.")
            return index, id_map
        except Exception as e:
            logger.error(f"Failed to create FAISS index: {e}")
            return None, None
            
    def search(self, query: str, k: int = 1) -> Optional[Dict]:
        """
        Finds the most relevant code snippet from the knowledge base.
        """
        if not self.index:
            return None
        
        try:
            query_embedding = self.embedding_model.embed_query(query)
            query_np = np.array([query_embedding], dtype='float32')
            
            distances, indices = self.index.search(query_np, k)
            
            if len(indices[0]) > 0:
                best_index = indices[0][0]
                item_id = self.id_map[best_index]
                # Find the full item from the knowledge base
                return next((item for item in self.knowledge_base if item['id'] == item_id), None)
        except Exception as e:
            logger.error(f"Knowledge base search failed: {e}")
            return None
        return None

class IntelligentTaskPlanner:
    """Enhanced task planning based on dataset characteristics"""
    
    def __init__(self, knowledge_base: CodeKnowledgeBase):
        self.kb = knowledge_base
    
    
class AutonomousTaskPlanner:
    """
    Complete autonomous task planner that generates, executes, and adapts data science workflows.
    
    Core capabilities:
    - Dynamic task plan generation based on user requests
    - Autonomous task execution with error handling
    - Plan adaptation when tasks fail
    - Context-aware decision making
    """
    
    def __init__(self, knowledge_base=None):
        """Initialize the autonomous planner with optional knowledge base."""
        self.knowledge_base = knowledge_base
        self.execution_history = []
        self.current_context = {}
        self.user_goal = ""
        self.max_retries = 3
        self.max_context_tokens = 15000
        
        # Task execution state
        self.completed_tasks = []
        self.failed_tasks = []
        self.current_notebook = None
        self.current_client = None
    
    # ====================================================================
    # AUTONOMOUS PLANNING METHODS
    # ====================================================================
    
    def analyze_and_plan(self, user_request: str, dataset_info: Dict[str, Any]) -> List[str]:
        """Generate a completely custom analysis plan - SIMPLIFIED for gemma3n:e2b"""
        self.user_goal = user_request

        # Much shorter prompt for gemma3n:e2b
        planning_prompt = """Create a data analysis plan.

    USER WANTS: {user_request}
    DATASET: {dataset_info.get('name', 'data.csv')}
    COLUMNS: {str(dataset_info.get('columns', []))[:100]}

    Create 4-5 specific analysis steps. Return JSON only:
    {{"tasks": ["step1", "step2", "step3", "step4"]}}"""

        try:
            response = chat(
                messages=[{"role": "user", "content": planning_prompt}],
                model="gemma3n:e2b",
                options={"temperature": 0.7, "num_ctx": 8000}
            )

            # FIXED: Complete JSON extraction with proper syntax
            content = response.message.content.strip()
            if not content:
                logger.error("Empty response from LLM")
                return self._create_fallback_plan(user_request, dataset_info)

            # Handle markdown code blocks properly
            if content.startswith('```'):
                # Remove ```json prefix and ```
                content = content[7:]  # Remove '```json'
                if content.endswith('```'):
                    content = content[:-3]  # Remove '```'
                content = content.strip()
            elif content.startswith('```'):
                # Remove generic ``` prefix and suffix
                content = content[3:]  # Remove '```
                if content.endswith('```'):
                    content = content[:-3]  # Remove '```
                content = content.strip()

            # Parse JSON
            data = json.loads(content)
            tasks = data.get("tasks", [])

            if tasks and len(tasks) >= 3:
                logger.info(f"🧠 Agent generated {len(tasks)} custom tasks")
                return tasks
            else:
                logger.warning("Invalid tasks generated, using fallback")
                return self._create_fallback_plan(user_request, dataset_info)

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            logger.error(f"Content was: {content[:200]}...")
            return self._create_fallback_plan(user_request, dataset_info)
        except Exception as e:
            logger.error(f"Planning failed: {e}")
            return self._create_fallback_plan(user_request, dataset_info)
        
    def _create_fallback_plan(self, user_request: str, dataset_info: Dict[str, Any]) -> List[str]:
        """Create completely generic fallback plan"""
        dataset_name = dataset_info.get('name', 'dataset.csv')
        
        return [
            f"Load and inspect {dataset_name}",
            "Perform comprehensive data quality assessment", 
            "Generate statistical summaries and insights",
            "Create appropriate visualizations for the data",
            "Address user's specific request through analysis",
            "Generate actionable conclusions and recommendations"
        ]

    
    # ====================================================================
    # TASK EXECUTION METHODS  
    # ====================================================================
    
    def execute_task(self, task: str, user_request: str, dataset_info: Dict[str, Any], 
                    step_index: int, notebook=None, client=None) -> Tuple[bool, str]:
        """
        Execute a single autonomous task.
        
        Args:
            task: The specific task to execute
            user_request: Original user request for context
            dataset_info: Dataset information
            step_index: Which step this is (0-based)
            notebook: Notebook object for adding cells
            client: Execution client for running code
            
        Returns:
            (success: bool, error_message: str)
        """
        logger.info(f"🔄 Executing autonomous task {step_index + 1}: {task}")
        
        try:
            # Generate code for this specific task
            agent_output = self._generate_task_code(
                task=task,
                user_request=user_request,
                dataset_name=dataset_info.get('name', 'data.csv'),
                step_context=f"Step {step_index + 1}",
                previous_context=self._get_execution_context()
            )
            
            if not agent_output:
                return False, "Failed to generate code for task"
            
            # Add cells to notebook if provided
            if notebook is not None:
                from coding_agent import append_cells
                append_cells(notebook, agent_output['markdown'], agent_output['code'])
            
            # Execute the code if client provided
            if client is not None:
                success, error, outputs = client.execute_cell_smart(
                    len(notebook.cells) - 1, force_execution=True
                )
                
                if success:
                    # Store successful execution
                    self.execution_history.append({
                        'task': task,
                        'step': step_index,
                        'success': True,
                        'code': agent_output['code'],
                        'outputs': outputs
                    })
                    self.completed_tasks.append(task)
                    logger.info(f"✅ Task {step_index + 1} completed successfully")
                    return True, ""
                else:
                    # Store failed execution
                    self.execution_history.append({
                        'task': task,
                        'step': step_index,
                        'success': False,
                        'error': error
                    })
                    self.failed_tasks.append(task)
                    logger.error(f"❌ Task {step_index + 1} failed: {error}")
                    return False, error
            
            # If no client, assume success for planning purposes
            self.completed_tasks.append(task)
            return True, ""
            
        except Exception as e:
            error_msg = f"Task execution crashed: {str(e)}"
            logger.error(error_msg)
            self.failed_tasks.append(task)
            return False, error_msg
    
    def _generate_task_code(self, task: str, user_request: str, dataset_name: str, 
                            step_context: str, previous_context: str) -> Optional[Dict[str, str]]:
            """Generate code for a specific task - SIMPLIFIED for gemma3n:e2b"""

            # Short, focused prompt
            code_prompt = """Generate code for: {task}

        Dataset: {dataset_name} (df already loaded)
        Context: {step_context}

        Write working Python code. Return JSON:
        {{"markdown": "## Task", "code": "# Working code here"}}"""

            try:
                # Use chat with explicit timeout
                import signal

                def timeout_handler(signum, frame):
                    raise TimeoutError("LLM call timed out")

                # Set timeout alarm (Unix systems)
                try:
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(90)  # 90 second timeout
                except Exception as e:
                    pass  # Windows doesn't support signal.alarm
                
                response = chat(
                    messages=[{"role": "user", "content": code_prompt}],
                    model="gemma3n:e2b",
                    options={"temperature": 0.7, "num_ctx": 10000}
                )

                # Disable timeout
                try:
                    signal.alarm(0)
                except:
                    pass
                
                # FIXED: Proper content extraction
                content = response.message.content.strip()

                if not content:
                    logger.error("Empty response from code generation")
                    return self._create_fallback_code(task)

                # Handle markdown blocks
                if content.startswith('```json'):
                    content = content[7:]
                    if content.endswith('```'):
                        content = content[:-3]
                    content = content.strip()
                elif content.startswith('```'):
                    content = content[3:]
                    if content.endswith('```'):
                        content = content[:-3]
                    content = content.strip()

                # Parse JSON
                data = json.loads(content)
                return {
                    'markdown': data.get('markdown', f'## {task}'),
                    'code': data.get('code', f'# {task}\npass')
                }

            except TimeoutError:
                logger.error("Code generation timed out")
                return self._create_fallback_code(task)
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing failed in code generation: {e}")
                return self._create_fallback_code(task)
            except Exception as e:
                logger.error(f"Code generation failed: {e}")
                return self._create_fallback_code(task)
    
    # ====================================================================
    # ADAPTIVE PLANNING METHODS
    # ====================================================================
    
    def adapt_plan_after_failure(self, original_tasks: List[str], failed_step_idx: int, 
                                error_context: str, user_request: str) -> List[str]:
        """
        Autonomously adapt the plan after a task failure.
        
        Args:
            original_tasks: The original task list
            failed_step_idx: Index of the failed task
            error_context: Error message from the failure
            user_request: Original user request for context
            
        Returns:
            Adapted task list
        """
        logger.info(f"🤔 Adapting plan due to failure at step {failed_step_idx + 1}")
        
        failed_task = original_tasks[failed_step_idx] if failed_step_idx < len(original_tasks) else "Unknown"
        remaining_tasks = original_tasks[failed_step_idx + 1:]
        completed_tasks = original_tasks[:failed_step_idx]
        
        adaptation_prompt = """You are an autonomous data science agent adapting your analysis plan after a failure.
        ORIGINAL USER REQUEST: {user_request}
        FAILED TASK: {failed_task}
        ERROR CONTEXT: {error_context}
        COMPLETED TASKS: {completed_tasks}
        REMAINING PLANNED TASKS: {remaining_tasks}

        Create an adapted plan for the remaining steps that:
        1. Works around the failure
        2. Still addresses the user's original request  
        3. Uses simpler, more reliable approaches
        4. Has 3-5 focused remaining steps maximum

        OUTPUT: JSON format with "adapted_tasks" array and "reasoning" field.
        Example: {{"adapted_tasks": ["simplified_task_1", "task_2"], "reasoning": "Switched to simpler approach due to error"}}"""

        try:
            response = chat(
                messages=[{"role": "user", "content": adaptation_prompt}],
                model="gemma3n:e2b",
                options={"temperature": 0.7, "num_ctx": self.max_context_tokens}
            )
            
            data = json.loads(response.message.content)
            adapted_tasks = data.get("adapted_tasks", [])
            reasoning = data.get("reasoning", "")
            
            logger.info(f"🔄 Plan adapted: {reasoning}")
            
            # Return complete adapted plan (completed + adapted remaining)
            return completed_tasks + adapted_tasks
            
        except Exception as e:
            logger.error(f"Plan adaptation failed: {e}")
            # Simple fallback adaptation
            return completed_tasks + [
                "Perform simplified analysis to address user request",
                "Generate basic insights and conclusions"
            ]
    
    # ====================================================================
    # EXECUTION ORCHESTRATION
    # ====================================================================
    
    def run_full_autonomous_analysis(self, user_request: str, dataset_info: Dict[str, Any], 
                                    notebook=None, client=None) -> bool:
        """
        Run complete autonomous analysis from planning through execution.
        
        Args:
            user_request: What the user wants to accomplish
            dataset_info: Information about the dataset
            notebook: Notebook object for adding cells (optional)
            client: Execution client for running code (optional)
            
        Returns:
            True if analysis completed successfully, False otherwise
        """
        logger.info(f"🚀 Starting autonomous analysis: {user_request}")
        
        # Step 1: Generate autonomous plan
        task_plan = self.analyze_and_plan(user_request, dataset_info)
        
        # Step 2: Execute tasks with adaptation capability
        for step_idx, task in enumerate(task_plan):
            success, error_msg = self.execute_task(
                task, user_request, dataset_info, step_idx, notebook, client
            )
            
            if not success:
                logger.warning(f"Task {step_idx + 1} failed, adapting plan...")
                
                # Adapt the plan
                adapted_plan = self.adapt_plan_after_failure(
                    task_plan, step_idx, error_msg, user_request
                )
                
                # Continue with adapted plan
                remaining_tasks = adapted_plan[step_idx:]
                for new_idx, new_task in enumerate(remaining_tasks, start=step_idx):
                    success, _ = self.execute_task(
                        new_task, user_request, dataset_info, new_idx, notebook, client
                    )
                    if not success:
                        logger.error(f"Adapted task also failed, stopping analysis")
                        return False
                
                break  # Exit original loop since we handled remaining tasks
        
        logger.info("🎉 Autonomous analysis completed successfully")
        return True
    
    # ====================================================================
    # CONTEXT AND STATE MANAGEMENT
    # ====================================================================
    
    def _get_execution_context(self) -> str:
        """Get context from previous executions for informed decision making."""
        if not self.execution_history:
            return "No previous execution context available."
        
        context = "PREVIOUS EXECUTION CONTEXT:\n"
        recent_executions = self.execution_history[-3:]  # Last 3 executions
        
        for i, exec_info in enumerate(recent_executions, 1):
            status = "✅ SUCCESS" if exec_info['success'] else "❌ FAILED"
            context += f"{i}. {status}: {exec_info['task']}\n"
            if not exec_info['success']:
                context += f"   Error: {exec_info.get('error', 'Unknown error')}\n"
        
        context += f"\nCompleted Tasks: {len(self.completed_tasks)}\n"
        context += f"Failed Tasks: {len(self.failed_tasks)}\n"
        
        return context
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of the autonomous analysis session."""
        return {
            'user_goal': self.user_goal,
            'total_tasks_executed': len(self.execution_history),
            'completed_tasks': self.completed_tasks,
            'failed_tasks': self.failed_tasks,
            'success_rate': len(self.completed_tasks) / max(len(self.execution_history), 1),
            'execution_history': self.execution_history
        }
    
    def reset_session(self):
        """Reset the planner state for a new analysis session."""
        self.execution_history = []
        self.current_context = {}
        self.user_goal = ""
        self.completed_tasks = []
        self.failed_tasks = []
        logger.info("🔄 Autonomous planner session reset")

class SubtaskPlanner:
    """Plans granular subtasks - one command per cell"""
    
    def __init__(self, knowledge_base=None):
        self.knowledge_base = knowledge_base
        

def perform_initial_dataset_analysis(dataset_name: str) -> Dict[str, Any]:
    """Quick analysis to inform task planning"""
    
    analysis_code = """
    import pandas as pd
    import numpy as np

    # Quick dataset inspection
    df = pd.read_csv('{dataset_name}')
    df.columns = df.columns.str.strip()

    # Gather key info for planning
    info = {{
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'sample': df.head(3).to_dict(),
        'missing': df.isnull().sum().to_dict(),
        'numeric_cols': df.select_dtypes(include=['number']).columns.tolist(),
        'categorical_cols': df.select_dtypes(include=['object']).columns.tolist()
    }}

    print("DATASET_INFO:", info)
    """
    
    # Execute this quickly to get dataset characteristics
    # (implement with temporary kernel)
    # Return the extracted info for planning
    
    return {
        'name': dataset_name,
        'preview': 'Dataset characteristics extracted',
        # Add actual extracted info here
    }
    
def generate_autonomous_task_plan(self, user_query: str, dataset_info: Dict) -> List[str]:
    """Let the LLM generate a custom task plan based on the specific request and data"""
    
    # Keep prompt concise for gemma3n:e2b context limits
    planning_prompt = """You are an autonomous data science agent. Create a custom analysis plan.

USER REQUEST: {user_query}
DATASET: {dataset_info.get('name', 'unknown')} - {dataset_info.get('shape', 'unknown shape')}
COLUMNS: {dataset_info.get('columns', [])[:10]}  # Limit to prevent context overflow

Generate 5-8 specific tasks that directly address this user's request with this dataset.
Make each task specific to the actual data and user goals, not generic analysis steps.

Return as JSON: {{"tasks": ["task1", "task2", ...], "reasoning": "why these specific tasks"}}"""

    try:
        response = chat(
            messages=[{"role": "user", "content": planning_prompt}],
            model="gemma3n:e2b",
            options={"temperature": 0.7, "num_ctx": 15000}  # Limit context for your model
        )
        
        # Parse and validate JSON response
        data = json.loads(response.message.content)
        tasks = data.get("tasks", [])
        reasoning = data.get("reasoning", "No reasoning provided")
        
        logger.info(f"🤖 Agent reasoning: {reasoning}")
        
        if not tasks:
            raise ValueError("No tasks generated")
            
        return tasks
        
    except Exception as e:
        logger.error(f"Autonomous planning failed: {e}")
        # Fallback to minimal autonomous plan
        return [
            f"Load and explore {dataset_info.get('name', 'dataset')}",
            f"Address user request: {user_query}",
            "Generate insights and conclusions"
        ]

def execute_with_timeout(func, timeout_seconds: int = 120, *args, **kwargs):
    """Execute function with timeout to prevent hanging"""
    import threading
    import queue
    
    result_queue = queue.Queue()
    exception_queue = queue.Queue()
    
    def target():
        try:
            result = func(*args, **kwargs)
            result_queue.put(result)
        except Exception as e:
            exception_queue.put(e)
    
    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout=timeout_seconds)
    
    if thread.is_alive():
        logger.error(f"Function timed out after {timeout_seconds} seconds")
        return None
    
    # Check for exceptions
    if not exception_queue.empty():
        raise exception_queue.get()
    
    # Get result
    if not result_queue.empty():
        return result_queue.get()
    
    return None


def execute_custom_task(self, task: str, user_query: str, dataset_info: Dict, step_idx: int) -> bool:
    """Execute a specific task with context awareness"""
    try:
        # Generate code for this specific task
        agent_output = generate_autonomous_cell(
            task=task,
            user_request=user_query,
            dataset_name=dataset_info.get('name', 'data.csv'),
            step_context=f"Step {step_idx + 1}",
            error_msg=None
        )
        
        if agent_output:
            logger.info(f"✅ Task '{task}' executed successfully")
            return True
        else:
            logger.error(f"❌ Failed to generate code for task: {task}")
            return False
            
    except Exception as e:
        logger.error(f"Task execution failed: {e}")
        return False

def generate_autonomous_cell(
    task: str,
    user_request: str, 
    dataset_name: str,
    step_context: str,
    error_msg: Optional[str] = None
) -> Optional[AgentOutput]:
    """Generate code that's specific to the task and user needs"""
    
    prompt = """You are executing a custom analysis plan. Generate code for this specific step.

ORIGINAL USER REQUEST: {user_request}
CURRENT TASK: {task}
CONTEXT: {step_context}
DATASET: {dataset_name}

Generate code that specifically addresses this task in service of the user's original goal.
- Be specific to the task and user needs
- Use df (already loaded and cleaned)
- Focus on what the user actually wants to know
- Keep it concise and targeted

{f"FIX THIS ERROR: {error_msg}" if error_msg else ""}

JSON format: {{"cell": {{"markdown": "## task title", "code": "specific_code"}}, "status": "continue"}}"""

    try:
        response = chat(
            messages=[{"role": "user", "content": prompt}],
            model="gemma3n:e2b",
            format=AgentOutput.model_json_schema(),
            options={"temperature": 0.7, "num_ctx": 15000}
        )
        
        return AgentOutput.model_validate(json.loads(response.message.content))
        
    except Exception as e:
        logger.error(f"Code generation failed: {e}")
        return None

def detect_action_loop(previous_actions: List[str], current_action: str) -> bool:
    """Enhanced loop detection with semantic similarity"""
    
    if len(previous_actions) < 2:
        return False
    
    current_lower = current_action.lower()
    recent_actions = previous_actions[-5:]  # Check last 5 actions
    
    # Check for exact repeats
    exact_repeats = sum(1 for action in recent_actions if action.lower() == current_lower)
    if exact_repeats >= 1:
        logger.warning(f"🔄 Exact repeat detected: {current_action}")
        return True
    
    # Enhanced semantic similarity detection
    analysis_categories = {
        'visualization': ['plot', 'chart', 'graph', 'visualiz', 'hist', 'scatter', 'heatmap'],
        'statistics': ['calculate', 'mean', 'std', 'statistic', 'describe', 'summary'],
        'data_inspection': ['check', 'shape', 'info', 'head', 'tail', 'column', 'dtype'],
        'missing_values': ['missing', 'null', 'nan', 'isnull', 'fillna']
    }
    
    # Determine current category
    current_category = None
    for category, keywords in analysis_categories.items():
        if any(keyword in current_lower for keyword in keywords):
            current_category = category
            break
    
    if current_category:
        # Count recent actions in same category
        category_count = 0
        for prev_action in recent_actions:
            prev_lower = prev_action.lower()
            if any(keyword in prev_lower for keyword in analysis_categories[current_category]):
                category_count += 1
        
        # If 2+ recent actions in same category, it's a loop
        if category_count >= 2:
            logger.warning(f"🔄 Category loop detected: {current_category}")
            return True
    
    # Word overlap check
    for prev_action in recent_actions[-3:]:
        current_words = set(current_lower.split())
        prev_words = set(prev_action.lower().split())
        overlap = len(current_words.intersection(prev_words))
        if overlap >= 3:
            logger.warning(f"🔄 High word overlap detected: {overlap} words")
            return True
    
    return False


def log_agent_step_realtime(chat_id: str, step_data: Dict):
    """Enhanced real-time logging with visible JSON output"""
    history_path = CHAT_HISTORY_DIR / f"{chat_id}.json"
    
    try:
        # Load existing history
        history = json.loads(history_path.read_text(encoding="utf-8"))
        
        # Create detailed step log
        step_log = {
            "timestamp": datetime.now().isoformat(),
            "step_type": "agent_micro_action",
            "step_number": step_data["step"],
            "action_description": step_data["action_description"],
            "model_response": {
                "markdown_generated": step_data["model_response"].get("markdown", ""),
                "code_generated": step_data["model_response"].get("code", ""),
                "code_lines_count": step_data["model_response"].get("code_lines", 0),
                "processing_time_seconds": step_data["model_response"].get("processing_time", 0)
            },
            "execution_results": {
                "success": step_data["status"] == "success",
                "error_message": step_data["model_response"].get("error"),
                "outputs_preview": step_data["model_response"].get("outputs", []),
                "cell_index": step_data["model_response"].get("execution_details", {}).get("cell_index")
            },
            "agent_state": {
                "total_steps_so_far": step_data["step"],
                "status": step_data["status"]
            }
        }
        
        # Append to history
        history.append(step_log)
        
        # Keep history manageable
        if len(history) > 200:
            history = history[-200:]
            
        # Save immediately
        history_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
        
        # 🆕 ENHANCED DISPLAY WITH JSON VISIBILITY
        print("\n" + "="*80)
        print(f"🤖 AGENT STEP {step_data['step']} - DECISION PROCESS")
        print("="*80)
        print(f"📝 Action: {step_data['action_description']}")
        print(f"⏱️  Time: {step_data['model_response'].get('processing_time', 0):.1f}s")
        print(f"📊 Status: {'✅ SUCCESS' if step_data['status'] == 'success' else '❌ FAILED'}")
        
        # Show agent's reasoning (if available)
        reasoning = step_data.get('reasoning', 'No reasoning provided')
        print(f"🧠 Agent Reasoning: {reasoning}")
        
        # Show generated code
        code = step_data["model_response"].get("code", "")
        if code:
            print(f"\n💻 Generated Code:")
            print("-" * 40)
            # Show full code, not truncated
            print(code)
            print("-" * 40)
        
        # Show execution results
        outputs = step_data["model_response"].get("outputs", [])
        if outputs:
            print(f"\n📋 Execution Output:")
            for output in outputs[:3]:  # Show first 3 outputs
                content = output.get("content", "")
                if content:
                    print(f"   {content}")
        
        if step_data["status"] != "success":
            error = step_data["model_response"].get("error", "Unknown error")
            print(f"\n⚠️  Error Details:")
            print(f"   {error}")
        
        # 🆕 DISPLAY JSON LOG LOCATION
        print(f"\n📄 JSON Log: {history_path}")
        print(f"📊 Total History Entries: {len(history)}")
        
        print("="*80 + "\n")
        
        logger.debug(f"📝 Logged step {step_data['step']} with enhanced display")
        
    except Exception as e:
        logger.error(f"Real-time logging failed: {e}")
        # Fallback display
        print(f"\n❌ LOGGING ERROR: {e}")
        print(f"Step {step_data.get('step', '?')}: {step_data.get('action_description', 'Unknown')}")


def execute_and_reflect(nb: nbformat.NotebookNode, cell_idx: int, 
                    optimized_client: OptimizedNotebookClient = None) -> bool:
    """Execute cell and reflect on success/failure for autonomous adaptation"""
    try:
        if optimized_client is None:
            # Simple execution without full client
            from nbclient import NotebookClient
            import jupyter_client
            
            km = jupyter_client.KernelManager(kernel_name='python3')
            km.start_kernel()
            client = NotebookClient(nb, kernel_manager=km, timeout=60)
            
            try:
                client.execute_cell(nb.cells[cell_idx], cell_idx)
                has_errors, error_msg = has_execution_errors(nb.cells[cell_idx])
                km.shutdown_kernel()
                return not has_errors
            except Exception as e:
                km.shutdown_kernel()
                logger.error(f"Execution failed: {e}")
                return False
        else:
            # Use existing optimized client
            ok, err, outputs = optimized_client.execute_cell_smart(cell_idx, force_execution=True)
            return ok
            
    except Exception as e:
        logger.error(f"execute_and_reflect failed: {e}")
        return False



def adapt_plan_after_failure(current_tasks: List[str], failed_step_idx: int, 
                           user_request: str, error_context: str = "") -> List[str]:
    """Autonomously adapt the analysis plan after a failure"""
    
    try:
        failed_task = current_tasks[failed_step_idx] if failed_step_idx < len(current_tasks) else "Unknown"
        remaining_tasks = current_tasks[failed_step_idx + 1:]
        
        adaptation_prompt = """You are an autonomous data science agent. A step in your analysis plan failed. 
Adapt the remaining plan to work around this failure and still achieve the user's goals.

ORIGINAL USER REQUEST: {user_request}
FAILED STEP: {failed_task}
ERROR CONTEXT: {error_context}
REMAINING PLANNED STEPS: {remaining_tasks}

Create a new plan for the remaining steps that:
1. Works around the failure
2. Still addresses the user's original request
3. Uses simpler, more reliable approaches
4. Has 2-4 focused steps maximum

Reply with JSON: {{"adapted_plan": ["step1", "step2", ...], "reasoning": "why this approach will work"}}"""

        response = chat(
            messages=[{"role": "user", "content": adaptation_prompt}],
            model="gemma3n:e2b",
            format={"adapted_plan": ["string"], "reasoning": "string"},
            options={"temperature": 0.7, "num_ctx": 15000}
        )
        
        data = json.loads(response.message.content)
        adapted_tasks = data.get("adapted_plan", [])
        reasoning = data.get("reasoning", "")
        
        logger.info(f"🤔 Agent adapted plan: {reasoning}")
        logger.info(f"📋 New tasks: {adapted_tasks}")
        
        # Return adapted plan
        completed_tasks = current_tasks[:failed_step_idx]
        return completed_tasks + adapted_tasks
        
    except Exception as e:
        logger.error(f"Plan adaptation failed: {e}")
        # Fallback: simplified remaining plan
        return current_tasks[:failed_step_idx] + [
            "Create simple data summary",
            "Generate basic insights",
            "Provide conclusions"
        ]

# ================================
# FILE SYSTEM UTILITIES
# ================================

def discover_data_files(search_paths: List[Path] = None) -> Dict[str, List[str]]:
    """Discover available data files in common locations"""
    if search_paths is None:
        search_paths = [
            Path("."),  # Current directory
            Path("data"),
            Path("datasets"),
            Path("csv"),
        ]
    
    file_types = {
        "csv": ["*.csv"],
        "excel": ["*.xlsx", "*.xls"],
        "json": ["*.json"],
        "parquet": ["*.parquet"],
    }
    
    discovered_files = {}
    
    for search_path in search_paths:
        try:
            abs_search_path = search_path.resolve()
            
            if not abs_search_path.exists():
                continue
                
            logger.info(f"Searching for data files in: {abs_search_path}")
            
            for file_type, patterns in file_types.items():
                for pattern in patterns:
                    try:
                        found_files = list(abs_search_path.glob(pattern))
                        for f in found_files:
                            filename = f.name
                            
                            if file_type not in discovered_files:
                                discovered_files[file_type] = []
                            
                            if filename not in discovered_files[file_type]:
                                discovered_files[file_type].append(filename)
                                
                    except Exception as e:
                        logger.warning(f"Error searching for {pattern} in {abs_search_path}: {e}")
                        
        except Exception as e:
            logger.warning(f"Error processing search path {search_path}: {e}")
    
    return discovered_files

def get_working_directory_info() -> str:
    """Get information about the current working directory"""
    cwd = Path.cwd()
    info = f"Current working directory: {cwd}\n"
    info += "Files in current directory:\n"
    
    try:
        for file in sorted(cwd.iterdir()):
            if file.is_file() and file.suffix in ['.csv', '.xlsx', '.json', '.parquet']:
                info += f"  - {file.name}\n"
    except Exception as e:
        info += f"  Error listing files: {e}\n"
    
    return info

# ================================
# ENHANCED PROMPT TEMPLATES
# ================================
# In your ORIGINAL coding_agent.py, REPLACE this variable


CODING_AGENT_SYSTEM_PROMPT = """
You are an Intermediate-level data science agent but you like coding small in one response and keep things simple and short to avoid confusion while coding python for the python notebook. You MUST follow these rules.
YOU WILL be provided THE DATASET NAME IN THE PROMPT USE THAT NAME AT ALL COST AND THEN WRITE PYTHON CODE
**CRITICAL RULES (UNBREAKABLE):**
1.  **CONTEXT IS LAW:** You will be given the "CURRENT STATE" of the notebook. You MUST use the exact column names and variables from this context. Do not hallucinate column names.
2.  **ONE ACTION PER CELL:** Your code must accomplish only the SINGLE, specific task you are given.
3.  **IMPORTS ARE REQUIRED:** You MUST import all necessary libraries (`pandas`, `seaborn`, `sklearn`, etc.) in every cell where they are used. but for basic eda and all things pandas is enough
4.  **NEVER RELOAD DATA:** If the context shows `df` is already loaded, DO NOT add code to load it again (`pd.read_csv`).
5.  **OUTPUT FORMAT:** Respond ONLY with a valid JSON object with "cell" and "status" keys.

---
**DYNAMIC CONTEXT FOR THIS TASK:**
{dynamic_context}
---

**VISUALIZATION-SPECIFIC RULES:**
If the CURRENT TASK is a visualization, you MUST follow these additional rules:
Do NOT use any plots or visualizations like `sns.histplot`, `sns.boxplot`, `plt.show`, etc.

Avoid loops over DataFrame columns. If needed, show one or two feature summaries directly (manually written).

Keep the code minimal and textual: use `.describe()`, `.value_counts()`, and `.corr()`.

Focus on results printing and simple numeric/categorical inspection.
just write pass in code cell and skip it
**OUTPUT STRUCTURE:**
{{
  "cell": {{
    "markdown": "## A Clear Title for the Step\\n\\n- A brief explanation of what the code does.",
    "code": "# Production-quality code for this one specific step."
  }},
  "status": "continue"
}}
**GUIDELINES & BEST PRACTICES:**
- first import required basic libraries and also import libraries throughout task when required!!! got rule don't violate ever when writing the code
- task list always not so precise so don't totally depend on it you might have to make your own improvements to the code like adding libraries
* **VARIABLE REUSE:**
    * NEVER reload data (`pd.read_csv`) if the 'df' variable already exists. Assume it is loaded and ready.
    * Build upon existing variables (`df`, `X_train`, `y_train`, etc.).

- never write plotting code with for loops together unbreakable rules for example : (
# Create scatter plots to visualize relationships between features
for i in range(len(numerical_cols)):
    for j in range(i + 1, len(numerical_cols)):
        col1 = numerical_cols[i]
        col2 = numerical_cols[j]
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=df[col1], y=df[col2])
        plt.title(f'Scatter plot of col1 vs col2')
        plt.xlabel(col1)
        plt.ylabel(col2)
        plt.show()
) such code are not allowed to write.
this kind of python code is forbidden at all cost loops in plotting codes are banned for you
        
* **COLUMN HANDLING:**
    * If cleaning columns, use `df.columns = df.columns.str.strip()`.
    * Use `df.select_dtypes(include=['number'])` to find numeric columns dynamically.
    * Always check for a column's existence in `df.columns` before using it in a plot or calculation.

* **MACHINE LEARNING WORKFLOW:**
    * For classification, the target variable should be converted to binary (0/1).
    * Features (`X`) should typically be all numerical columns except the target.
    * Always use `sklearn.metrics` functions for evaluation (e.g., `accuracy_score`), NOT the `.evaluate()` method.

* **CODE QUALITY:**
    * Generate production-quality, error-free code.
    * Include `print()` statements to show the results of calculations or data manipulations.
    * Add comments to explain complex code.
    * When appropriate, use `plotly` for interactive visualizations.
"""
# In coding_agent.py, add this new prompt variable near the top

CODE_FIXER_SYSTEM_PROMPT = """
You are an expert-level AI code debugger. Your ONLY task is to fix a single block of Python code that has failed.

**CRITICAL RULES:**
1.  Analyze the provided "FAILED CODE" and "ERROR MESSAGE".
2.  Use the "SUCCESSFUL CODE HISTORY" to understand the context (what variables exist, etc.).
3.  Rewrite the code to fix the specific error.
4.  You MUST respond with ONLY a valid JSON object. Do not add any text, explanations, or markdown outside of the JSON structure. Your entire response must be the JSON object itself.

**EXAMPLE OF A PERFECT RESPONSE:**
The user provides a failed attempt to convert a column with string 'M' to an integer.
YOUR RESPONSE MUST BE:
```json
{
  "cell": {
    "markdown": "## Corrected Data Type Conversion\\n\\n- The 'diagnosis' column contains strings ('M', 'B'), which cannot be directly converted to integers. The fix is to map these strings to numerical values (M=1, B=0) before performing numerical operations.",
    "code": "df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})\\n# Now that the column is numeric, subsequent operations will work.\\nprint(df.head())"
  },
  "status": "continue"
}
Now, apply this logic to the user's request.
"""

SYNTAX_FIXER_PROMPT = """
You are a Python syntax correction bot. Your only task is to fix the syntax of the provided code snippet. Do not change the logic. Do not add new features. Only fix syntax errors like missing commas, incorrect indentation, or unclosed parentheses.

Respond ONLY with a valid JSON object containing the corrected code.

**EXAMPLE:**
USER PROVIDES:
- Code: "print('hello)"
- Error: "SyntaxError: EOL while scanning string literal"

YOUR RESPONSE MUST BE:
```json
{
  "cell": {
    "markdown": "## Corrected Syntax",
    "code": "print('hello')"
  },
  "status": "continue"
}
"""

# ================================
# ENHANCED MEMORY MANAGER
# ================================

class AgentMemoryManager:
    """Enhanced memory manager with execution state tracking"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.memory_file = AGENT_MEMORY_DIR / f"{session_id}_memory.json"
        
        try:
            self.embeddings = OllamaEmbeddings(model="nomic-embed-text:v1.5")
        except Exception as e:
            logger.warning(f"Failed to initialize embeddings: {e}")
            self.embeddings = None
        
        self.vector_store = self._initialize_faiss_store()
        self.session_data = self._load_session_memory()
        self.available_files = discover_data_files()
        self.working_dir_info = get_working_directory_info()
        
        # Track notebook execution state
        self.executed_variables = set()
        self.notebook_progress = {}
    
    def _initialize_faiss_store(self) -> Optional[FAISS]:
        """Initialize FAISS vector store"""
        if not self.embeddings:
            return None
            
        try:
            embedding_dim = 768
            index = faiss.IndexFlatL2(embedding_dim)
            
            vector_store = FAISS(
                embedding_function=self.embeddings,
                index=index,
                docstore=InMemoryDocstore({}),
                index_to_docstore_id={}
            )
            logger.info(f"Created new FAISS index for session {self.session_id}")
            return vector_store
        except Exception as e:
            logger.error(f"Error initializing FAISS store: {e}")
            return None
    
    def _load_session_memory(self) -> SessionMemory:
        """Load session memory from disk"""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return SessionMemory(**data)
            except Exception as e:
                logger.error(f"Error loading session memory: {e}")
        
        return SessionMemory(session_id=self.session_id)
    
    def update_execution_state(self, variables: List[str], step: str, outputs: str):
        """Update the execution state with new variables and outputs"""
        self.executed_variables.update(variables)
        self.notebook_progress[step] = {
            "variables": variables,
            "outputs": outputs[:500],  # Store first 500 chars of outputs
            "timestamp": datetime.now().isoformat()
        }
    
    def get_execution_context(self) -> str:
        """Get context about what has been executed so far"""
        context = f"Available Files: {', '.join(self.available_files.get('csv', []))}\n"
        context += f"Executed Variables: {', '.join(sorted(self.executed_variables))}\n"
        
        if self.notebook_progress:
            context += "Previous Steps:\n"
            for step, info in self.notebook_progress.items():
                context += f"  {step}: Variables created: {', '.join(info['variables'])}\n"
                if info['outputs']:
                    context += f"    Output sample: {info['outputs'][:100]}...\n"
        
        return context
    
    def get_file_context(self) -> str:
        """Get context about available files"""
        context = f"{self.working_dir_info}\n"
        context += "Available data files:\n"
        
        for file_type, files in self.available_files.items():
            if files:
                context += f"  {file_type.upper()}: {', '.join(files)}\n"
        
        return context
    
    def add_interaction(self, step: str, user_input: str, agent_output: str, 
                    notebook_context: Dict[str, Any] = None):
        """Add interaction to memory"""
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "user_input": user_input,
            "agent_output": agent_output,
            "notebook_context": notebook_context or {},
            "execution_state": self.get_execution_context()
        }
        
        self.session_data.messages.append(interaction)
        self.session_data.last_updated = datetime.now().isoformat()
        
        if len(self.session_data.messages) > MAX_MEMORY_MESSAGES:
            self.session_data.messages = self.session_data.messages[-MAX_MEMORY_MESSAGES:]
        
        if self.vector_store:
            try:
                memory_text = f"Step {step}: {user_input}\nOutput: {agent_output}"
                self.vector_store.add_texts([memory_text])
            except Exception as e:
                logger.warning(f"Failed to add to vector store: {e}")
        
        self._save_session_memory()
    
    def get_relevant_context(self, query: str, k: int = 3) -> str:
        """Get relevant context from memory"""
        try:
            if self.vector_store:
                docs = self.vector_store.similarity_search(query, k=k)
                context = []
                for doc in docs:
                    context.append(doc.page_content)
                return "\n".join(context)
        except Exception as e:
            logger.warning(f"Error retrieving context: {e}")
        
        # Fallback to recent messages
        recent_messages = self.session_data.messages[-2:]
        context = []
        for msg in recent_messages:
            context.append(f"Step {msg['step']}: {msg['user_input'][:100]}...")
        return "\n".join(context)
    
    def _save_session_memory(self):
        """Save session memory to disk"""
        try:
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(self.session_data.model_dump(), f, indent=2, ensure_ascii=False)
            logger.debug(f"Saved session memory for {self.session_id}")
        except Exception as e:
            logger.error(f"Error saving session memory: {e}")

# ================================
# VARIABLE TRACKING
# ================================

class VariableTracker:
    """Track variable dependencies across notebook cells."""
    
    def __init__(self):
        self.cell_variables = {}
        self.variable_sources = {}
        self.all_defined_variables = set()
        
    def analyze_cell(self, cell_idx: int, code: str):
        """Analyze what variables a cell defines and uses."""
        try:
            tree = ast.parse(code)
            defined = set()
            used = set()
            
            class VariableVisitor(ast.NodeVisitor):
                def visit_Assign(self, node):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            defined.add(target.id)
                        elif isinstance(target, ast.Tuple) or isinstance(target, ast.List):
                            for elt in target.elts:
                                if isinstance(elt, ast.Name):
                                    defined.add(elt.id)
                    self.generic_visit(node)
                
                def visit_AnnAssign(self, node):
                    if isinstance(node.target, ast.Name):
                        defined.add(node.target.id)
                    self.generic_visit(node)
                
                def visit_Name(self, node):
                    if isinstance(node.ctx, ast.Load):
                        used.add(node.id)
                    self.generic_visit(node)
                
                def visit_FunctionDef(self, node):
                    defined.add(node.name)
                    self.generic_visit(node)
                
                def visit_ClassDef(self, node):
                    defined.add(node.name)
                    self.generic_visit(node)
            
            visitor = VariableVisitor()
            visitor.visit(tree)
            
            # Filter out built-ins and common imports
            builtin_names = {'print', 'len', 'str', 'int', 'float', 'list', 'dict', 
                            'set', 'tuple', 'range', 'enumerate', 'zip', 'map', 'filter',
                            'pandas', 'pd', 'numpy', 'np', 'matplotlib', 'plt', 'sklearn',
                            'seaborn', 'sns', 'warnings', 'os', 'sys', 'Exception', 
                            'FileNotFoundError', 'e', 'exit'}
            used = used - builtin_names
            
            self.cell_variables[cell_idx] = {"defined": defined, "used": used}
            self.all_defined_variables.update(defined)
            
            for var in defined:
                self.variable_sources[var] = cell_idx
                
            logger.debug(f"Cell {cell_idx}: defined={defined}, used={used}")
                
        except SyntaxError as e:
            logger.warning(f"Syntax error in cell {cell_idx}: {e}")
            self.cell_variables[cell_idx] = {"defined": set(), "used": set()}
            return False
        return True
    
    def get_available_variables(self) -> List[str]:
        """Get all variables that have been defined in executed cells"""
        return sorted(list(self.all_defined_variables))
    
    def has_variable(self, var_name: str) -> bool:
        """Check if a variable has been defined"""
        return var_name in self.all_defined_variables
    
    def get_dependencies(self, cell_idx: int) -> List[int]:
        """Get cells that need to be executed before this cell."""
        if cell_idx not in self.cell_variables:
            return []
        
        used_vars = self.cell_variables[cell_idx]["used"]
        dependencies = []
        
        for var in used_vars:
            if var in self.variable_sources:
                dep_cell = self.variable_sources[var]
                if dep_cell < cell_idx:
                    dependencies.append(dep_cell)
        
        return sorted(set(dependencies))

# ================================
# NOTEBOOK MANAGEMENT
# ================================

def load_or_create_notebook(chat_id: str) -> nbformat.NotebookNode:
    """Load an existing notebook or create a new one inside its session folder."""
    # --- START OF FIX ---
    session_dir = NOTEBOOKS_DIR / chat_id
    session_dir.mkdir(exist_ok=True)
    nb_path = session_dir / f"{chat_id}.ipynb"
    # --- END OF FIX ---

    if nb_path.exists():
        try:
            with open(nb_path, 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)
            logger.info(f"Loaded existing notebook for chat {chat_id}")
            return nb
        except Exception as e:
            logger.error(f"Error reading notebook {nb_path}, creating a new one. Error: {e}")
    
    logger.info(f"Creating a new notebook for chat {chat_id}")
    nb = new_notebook()
    nb.metadata["name"] = f"Analysis Session {chat_id}"
    return nb

def save_notebook(chat_id: str, nb: nbformat.NotebookNode):
    """Saves the notebook inside its session folder."""
    # --- START OF FIX ---
    session_dir = NOTEBOOKS_DIR / chat_id
    session_dir.mkdir(exist_ok=True)
    nb_path = session_dir / f"{chat_id}.ipynb"
    # --- END OF FIX ---
    nbformat.write(nb, str(nb_path))
    logger.info(f"Notebook saved for chat {chat_id} at {nb_path}")

def clean_notebook_for_fresh_start(nb: nbformat.NotebookNode, chat_id: str, chat_context: Dict[str, Any] = None) -> nbformat.NotebookNode:
    """
    Creates a completely clean, empty notebook. NO automatic data loading.
    """
    nb.cells = []
    logger.info("Starting with a completely clean notebook as commanded.")
    save_notebook(chat_id, nb)
    return nb

# ================================
# OPTIMIZED NOTEBOOK CLIENT
# ================================

class OptimizedNotebookClient:
    """Optimized notebook client with variable state tracking."""
    
    def __init__(self, nb, timeout=600):
        self.nb = nb
        self.timeout = timeout
        self.client = None
        self.km = None
        self._kernel_started = False
        self.executed_cells = set(nb.metadata.get("executed_cell_indices", []))
        self.variable_tracker = VariableTracker()
        self.execution_order = []
        self.failed_cells = set()
        self.cell_outputs_history = {}
        self.chat_context = {}  # Store chat context
        self._analyze_existing_cells()  # This was failing
        
    def _analyze_existing_cells(self):
        """Analyze existing cells in the notebook for variable dependencies."""
        for idx, cell in enumerate(self.nb.cells):
            if cell.cell_type == "code" and cell.source.strip():
                syntax_ok = self.variable_tracker.analyze_cell(idx, cell.source)
                if not syntax_ok:
                    self.failed_cells.add(idx)
                    logger.error(f"Cell {idx} has syntax errors and is marked as failed")
        
    # Update start_kernel to use context
    def start_kernel(self):
        """
        Start the kernel with the working directory set to the notebook's session folder.
        """
        try:
            # --- START OF THE FIX ---
            # Determine the session directory from the notebook's path metadata
            session_dir = Path(self.nb.metadata.get("path", ".")).parent.resolve()
            logger.info(f"Setting kernel working directory to: {session_dir}")

            # Start the kernel with the corrected working directory
            self.km = KernelManager(kernel_name='python3')
            self.km.start_kernel(cwd=str(session_dir)) # This is the critical change
            # --- END OF THE FIX ---

            kc = self.km.client()
            kc.start_channels()

            try:
                kc.wait_for_ready(timeout=60)
            except RuntimeError:
                logger.error("Kernel failed to become ready")
                self._cleanup_kernel()
                return False

            self.client = NotebookClient(
                self.nb,
                timeout=self.timeout,
                kernel_manager=self.km,
                allow_errors=True
            )
            self.client.kc = kc
            self.client.km = self.km
            self._kernel_started = True
            logger.info("Optimized kernel started successfully in the correct directory.")
            return True

        except Exception as e:
            logger.error(f"Failed to start kernel: {e}")
            self._cleanup_kernel()
            return False
    
    def _fix_cell_source_for_context(self, source: str, chat_context: Dict[str, Any], cell_idx: int) -> str:
        """Fix cell source code to use correct context (filename, etc.)"""
        if not chat_context:
            return source

        fixed_source = source
        dataset_name = chat_context.get('dataset_name', '')

        # Fix common placeholder patterns
        if dataset_name and cell_idx <= 2:  # Only fix early cells
            # Use regex for more precise replacements
            import re

            # Replace quoted placeholders first
            patterns = [
                (r"'your_dataset\.csv'", f"'{dataset_name}'"),
                (r'"your_dataset\.csv"', f"'{dataset_name}'"),
                (r"'dataset\.csv'", f"'{dataset_name}'"),
                (r'"dataset\.csv"', f"'{dataset_name}'"),
                (r"'data\.csv'", f"'{dataset_name}'"),
                (r'"data\.csv"', f"'{dataset_name}'"),
                (r"'filename\.csv'", f"'{dataset_name}'"),
                (r'"filename\.csv"', f"'{dataset_name}'")
            ]

            for pattern, replacement in patterns:
                if re.search(pattern, fixed_source):
                    fixed_source = re.sub(pattern, replacement, fixed_source)
                    logger.info(f"Fixed filename pattern: {pattern} -> {replacement}")
                    break  # Only replace first match to avoid multiple replacements

        return fixed_source

    
    def _restore_kernel_state(self, chat_context: Dict[str, Any] = None) -> bool:
        """Re-execute previously executed cells with error handling and context fixes"""
        if not self.executed_cells:
            return True

        logger.info(f"Restoring kernel state by re-executing {len(self.executed_cells)} cells")

        sorted_executed = sorted(self.executed_cells)

        for cell_idx in sorted_executed:
            if cell_idx < len(self.nb.cells):
                cell = self.nb.cells[cell_idx]

                if cell.cell_type == "code" and cell.source.strip():
                    try:
                        # **NEW: Fix problematic cells before execution**
                        fixed_source = self._fix_cell_source_for_context(cell.source, chat_context, cell_idx)

                        if fixed_source != cell.source:
                            logger.info(f"Fixed cell {cell_idx} source for context compatibility")
                            original_source = cell.source
                            cell.source = fixed_source

                        self.client.execute_cell(cell, cell_idx)

                        has_errors, error_msg = has_execution_errors(cell)
                        if has_errors:
                            logger.error(f"Error during state restoration: {error_msg}")

                            # **NEW: Try to skip non-critical errors during restoration**
                            if "FileNotFoundError" in error_msg and cell_idx == 1:  # First code cell
                                logger.warning("Skipping first cell due to FileNotFoundError - will regenerate")
                                continue
                            else:
                                self.failed_cells.add(cell_idx)
                                return False

                        outputs = parse_cell_outputs(cell)
                        self.cell_outputs_history[cell_idx] = outputs
                        logger.debug(f"Restored state from cell {cell_idx}")

                    except Exception as e:
                        logger.error(f"Failed to restore state: {e}")

                        # **NEW: Skip problematic cells during restoration**
                        if "FileNotFoundError" in str(e) and cell_idx <= 2:
                            logger.warning(f"Skipping cell {cell_idx} due to file not found - will regenerate")
                            continue
                        else:
                            self.failed_cells.add(cell_idx)
                            return False

        return True

    
    def execute_cell_smart(self, cell_index: int, force_execution: bool = False) -> Tuple[bool, Optional[str], List]:
        """Execute a cell with robust error checking."""
        if not self._kernel_started or not self.client or not self.client.kc:
            return False, "Kernel not available", []

        if cell_index >= len(self.nb.cells):
            return False, "Cell index out of range", []

        cell = self.nb.cells[cell_index]
        if cell.cell_type != "code":
            return True, None, []

        if cell_index in self.failed_cells and not force_execution:
            return False, "Cell previously failed", []

        if not force_execution and cell_index in self.executed_cells:
            return True, None, self.cell_outputs_history.get(cell_index, [])

        if cell.source.strip():
            syntax_ok = self.variable_tracker.analyze_cell(cell_index, cell.source)
            if not syntax_ok:
                self.failed_cells.add(cell_index)
                return False, "Syntax error in cell", []

        try:
            logger.info(f"Executing cell {cell_index}")

            cell.outputs = []

            if cell_index in self.failed_cells:
                self.failed_cells.remove(cell_index)

            self.client.execute_cell(cell, cell_index)

            has_errors, error_msg = has_execution_errors(cell)
            if has_errors:
                logger.error(f"Execution error in cell {cell_index}: {error_msg}")
                self.failed_cells.add(cell_index)
                return False, error_msg, []

            # Cell executed successfully
            self.executed_cells.add(cell_index)
            self.execution_order.append(cell_index)

            self.nb.metadata["executed_cell_indices"] = list(self.executed_cells)

            cell.metadata["executed"] = True
            cell.metadata["execution_time"] = time.time()

            # Check if this was a visualization cell and replace complex outputs
            if detect_visualization_code(cell.source):
                outputs = create_visualization_validation_output()
                logger.info(f"📊 Visualization cell detected - simplified output for memory efficiency")
            else:
                outputs = parse_cell_outputs(cell)

            self.cell_outputs_history[cell_index] = outputs
            logger.info(f"✅ Successfully executed cell {cell_index}")
            return True, None, outputs

        except Exception as e:
            error_msg = traceback.format_exc()
            logger.error(f"Exception executing cell {cell_index}: {error_msg}")
            self.failed_cells.add(cell_index)
            return False, error_msg, []

    def get_recent_outputs_context(self, num_recent: int = 2) -> str:
        """Get context from recent cell outputs."""
        recent_outputs = ""
        recent_cells = sorted(self.cell_outputs_history.keys())[-num_recent:]
        
        for cell_idx in recent_cells:
            outputs = self.cell_outputs_history[cell_idx]
            for output in outputs:
                if output.content and len(output.content.strip()) > 0:
                    recent_outputs += f"Cell {cell_idx}: {output.content[:200]}...\n"
        
        return recent_outputs
    
    def has_failed_cells(self) -> bool:
        return len(self.failed_cells) > 0
    
    def get_failed_cells(self) -> Set[int]:
        return self.failed_cells.copy()
    
    def _cleanup_kernel(self):
        """Internal method to clean up kernel resources."""
        try:
            if self.client and hasattr(self.client, 'kc'):
                if self.client.kc:
                    self.client.kc.stop_channels()
                    self.client.kc = None
            
            if self.km:
                if self.km.is_alive():
                    self.km.shutdown_kernel()
                self.km = None
                
        except Exception as e:
            logger.error(f"Error during kernel cleanup: {e}")
    
    def shutdown(self):
        """Safely shutdown the kernel."""
        try:
            if self._kernel_started:
                self._cleanup_kernel()
                logger.info("Optimized kernel shutdown successfully")
        except Exception as e:
            logger.error(f"Error shutting down kernel: {e}")
        finally:
            self._kernel_started = False
            self.client = None

# ================================
# HELPER FUNCTIONS
# ================================

def generate_analysis_conclusion(nb: nbformat.NotebookNode, user_request: str, dataset_name: str, memory_manager: AgentMemoryManager) -> Optional[str]:
    """
    Makes a final LLM call to summarize the key findings from the analysis.
    This version is robust against non-string data in memory.
    """
    summary_outputs = []
    # Iterate through all messages in the agent's memory
    for msg in memory_manager.session_data.messages:
        context = msg.get("notebook_context", {})
        # Check if the step was successful and produced an output
        if context.get("success") and context.get("output"):
            output = context["output"]
            
            # --- THIS IS THE CRITICAL FIX ---
            # Ensure the output is a string before appending.
            # This handles cases where the output might be a list, dict, or None.
            if isinstance(output, str):
                # Only include non-empty, non-visualization outputs
                if output.strip() and "Visualization created" not in output:
                    summary_outputs.append(output[:300]) # Limit length for context
            # If the output is not a string, it is safely ignored.

    if not summary_outputs:
        return "The analysis was performed, but no textual output was generated to summarize."

    # Now, full_context is guaranteed to be a list of strings and will not crash .join()
    full_context = "\n".join(summary_outputs)

    summary_prompt = """
You are a data scientist summarizing the results of a notebook execution for a user.
The user's original request was: "{user_request}"
The dataset analyzed was: "{dataset_name}"

Below are the key outputs from the executed code cells.
Synthesize these outputs into a concise, easy-to-understand summary of the key findings.
Do not mention code or technical steps. Start with the main conclusion.

--- COLLECTED OUTPUTS ---
{full_context[:10000]}
--- END OF OUTPUTS ---

Provide the final summary in Markdown format.
"""
    try:
        logger.info("Generating final analysis conclusion...")
        response = chat(
            model="gemma3n:e2b",
            messages=[{"role": "user", "content": summary_prompt}],
        )
        return response['message']['content']
    except Exception as e:
        logger.error(f"Could not generate final conclusion: {e}")
        return "The analysis is complete, but an error occurred while generating the final summary."


def generate_code_correction_prompt(
    task_description: str,
    nb: nbformat.NotebookNode,
    failed_code: str,
    error_message: str
) -> list:
    """
    Generates a focused prompt for the "Code Fixer" model, including
    successful history, the failed code, and the error.
    """
    # Build a summary of the successfully executed cells
    successful_history = []
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'code' and not cell.get('metadata', {}).get('error'):
            output_preview = str(cell.get('outputs', [])[:150]) + "..."
            successful_history.append(f"### Cell {i+1} (Success)\n```python\n{cell.source}\n```\nOUTPUT PREVIEW: {output_preview}\n")

    successful_history_str = "\n".join(successful_history)
    if not successful_history_str:
        successful_history_str = "No cells have been executed successfully yet."

    # Construct the user message for the fixer prompt
    user_message = """
**SUCCESSFUL CODE HISTORY:**
{successful_history_str}
---
**ORIGINAL TASK:**
{task_description}
---
**THE FAILED CODE:**
```python
{failed_code}
```
---
**THE ERROR MESSAGE:**
{error_message}
```
Now, please provide the corrected code in the required JSON format to fix the error and accomplish the original task.

"""
    return [{"role": "system", "content": CODE_FIXER_SYSTEM_PROMPT}, {"role": "user", "content": user_message}]


def get_actual_column_context(df_variable_available: bool) -> str:
    """Get actual column names from the current DataFrame"""
    if not df_variable_available:
        return "DataFrame not loaded yet"
    
    try:
        # This would need to be executed in the kernel context
        return "Use df.columns.tolist() to get actual column names"
    except:
        return "Column information unavailable"
# In coding_agent.py, add these two missing functions

def load_chat_history_from_file(chat_id: str) -> List[Dict[str, Any]]:
    """Loads a specific chat history from its JSON file."""
    path = CHAT_HISTORY_DIR / f"{chat_id}.json"
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return []

def save_chat_history_to_file(chat_id: str, history: List[Dict[str, Any]]):
    """Saves a specific chat history to its file."""
    path = CHAT_HISTORY_DIR / f"{chat_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

def detect_visualization_code(code: str) -> bool:
    """Detect if code contains visualization commands"""
    viz_keywords = ['plt.', 'sns.', 'matplotlib', 'seaborn', 'plot(', 'hist(', 'scatter(', 'show()', 'figure(']
    return any(keyword in code.lower() for keyword in viz_keywords)

def create_visualization_validation_output():
    """Create simple success output for visualizations"""
    return [NotebookCellOutput(type="stream", content="Visualization created successfully")]

# In coding_agent.py, REPLACE the entire execute_planned_task_with_memory function with this one.

def execute_planned_task_with_memory(chat_id: str, tasks: list, dataset_name: str, memory_manager: AgentMemoryManager):
    """
    Executes tasks with a multi-layered retry strategy and robust final summary generation.
    """
    logger.info(f"Starting task execution for chat_id: {chat_id}")
    history_path = CHAT_HISTORY_DIR / f"{chat_id}.json"
    knowledge_base = CodeKnowledgeBase()
    client = None
    nb = None
    last_step_output = "No output from previous step."
    last_step_error = ""
    last_failed_code = ""
    all_tasks_succeeded = True

    try:
        chat_history_context = get_chat_history_context(history_path)
        chat_history_context['dataset_name'] = Path(dataset_name).name
        nb, client = setup_notebook_and_kernel(chat_id)

        for i, task in enumerate(tasks):
            step_number = i + 1
            logger.info(f"--- Starting Task {step_number}/{len(tasks)}: {task} ---")
            task_success = False

            for attempt in range(1, MAX_CODE_RETRIES + 2):
                update_progress(history_path, [], int(((step_number - 1) / len(tasks)) * 100), status=f"Task {step_number}, Attempt {attempt}: {task[:40]}...")
                logger.info(f"Attempt {attempt} for task: '{task}'")

                agent_response = None
                messages = []

                if attempt == 1:
                    messages = generate_next_cell_prompt(task, nb, chat_history_context, client, last_step_output, last_step_error)
                    agent_response = generate_code_from_messages(messages)
                elif attempt == 2:
                    logger.info("Attempt 1 failed. Searching knowledge base...")
                    kb_solution = knowledge_base.search(f"{task} - Error: {last_step_error}")
                    if kb_solution:
                        logger.info(f"Found relevant solution in KB: '{kb_solution['description']}'")
                        code_to_execute = kb_solution['code'].replace('{dataset_name}', dataset_name)
                        markdown = f"## Applying Trusted Solution: {kb_solution['description']}"
                        agent_response = AgentOutput(cell=GeneratedCell(markdown=markdown, code=code_to_execute), status="continue")
                    else:
                        logger.warning("No KB solution found. Escalating to Code Fixer.")
                        continue
                else: # attempt == 3
                    logger.info("Attempt 2 failed. Escalating to Code Fixer...")
                    messages = generate_code_correction_prompt(task, nb, last_failed_code, last_step_error)
                    agent_response = generate_code_from_messages(messages)

                if agent_response is None:
                    last_step_error = "LLM failed to generate a valid response."
                    continue

                append_cells(nb, agent_response.cell.markdown, agent_response.cell.code)
                cell_index = len(nb.cells) - 1
                success, output, error = execute_cell_smart(client, nb, cell_index)

                memory_manager.add_interaction(
                    step=f"Step {step_number} (Attempt {attempt})",
                    user_input=f"Task: {task}", agent_output=agent_response.cell.code,
                    notebook_context={"success": success, "error": error, "output": output}
                )

                if success:
                    logger.info(f"✅ Task '{task}' SUCCEEDED on attempt {attempt}.")
                    task_success = True
                    last_step_output = output
                    last_step_error = ""
                    save_notebook(chat_id, nb)
                    break
                else:
                    logger.warning(f"⚠️ Attempt {attempt} for task '{task}' FAILED.")
                    last_step_error = error
                    last_failed_code = agent_response.cell.code
                    save_notebook(chat_id, nb)

            if not task_success:
                all_tasks_succeeded = False
                final_error_message = f"Task '{task}' FAILED after all attempts."
                logger.error(final_error_message)
                append_single_markdown_cell(nb, f"## Analysis Stopped\n\nThe task '{task}' failed after {MAX_CODE_RETRIES + 1} attempts.\n\n**Final Error:**\n```\n{last_step_error}\n```")
                break

        # --- THIS BLOCK IS NOW OUTSIDE THE MAIN TRY/EXCEPT FOR TASK FAILURES ---
        if all_tasks_succeeded:
            logger.info("✅ All tasks completed successfully. Generating final analysis summary...")
            status_message = "completed: Analysis finished."
        else:
            logger.info("⚠️ Some tasks failed. Generating summary of completed steps...")
            status_message = f"failed: Analysis stopped at task '{task}'."

        try:
            conclusion = generate_analysis_conclusion(nb, chat_history_context.get('user_query', ''), dataset_name, memory_manager)
            if conclusion:
                summary_title = "Final Analysis Summary" if all_tasks_succeeded else "Summary Before Failure"
                append_single_markdown_cell(nb, f"## {summary_title}\n\n{conclusion}")
                
                final_summary_obj = {"conclusion": conclusion, "timestamp": datetime.now().isoformat()}
                history = load_chat_history_from_file(chat_id)
                history.append(final_summary_obj)
                save_chat_history_to_file(chat_id, history)
        except Exception as e:
            logger.error(f"Failed to generate conclusion, but tasks were completed. Error: {e}")
            # Still report success if the tasks were done
            status_message = "completed: Tasks finished, but summary failed."

        update_progress(history_path, [], 100, status=status_message)

    except Exception as e:
        error_message = f"FATAL ERROR DURING TASK EXECUTION: {str(e)}"
        logger.error(error_message, exc_info=True)
        update_progress(history_path, [], 100, status=f"failed: {error_message}")
    finally:
        logger.info("Executing cleanup phase.")
        if client and nb:
            save_notebook(chat_id, nb)
        if client:
            shutdown(client, chat_id)
        logger.info(f"Agent work for chat {chat_id} is complete.")


def execute_code_reliably(code_to_execute: str, setup_code: str = "") -> tuple[bool, str, list]:
    """
    Creates an isolated kernel, executes setup code and the target code,
    captures the output, and shuts down. This prevents all state mismatches.
    """
    nb = new_notebook()
    # Add setup code (like loading the dataframe) first, but don't show its output
    if setup_code:
        nb.cells.append(new_code_cell(setup_code))
    
    # Add the actual code to be executed
    nb.cells.append(new_code_cell(code_to_execute))

    km = KernelManager(kernel_name='python3')
    try:
        km.start_kernel()
        client = NotebookClient(nb, km=km, timeout=300, allow_errors=True)
        
        # Execute the setup cell silently
        if setup_code:
            client.execute_cell(nb.cells[0], 0)
        
        # Execute the target cell
        client.execute_cell(nb.cells[-1], len(nb.cells) - 1)
        
    finally:
        if km.is_alive():
            km.shutdown_kernel()

    # Process results from the target cell
    target_cell = nb.cells[-1]
    error = None
    outputs = []
    for out in target_cell.get("outputs", []):
        if out.output_type == "error":
            error = "\n".join(out.get("traceback", []))
        elif out.output_type == "stream":
            outputs.append(out.get("text", ""))
        elif out.output_type == "execute_result":
            outputs.append(out.get("data", {}).get("text/plain", ""))
    
    if error:
        return False, error, outputs
    return True, "", outputs

def debug_agent_memory(chat_id: str):
    """Debug agent memory to see what's happening"""
    
    memory_dir = Path("agent_memory")
    memory_file = memory_dir / f"agent_{chat_id}.json"
    
    logger.info(f"🔍 Checking agent memory:")
    logger.info(f"📁 Memory directory exists: {memory_dir.exists()}")
    logger.info(f"📄 Memory file path: {memory_file}")
    logger.info(f"📄 Memory file exists: {memory_file.exists()}")
    
    if memory_file.exists():
        try:
            content = memory_file.read_text(encoding="utf-8")
            logger.info(f"📊 Memory file size: {len(content)} characters")
            
            data = json.loads(content)
            logger.info(f"📊 Memory entries: {len(data.get('memory_entries', []))}")
            
            # Show last few entries
            entries = data.get('memory_entries', [])
            if entries:
                logger.info(f"🔍 Last entry: {entries[-1].get('step_type', 'unknown')}")
            else:
                logger.warning("⚠️ Memory file exists but no entries found")
                
        except Exception as e:
            logger.error(f"❌ Error reading memory file: {e}")
    else:
        logger.warning("⚠️ Agent memory file does not exist")
        
        # List all files in agent_memory directory
        if memory_dir.exists():
            files = list(memory_dir.glob("*.json"))
            logger.info(f"📁 Files in agent_memory: {[f.name for f in files]}")
        else:
            logger.warning("⚠️ agent_memory directory doesn't exist")

def should_terminate_analysis(step_count: int, analysis_counts: Dict[str, int], 
                            completed_analysis: Dict[str, bool]) -> Tuple[bool, str]:
    """Check if analysis should terminate early"""
    
    # Basic analysis completion criteria
    basic_done = (
        completed_analysis.get('data_loaded', False) and
        completed_analysis.get('missing_values_checked', False) and
        completed_analysis.get('basic_stats_calculated', False)
    )
    
    # Advanced analysis completion
    advanced_done = (
        completed_analysis.get('visualizations_created', False) or
        completed_analysis.get('correlations_analyzed', False)
    )
    
    # Check for excessive repetition
    total_repetitions = sum(count for count in analysis_counts.values() if count >= 2)
    
    # Termination conditions
    if step_count >= 20 and basic_done and advanced_done:
        return True, "Analysis complete: All major steps finished"
    elif step_count >= 15 and total_repetitions >= 3:
        return True, "Analysis complete: Excessive repetition detected"
    elif step_count >= 25:
        return True, "Analysis complete: Maximum steps reached"
    
    return False, f"Analysis continuing: {step_count} steps completed"

def get_dataset_summary(dataset_name: str) -> Dict[str, Any]:
    """Quick dataset analysis for task planning"""
    try:
        import pandas as pd
        df = pd.read_csv(dataset_name)
        df.columns = df.columns.str.strip()
        
        return {
            'name': dataset_name,
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'sample_data': df.head(3).to_dict('records'),
            'missing_values': df.isnull().sum().to_dict(),
            'numeric_cols': df.select_dtypes(include=['number']).columns.tolist(),
            'categorical_cols': df.select_dtypes(include=['object']).columns.tolist()
        }
    except Exception as e:
        logger.error(f"Dataset analysis failed: {e}")
        return {'name': dataset_name, 'status': 'analysis_failed'}




def append_cells(nb: nbformat.NotebookNode, markdown_text: str, code_text: str):
    """Append markdown and code cells with validation"""
    
    # Clean markdown text - ensure no code blocks
    cleaned_markdown = markdown_text.strip()
    if "```" in cleaned_markdown:
        lines = cleaned_markdown.split('\n')
        cleaned_lines = []
        in_code_block = False
        
        for line in lines:
            if line.strip().startswith('```'):
                in_code_block = not in_code_block
                continue
            if not in_code_block:
                cleaned_lines.append(line)
        
        cleaned_markdown = '\n'.join(cleaned_lines).strip()
    
    # Ensure code has actual content
    cleaned_code = textwrap.dedent(code_text).strip()
    
    if not cleaned_code or (cleaned_code.startswith('#') and len(cleaned_code.split('\n')) == 1):
        logger.warning("Code cell would be empty or only comments, using placeholder")
        cleaned_code = "# Code execution placeholder\npass"
    
    # Create cells
    md_cell = new_markdown_cell(cleaned_markdown)
    code_cell = new_code_cell(cleaned_code)
    
    nb.cells.append(md_cell)
    nb.cells.append(code_cell)
    
    logger.info(f"Added cells - Markdown: {len(cleaned_markdown)} chars, Code: {len(cleaned_code)} chars")

def execute_code_in_kernel(self, code: str) -> Tuple[bool, Optional[str], List]:
    """Execute code directly in kernel and return results"""
    if not self._kernel_started or not self.client:
        return False, "Kernel not available", []
    
    try:
        # Create temporary cell
        temp_cell = nbformat.v4.new_code_cell(code)
        temp_cell.outputs = []
        
        # Execute
        self.client.execute_cell(temp_cell, -1)
        
        # Check for errors
        has_errors, error_msg = has_execution_errors(temp_cell)
        if has_errors:
            return False, error_msg, []
        
        # Parse outputs
        outputs = parse_cell_outputs(temp_cell)
        return True, None, outputs
        
    except Exception as e:
        return False, str(e), []


def append_single_markdown_cell(nb: nbformat.NotebookNode, markdown_text: str):
    """Append only a markdown cell"""
    cleaned_markdown = markdown_text.strip()
    md_cell = new_markdown_cell(cleaned_markdown)
    nb.cells.append(md_cell)

def get_chat_history_context(history_path: Path) -> Dict[str, Any]:
    """Extract relevant context from chat history"""
    try:
        with open(history_path, 'r', encoding='utf-8') as f:
            history = json.load(f)
        
        context = {
            "user_query": "",
            "dataset_name": "",
            "specific_requests": []
        }
        
        # Extract user query and specific requests
        for entry in history:
            if "user" in entry and "query" in entry["user"]:
                context["user_query"] = entry["user"]["query"]
                
                # Extract dataset name from query
                if ".csv" in context["user_query"]:
                    import re
                    csv_match = re.search(r'(\w+\.csv)', context["user_query"])
                    if csv_match:
                        context["dataset_name"] = csv_match.group(1)
                
                # Extract specific analysis requests
                context["specific_requests"].append(context["user_query"])
        
        return context
        
    except Exception as e:
        logger.error(f"Failed to extract chat history context: {e}")
        return {"user_query": "", "dataset_name": "", "specific_requests": []}

def parse_cell_outputs(cell) -> List[NotebookCellOutput]:
    outputs = []
    for out in cell.get("outputs", []):
        if out.output_type == "stream":
            content = "".join(out.get("text", []))
            outputs.append(NotebookCellOutput(type="stream", content=content))
        elif out.output_type == "error":
            content = "\n".join(out.get("traceback", []))
            outputs.append(NotebookCellOutput(type="error", content=content))
        elif out.output_type == "execute_result":
            # --- START OF FIX ---
            # Prioritize text output and explicitly ignore images
            if "text/plain" in out.get("data", {}):
                text = out["data"]["text/plain"]
                outputs.append(NotebookCellOutput(type="execute_result", content=text))
            # If there's image data but no text, we append nothing.
            # --- END OF FIX ---
        else:
            # This will also catch display_data with images and ignore them
            if 'data' in out and ('image/png' in out['data'] or 'image/jpeg' in out['data'] or 'image/svg+xml' in out['data']):
                outputs.append(NotebookCellOutput(type="image", content="[Image data omitted]"))
            else:
                outputs.append(NotebookCellOutput(type="unknown", content=""))
    return outputs

def is_critical_unresolvable_error(error_msg: str) -> bool:
    """Check if error is critical and unresolvable, should terminate execution"""
    critical_patterns = [
        "SyntaxError: invalid syntax",
        "IndentationError:",
        "NameError: name 'pd' is not defined",
        "ModuleNotFoundError: No module named",
        "SystemExit",
        "KeyboardInterrupt"
    ]
    
    return any(pattern in str(error_msg) for pattern in critical_patterns)

def has_execution_errors(cell) -> Tuple[bool, str]:
    """Check if a cell has execution errors."""
    error_msg = ""
    has_errors = False
    
    for output in cell.get("outputs", []):
        if output.get("output_type") == "error":
            has_errors = True
            traceback_lines = output.get("traceback", [])
            if traceback_lines:
                clean_traceback = []
                for line in traceback_lines:
                    clean_line = re.sub(r'\x1b\[[0-9;]*m', '', line)
                    clean_traceback.append(clean_line)
                error_msg = "\n".join(clean_traceback)
            else:
                error_msg = f"{output.get('ename', 'Error')}: {output.get('evalue', 'Unknown error')}"
            break
    
    return has_errors, error_msg

def extract_structured_data_context(nb: nbformat.NotebookNode,
                                optimized_client: OptimizedNotebookClient) -> Dict[str, Any]:
    """Extract comprehensive data context from notebook execution with column name handling"""
    context = {
        "variables": optimized_client.variable_tracker.get_available_variables(),
        "columns": [],
        "column_issues": False,
        "sample_data": "",
        "recent_outputs": []
    }
    
    try:
        recent_cells = sorted(optimized_client.cell_outputs_history.keys(), reverse=True)[:2]
        
        for cell_idx in recent_cells:
            outputs = optimized_client.cell_outputs_history[cell_idx]
            for output in outputs:
                if output.content and len(output.content.strip()) > 0:
                    content = output.content
                    
                    # Check for column display patterns
                    if any(keyword in content.lower() for keyword in ['shape:', 'columns:', 'index([', 'dtype']):
                        context["recent_outputs"].append(f"Cell {cell_idx}: {content[:300]}")
                        
                        # Extract actual column names from output
                        if 'Index([' in content:
                            # Extract column names from pandas Index display
                            import re
                            match = re.search(r"Index\(\[(.*?)\]", content, re.DOTALL)
                            if match:
                                cols_str = match.group(1)
                                # Extract quoted column names
                                col_matches = re.findall(r"'([^']*)'", cols_str)
                                if col_matches:
                                    context["columns"] = col_matches
                                    # Check for leading/trailing spaces
                                    for col in col_matches:
                                        if col != col.strip():
                                            context["column_issues"] = True
                        
                        # Extract from df.columns output
                        elif 'Time Stamp' in content and 'Current' in content:
                            # Fallback column detection
                            if ' Current-A' in content or 'Current-A ' in content:
                                context["column_issues"] = True
                                context["columns"] = ['Time Stamp', ' Current-A', ' Current-B', ' Current-C', 'bear_fault', 'healthy', 'watt', 'size']
                            else:
                                context["columns"] = ['Time Stamp', 'Current-A', 'Current-B', 'Current-C', 'bear_fault', 'healthy', 'watt', 'size']
                        
                        if 'shape:' in content.lower():
                            context["sample_data"] = content[:300]
    
    except Exception as e:
        logger.warning(f"Error extracting data context: {e}")
    
    return context
def hard_restart_ollama_service():
    """
    Performs a platform-dependent hard restart of the Ollama service by
    killing any existing process and starting a new one.
    """
    os_name = platform.system()
    logger.warning(f"🚨 Performing a hard restart of the Ollama service on {os_name}...")

    try:
        # --- Terminate any existing Ollama process ---
        if os_name == "Windows":
            # For Windows, find and kill ollama.exe
            kill_command = ["taskkill", "/F", "/IM", "ollama.exe"]
            subprocess.run(kill_command, capture_output=True, text=True, check=False)
        else:
            # For Linux and macOS, use pkill
            kill_command = ["pkill", "-9", "ollama"]
            subprocess.run(kill_command, capture_output=True, text=True, check=False)
        
        logger.info("✅ Terminate command issued for any existing Ollama process.")
        time.sleep(3) # Wait a moment for the port to be released.

    except FileNotFoundError:
        # This can happen if pkill/taskkill is not in the system's PATH.
        logger.error("Could not execute kill command. Please ensure 'pkill' (Linux/macOS) or 'taskkill' (Windows) is in your system's PATH.")
    except Exception as e:
        logger.warning(f"Could not terminate Ollama (this is okay if it wasn't running). Error: {e}")

    try:
        # --- Restart the Ollama service in the background ---
        logger.info("🚀 Attempting to restart 'ollama serve' in the background...")
        # Use Popen to run 'ollama serve' as a new, detached process
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        logger.info("✅ Ollama restart command has been sent. Waiting 15 seconds for the service to initialize...")
        time.sleep(15) # Give the service ample time to start up before the next attempt.
        logger.info("✅ Service should now be restarted.")

    except Exception as e:
        logger.error(f"❌ CRITICAL: The command to restart 'ollama serve' failed: {e}")
        logger.error("The agent will likely fail on its next attempt unless Ollama is restarted manually.")

# ================================
# ENHANCED CELL GENERATION
# ================================

# In coding_agent.py, replace the entire function

def generate_next_cell_with_memory(
    task_description: str,
    nb: nbformat.NotebookNode,
    chat_history_context: Dict[str, Any],
    optimized_client: OptimizedNotebookClient,
    last_step_output: str,
    last_step_error: str,
    override_messages: Optional[list] = None
) -> Optional[AgentOutput]:
    """
    Generates the next cell by preparing a prompt and calling the LLM.

    This function is now a simple orchestrator. It either uses a pre-built
    "override_messages" prompt (for fixing code) or constructs a standard
    prompt for a new task.
    """
    messages = []
    if override_messages:
        # If a special prompt (like the Code Fixer) is provided, use it directly.
        messages = override_messages
    else:
        # Otherwise, build the standard prompt for a new task.
        data_context = extract_structured_data_context(nb, optimized_client)
        df_loaded = 'df' in data_context['variables']
        dataset_name = chat_history_context.get('dataset_name', 'dataset.csv')
        available_columns = data_context.get('columns', [])

        dynamic_context_parts = [
            "**CURRENT STATE:**",
            f"- DataFrame `df` is loaded: {'YES' if df_loaded else 'NO'}",
            f"- Filename for loading: '{dataset_name}'",
            f"- Available DataFrame Columns: {available_columns if available_columns else 'Unknown'}"
        ]
        if last_step_error:
            dynamic_context_parts.append(f"\n**PREVIOUS STEP FAILED:**\n- ERROR: {last_step_error}")
        else:
            dynamic_context_parts.append(f"\n**PREVIOUS STEP SUCCEEDED:**\n- OUTPUT PREVIEW:\n{last_step_output[:1500]}")
        
        dynamic_context = "\n".join(dynamic_context_parts)
        system_prompt = CODING_AGENT_SYSTEM_PROMPT.format(dynamic_context=dynamic_context)
        user_message = f"**CURRENT TASK:** {task_description}\n\nGenerate the JSON for this single task."
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}]

    # --- This section calls the LLM and parses the response ---
    raw_response_content = "No response from model."
    try:
        resp = chat_with_timeout(
            messages=messages,
            model="gemma3n:e2b",
            format="json",
            options={"temperature": 0.7}
        )

        if resp is None:
            raise Exception("LLM call timed out and returned no response.")

        raw_response_content = resp.message.content
        
        print("=" * 60)
        print("🔥 RAW LLM RESPONSE:")
        print(raw_response_content)
        print("=" * 60)
        
        json_match = re.search(r"\{.*\}", raw_response_content, re.DOTALL)
        if not json_match:
            raise json.JSONDecodeError("No valid JSON object found in the response.", raw_response_content, 0)
        
        cleaned_json = json_match.group(0)
        data = json.loads(cleaned_json)
        
        if "cell" not in data or "status" not in data:
            error_msg = f"LLM response was missing required keys ('cell' or 'status'). Response: {data}"
            print(f"❌ {error_msg}")
            raise Exception(error_msg)
            
        return AgentOutput.model_validate(data)
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON Parse Error: {e}")
        print(f"Raw content that failed parsing: {raw_response_content}")
        return None
    except Exception as e:
        print(f"❌ General Error in cell generation: {e}")
        return None

def setup_notebook_and_kernel(chat_id: str) -> Tuple[nbformat.NotebookNode, OptimizedNotebookClient]:
    """Initializes the notebook and kernel for an agent session."""
    session_dir = NOTEBOOKS_DIR / chat_id
    nb = load_or_create_notebook(chat_id)
    nb.metadata['path'] = str(session_dir / f"{chat_id}.ipynb")
    
    # Start with a clean notebook for the execution run
    nb.cells = []
    logger.info("Starting execution with a clean notebook.")
    
    client = OptimizedNotebookClient(nb, timeout=600)
    if not client.start_kernel():
        raise Exception("Fatal: Jupyter kernel failed to start.")
        
    return nb, client

def safe_format_code(code_template: str, **kwargs) -> str:
    """
    Safely formats a code string, only replacing explicitly provided placeholders.
    It ignores other curly braces, preventing KeyErrors.
    """
    class SafeDict(dict):
        def __missing__(self, key):
            return f"{{{key}}}" # Return the placeholder itself if key is not found
    
    return code_template.format_map(SafeDict(**kwargs))

def generate_next_cell_prompt(
    task_description: str,
    nb: nbformat.NotebookNode,
    chat_history_context: Dict[str, Any],
    optimized_client: OptimizedNotebookClient,
    last_step_output: str,
    last_step_error: str
) -> list:
    """Builds the standard prompt for generating the next code cell."""
    data_context = extract_structured_data_context(nb, optimized_client)
    df_loaded = 'df' in data_context['variables']
    dataset_name = chat_history_context.get('dataset_name', 'dataset.csv')
    available_columns = data_context.get('columns', [])

    dynamic_context_parts = [
        "**CURRENT STATE:**",
        f"- DataFrame `df` is loaded: {'YES' if df_loaded else 'NO'}",
        f"- Filename for loading: '{dataset_name}'",
        f"- Available DataFrame Columns: {available_columns if available_columns else 'Unknown'}"
    ]
    if last_step_error:
        dynamic_context_parts.append(f"\n**PREVIOUS STEP FAILED:**\n- ERROR: {last_step_error}")
    else:
        dynamic_context_parts.append(f"\n**PREVIOUS STEP SUCCEEDED:**\n- OUTPUT PREVIEW:\n{last_step_output[:1500]}")
    
    dynamic_context = "\n".join(dynamic_context_parts)
    system_prompt = CODING_AGENT_SYSTEM_PROMPT.format(dynamic_context=dynamic_context)
    user_message = f"**CURRENT TASK:** {task_description}\n\nGenerate the JSON for this single task."
    
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}]

# --- START OF FIX ---
def generate_code_from_messages(messages: list) -> Optional[AgentOutput]:
    """
    Calls the LLM with a given prompt and parses the JSON response.
    If any error occurs (timeout, parsing), it will hard-restart Ollama
    and retry the call ONCE for the same attempt.
    """
    max_restarts = 1
    restarts = 0

    while restarts <= max_restarts:
        raw_response_content = "No response from model."
        try:
            resp = chat_with_timeout(
                messages=messages,
                model="gemma3n:e2b",
                format="json",
                options={"temperature": 0.7}
            )

            # ALWAYS PRINT THE RAW RESPONSE FOR DEBUGGING
            # This block will execute before any parsing is attempted.
            print("=" * 60)
            print("🔥 RAW LLM RESPONSE:")
            if resp and hasattr(resp, 'message') and hasattr(resp.message, 'content'):
                raw_response_content = resp.message.content
                print(raw_response_content)
            else:
                print("Model did not return a valid response object.")
            print("=" * 60)

            if resp is None:
                raise Exception("LLM call timed out.")

            # Now, proceed with parsing the raw_response_content
            json_match = re.search(r"\{.*\}", raw_response_content, re.DOTALL)
            if not json_match:
                raise json.JSONDecodeError("No valid JSON object found in the response.", raw_response_content, 0)
            
            cleaned_json = json_match.group(0)
            data = json.loads(cleaned_json)
            
            if "cell" not in data or "status" not in data:
                raise Exception("LLM response was missing required keys ('cell' or 'status').")
            
            return AgentOutput.model_validate(data)
            
        except Exception as e:
            logger.error(f"Error in LLM call or parsing: {e}")
            restarts += 1
            
            if restarts <= max_restarts:
                logger.warning("Attempting Ollama restart and retrying the SAME attempt...")
                if "timed out" not in str(e):
                    hard_restart_ollama_service()
            else:
                logger.error("Max restarts for this attempt reached. Failing.")
                return None

    return None
# --- END OF FIX ---

def execute_cell_smart(client: OptimizedNotebookClient, nb: nbformat.NotebookNode, cell_index: int) -> Tuple[bool, str, str]:
    """Wrapper for executing a cell and getting simplified output."""
    success, error, outputs = client.execute_cell_smart(cell_index, force_execution=True)
    
    output_str = ""
    for out in outputs:
        if out.content:
            output_str += out.content + "\n"

    # Simplify visualization output for memory
    if detect_visualization_code(nb.cells[cell_index].source):
        output_str = "Visualization created successfully."

    return success, output_str, error

def shutdown(client: OptimizedNotebookClient, chat_id: str):
    """Safely shuts down the kernel."""
    logger.info(f"Shutting down kernel for chat {chat_id}.")
    client.shutdown()
    
def get_enhanced_step_guidance(step_index: int, task_name: str) -> str:
    """Enhanced guidance for each step"""
    
    guidance_map = {
        0: """
ENHANCED DATA LOADING:
- Load data with comprehensive validation
- Clean column names: df.columns = df.columns.str.strip()
- Perform data quality assessment
- Generate data summary statistics
""",
        1: """
ADVANCED DATA EXPLORATION:
- Create multiple visualization types
- Generate statistical summaries
- Interactive plots with plotly
- Data quality assessment
""",
        2: """
INTELLIGENT PREPROCESSING:
- Handle missing values intelligently
- Use dynamic column discovery
- Apply appropriate transformations
- Validate data consistency
""",
        6: """
TARGET VARIABLE PREPARATION:
- Identify appropriate target variable (healthy/bear_fault for classification)
- Use df.select_dtypes(include=['number']) for features
- Ensure proper train/test split
- Validate target distribution
""",
        7: """
ENHANCED MODEL TRAINING:
- Select appropriate model for data characteristics
- Use proper sklearn evaluation methods
- Include cross-validation
- Generate comprehensive metrics
""",
        8: """
COMPREHENSIVE EVALUATION:
- Use sklearn.metrics functions (not .evaluate())
- Generate multiple evaluation metrics
- Create visualizations of results
- Include model interpretation
"""
    }
    
    return guidance_map.get(step_index, "# Advanced Analysis Step")

def get_enhanced_step_guidance_with_context(step_index: int, task_name: str, chat_context: Dict[str, Any]) -> str:
    """Enhanced guidance for each step - FULLY DYNAMIC with NO hardcoded assumptions"""
    
    # Get base guidance first
    base_guidance = get_enhanced_step_guidance(step_index, task_name)
    
    # Return base guidance if no context available
    if not chat_context:
        return base_guidance
    
    dataset_name = chat_context.get('dataset_name', 'dataset.csv')
    user_request = chat_context.get('user_query', '')
    
    # DYNAMIC: Extract analysis intent from user request (NO hardcoded domains)
    analysis_intent = _extract_analysis_intent(user_request)
    
    # Step-specific context enhancements - GENERIC for ANY dataset
    if step_index == 0:  # Data loading
        context_guidance = """
DYNAMIC DATA LOADING CONTEXT:
- Dataset file: '{dataset_name}'
- User's goal: "{user_request}"
- Load with: df = pd.read_csv('{dataset_name}')
- Prepare for analysis type: {analysis_intent}
- NO assumptions about column names or data structure
"""
        return base_guidance + context_guidance
    
    elif step_index == 1:  # Data exploration
        context_guidance = """
ADAPTIVE EXPLORATION CONTEXT:
- User objective: "{user_request}"
- Dataset: {dataset_name}
- Discovery approach: Examine ACTUAL data structure first
- Let data characteristics guide exploration strategy
- NO predetermined analysis paths
"""
        return base_guidance + context_guidance
    
    elif step_index in [2, 3, 4]:  # Quality assessment and basic analysis
        context_guidance = """
DATA-DRIVEN QUALITY ASSESSMENT:
- Target user need: "{user_request}"
- Dataset: {dataset_name}
- Discover data types, missing values, distributions dynamically
- Adapt quality checks to discovered data characteristics
"""
        return base_guidance + context_guidance
    
    elif step_index in [5, 6, 7]:  # Pattern analysis and insights
        context_guidance = """
INTELLIGENT PATTERN DISCOVERY:
- User's question: "{user_request}"
- Dataset: {dataset_name}
- Find patterns relevant to user's specific goal
- Generate insights based on discovered data relationships
- NO predefined analysis templates
"""
        return base_guidance + context_guidance
    
    elif step_index in [8, 9, 10]:  # Advanced analysis/ML
        context_guidance = """
ADAPTIVE ADVANCED ANALYSIS:
- Addressing: "{user_request}"
- Dataset: {dataset_name}
- Choose methods based on discovered data types and patterns
- Select features and targets dynamically
- Adapt approach to actual data characteristics
"""
        return base_guidance + context_guidance
    
    else:  # Any other steps
        context_guidance = """
GENERAL CONTEXT:
- User objective: "{user_request}"
- Dataset: {dataset_name}
- Analysis approach: Data-driven and adaptive
- Focus: Answer user's specific question through discovered insights
"""
        return base_guidance + context_guidance

def _extract_analysis_intent(user_request: str) -> str:
    """Extract analysis intent from user request - NO hardcoded domains"""
    
    if not user_request:
        return "general_analysis"
    
    request_lower = user_request.lower()
    
    # Generic analysis type detection (NO domain-specific hardcoding)
    if any(word in request_lower for word in ['predict', 'forecast', 'model', 'classification', 'regression']):
        return "predictive_modeling"
    elif any(word in request_lower for word in ['pattern', 'trend', 'correlation', 'relationship']):
        return "pattern_analysis"
    elif any(word in request_lower for word in ['compare', 'difference', 'segment', 'group']):
        return "comparative_analysis"
    elif any(word in request_lower for word in ['anomaly', 'outlier', 'unusual', 'detect']):
        return "anomaly_detection"
    elif any(word in request_lower for word in ['optimize', 'improve', 'recommendation']):
        return "optimization_analysis"
    elif any(word in request_lower for word in ['summary', 'overview', 'describe', 'understand']):
        return "descriptive_analysis"
    else:
        return "exploratory_analysis"

def fix_code_with_context(original_code: str, error_msg: str, data_context: Dict[str, Any]) -> Optional[str]:
    """Fix code with enhanced context and specific error handling"""
    
    # Specific fixes for common errors
    if "missing_values_strategy" in error_msg:
        fixed_code = """
# Simple missing value handling
for col in df.select_dtypes(include=['number']).columns:
    if df[col].isnull().any():
        df[col].fillna(df[col].mean(), inplace=True)

for col in df.select_dtypes(include=['object']).columns:
    if df[col].isnull().any():
        df[col].fillna('Unknown', inplace=True)

print("Missing values handled successfully")
print(df.isnull().sum())
"""
        return fixed_code
    
    # Use LLM for other errors
    context_str = """
Available columns: {', '.join(data_context.get('columns', []))}
Available variables: {', '.join(data_context.get('variables', []))}
"""
    
    messages = [
        {"role": "system", "content": CODE_FIXER_SYSTEM_PROMPT},
        {"role": "user", "content": """
ORIGINAL CODE:
{original_code}

ERROR:
{error_msg}

CONTEXT:
{context_str}

Fix the code with by writing robust code with expection handling to avoid code execution failure.
"""}
    ]
    
    try:
        resp = chat(
            messages=messages,
            model="gemma3n:e2b",
            format={"code": "Corrected Python code here"},
            options={"temperature": 0.7}
        )
        data = json.loads(resp.message.content)
        return data.get("code")
    except Exception as e:
        logger.error(f"Failed to get code fix from LLM: {e}")
        # Return a simple fallback
        return """
# Simple fallback code
print("Processing data...")
print(f"DataFrame shape: {df.shape}")
print(f"Missing values: {df.isnull().sum().sum()}")
"""


# ================================
# MAIN EXECUTION LOGIC
# ================================

def find_planned_chat():
    """Finds a RECENT chat session with a 'planned' status to prevent acting on old tasks."""
    # Check files from newest to oldest
    for file in sorted(CHAT_HISTORY_DIR.glob("*.json"), key=os.path.getmtime, reverse=True):
        try:
            # Check if the file was modified in the last 5 minutes to avoid stale tasks
            file_mod_time = file.stat().st_mtime
            if (time.time() - file_mod_time) > 300: # 5 minutes
                continue # Skip old files

            logger.debug(f"Checking recent file: {file}")
            with open(file, "r", encoding="utf-8") as f:
                history = json.load(f)

            # Find the trigger object
            for obj in reversed(history):
                if "task" in obj and obj["task"].get("status") == "planned":
                    task_block = obj["task"]
                    tasks = task_block.get("tasks", [])
                    # IMPORTANT: Get the dataset name from the trigger object
                    dataset_name = task_block.get("dataset_name")

                    if tasks and dataset_name:
                        logger.info(f"Found active task for chat {file.stem} with dataset {dataset_name}")
                        return file, history, tasks, dataset_name
        except Exception as e:
            logger.error(f"Error reading {file}: {e}")
            continue
    return None, None, None, None

def update_progress(file_path: Path, history_obj: List, pct: int, status: str):
    """Safely appends a progress update with better real-time handling"""
    progress_msg = {
        "progress": pct,
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "step_type": "progress_update"
    }

    try:
        # Always load fresh from disk for real-time updates
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                current_history = json.load(f)
        else:
            current_history = []

        # Remove old progress updates to prevent bloat
        current_history = [item for item in current_history if item.get("step_type") != "progress_update"]
        
        # Add new progress
        current_history.append(progress_msg)
        
        # Write immediately with flush
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(current_history, f, indent=2, ensure_ascii=False)
            f.flush()  # Force write to disk
            
        logger.info(f"📊 Progress updated: {pct}% - {status}")
        
    except Exception as e:
        print(json.dumps(progress_msg))
        logger.error(f"Progress update failed: {e}")


def get_last_user_query(history: List[Dict[str, Any]]) -> str:
    for obj in reversed(history):
        if "user" in obj and "query" in obj["user"]:
            return obj["user"]["query"]
    return ""

def decide_next_micro_step(main_task: str, full_context: str) -> Dict:
    """
    Asks the LLM to decide if the main task is complete, or what the next micro-step should be.
    """
    decision_prompt = """
You are a data science agent executing a plan.
Your CURRENT MAIN TASK is: "{main_task}"

Here is the context of everything done so far:
---
{full_context}
---

Based on the context, have you fully completed the main task?
If YES, respond with action: 'complete'.
If NO, decide the very next, small, specific micro-step to continue working on the main task.

Respond ONLY with a valid JSON object with the following schema:
{{
  "action": "'continue' or 'complete'",
  "reasoning": "A brief thought process on why you made this decision.",
  "micro_task": "The specific micro-task to perform next (or null if complete)."
}}
"""
    try:
        response = chat(
            model="gemma3n:e2b",
            messages=[{"role": "user", "content": decision_prompt}],
            format="json"
        )
        decision = json.loads(response['message']['content'])
        return {
            "action": decision.get("action", "complete"),
            "reasoning": decision.get("reasoning", ""),
            "micro_task": decision.get("micro_task")
        }
    except Exception as e:
        logger.error(f"Failed to decide next micro-step: {e}")
        return {"action": "complete", "reasoning": "Error during decision making."}



# ====================================================================
# SUPPORTING FUNCTIONS FOR PURE LLM EXECUTION
# ====================================================================
def check_analysis_completion(completed_analysis: Dict[str, bool], step_count: int) -> Tuple[bool, str]:
    """Check if analysis is logically complete"""
    
    # Must have completed basic analysis steps
    required_steps = ['data_loaded', 'missing_values_checked', 'basic_stats_calculated']
    basic_complete = all(completed_analysis.get(step, False) for step in required_steps)
    
    # Optional advanced steps
    advanced_steps = ['correlations_analyzed', 'visualizations_created']
    advanced_complete = any(completed_analysis.get(step, False) for step in advanced_steps)
    
    # Completion criteria
    if step_count >= 15 and basic_complete and advanced_complete:
        return True, "Analysis complete: All major analysis steps finished"
    elif step_count >= 10 and basic_complete:
        return True, "Analysis complete: Basic analysis finished, stopping to avoid repetition"
    elif step_count >= 25:
        return True, "Analysis complete: Maximum reasonable steps reached"
    
    return False, f"Analysis continuing: {step_count} steps completed"

def ask_llm_for_next_step(user_request: str, dataset_name: str, 
                         current_context: str, step_number: int) -> Optional[Dict]:
    """Ask LLM to autonomously decide what to do next - NO predetermined tasks"""
    
    prompt = """You are an autonomous data science agent. Based on the user's request and current notebook state, decide what specific action to take next.

USER'S ORIGINAL REQUEST: {user_request}
DATASET: {dataset_name}
CURRENT STEP: {step_number}

CURRENT NOTEBOOK STATE:
{current_context}

THINK AUTONOMOUSLY:
- What has already been accomplished?
- What is still needed to fulfill the user's request?
- What would be the most logical next step?
- Is the analysis complete and the user's question answered?

If the analysis fully addresses the user's request, return action: "complete"
Otherwise, decide on the next specific action needed.

Return JSON format only:
{{
  "action": "analyze" or "complete",
  "description": "specific description of what to do next",
  "reasoning": "why this step is necessary"
}}"""

    try:
        response = chat(
            messages=[{"role": "user", "content": prompt}],
            model="gemma3n:e2b",
            options={"temperature": 0.7, "num_ctx": 12000}  # Optimized for your model
        )
        
        content = response.message.content.strip()
        
        # Handle markdown code blocks
        if content.startswith('```'):
            content = content[7:-3].strip()
        elif content.startswith('```'):
            content = content[3:-3].strip()
        
        return json.loads(content)
        
    except Exception as e:
        logger.error(f"🤖 LLM decision making failed: {e}")
        return None


def execute_llm_generated_step(nb, optimized_client, action: Dict, user_request: str, 
                              dataset_name: str, chat_id: str) -> bool:
    """Execute a step by having LLM generate appropriate code - CORRECTED VERSION"""
    
    import time
    from datetime import datetime
    
    step_start_time = time.time()
    
    prompt = """Generate Python code for this specific analysis step.

USER'S ORIGINAL REQUEST: {user_request}
DATASET: {dataset_name}
CURRENT TASK: {action['description']}
REASONING: {action.get('reasoning', 'No reasoning provided')}

IMPORTANT: The "code" field must contain ONLY executable Python code. NO markdown, NO backticks, NO code block formatting.

Return JSON format only:
{{
  "markdown": "## Clear step title",
  "code": "import pandas as pd\\nprint('Hello world')"
}}"""

    try:
        response = chat(
            messages=[{"role": "user", "content": prompt}],
            model="gemma3n:e2b",
            options={"temperature": 0.7, "num_ctx": 10000}
        )
        
        # FIXED: Proper markdown cleaning logic
        content = response.message.content.strip()
        original_content = content  # Keep for logging
        
        if content.startswith('```'):
            content = content[7:]  # Remove ```json
            if content.endswith('```'):
                content = content[:-3]  # Remove closing ```
        elif content.startswith('```'):
            content = content[3:]  # Remove ```
            if content.endswith('```'):
                content = content[:-3]  # Remove closing ```
        
        content = content.strip()

        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            logger.error(f"Raw content: '{original_content[:200]}...'")
            logger.error(f"Cleaned content: '{content[:200]}...'")
            
            # Log the failure to agent memory
            log_agent_memory_step(chat_id, {
                "step_type": "code_generation_failure",
                "timestamp": datetime.now().isoformat(),
                "action": action['description'],
                "error": f"JSON parsing failed: {e}",
                "raw_response": original_content[:500],
                "cleaned_response": content[:500]
            })
            return False

        # Validate and clean code
        code = data.get('code', '')
        markdown = data.get('markdown', '## Analysis Step')
        
        # FIXED: Proper code cleaning
        if '```' in code and not code.strip().startswith('```'):
            logger.warning("Model still returning markdown formatting, applying emergency cleaning")
            code = code.replace('```python', '').replace('```')
            code = '\n'.join(line for line in code.split('\n') if not line.strip().startswith('```'))
            data['code'] = code
        
        if not code.strip():
            logger.error("No executable code generated")
            log_agent_memory_step(chat_id, {
                "step_type": "empty_code_failure",
                "timestamp": datetime.now().isoformat(),
                "action": action['description'],
                "error": "No executable code generated",
                "model_response": data
            })
            return False
        
        logger.info(f"Generated code length: {len(code)} characters")
        logger.debug(f"Generated code preview: {code[:100]}...")
        
        # Add cells to notebook
        append_cells(nb, markdown, code)
        save_notebook(chat_id, nb)
        
        # Execute the code
        code_cell_index = len(nb.cells) - 1
        success, error, outputs = optimized_client.execute_cell_smart(
            code_cell_index, force_execution=True
        )
        
        processing_time = time.time() - step_start_time
        
        # 🆕 LOG TO AGENT MEMORY - Success or Failure
        memory_entry = {
            "step_type": "code_execution",
            "timestamp": datetime.now().isoformat(),
            "action_description": action['description'],
            "user_request": user_request,
            "dataset": dataset_name,
            "success": success,
            "processing_time_seconds": processing_time,
            "model_response": {
                "raw_response": original_content[:300],
                "parsed_data": data,
                "generated_code": code,
                "generated_markdown": markdown
            },
            "execution_results": {
                "success": success,
                "error": error if not success else None,
                "outputs": [{"type": out.type, "content": out.content[:200]} for out in outputs] if outputs else [],
                "cell_index": code_cell_index
            }
        }
        
        # Log to agent memory
        log_agent_memory_step(chat_id, memory_entry)
        
        if success:
            logger.info(f"✅ Code executed successfully in {processing_time:.1f}s")
            return True
        else:
            logger.error(f"❌ Code execution failed: {error}")
            return False
            
    except Exception as e:
        processing_time = time.time() - step_start_time
        logger.error(f"💥 Critical failure in code generation: {e}")
        
        # Log critical failure to agent memory
        log_agent_memory_step(chat_id, {
            "step_type": "critical_failure",
            "timestamp": datetime.now().isoformat(),
            "action": action['description'],
            "error": str(e),
            "processing_time_seconds": processing_time,
            "exception_type": type(e).__name__
        })
        return False


def log_agent_memory_step(chat_id: str, memory_data: Dict):
    """Log detailed agent memory to separate JSON file"""
    
    import os
    from pathlib import Path
    
    # Create agent_memory directory if it doesn't exist
    memory_dir = Path("agent_memory")
    memory_dir.mkdir(exist_ok=True)
    
    # Agent memory file path
    memory_file = memory_dir / f"agent_{chat_id}.json"
    
    try:
        # Load existing memory or create new
        if memory_file.exists():
            existing_memory = json.loads(memory_file.read_text(encoding="utf-8"))
        else:
            existing_memory = {
                "agent_session": chat_id,
                "created": datetime.now().isoformat(),
                "memory_entries": []
            }
        
        # Add new memory entry
        existing_memory["memory_entries"].append(memory_data)
        existing_memory["last_updated"] = datetime.now().isoformat()
        existing_memory["total_entries"] = len(existing_memory["memory_entries"])
        
        # Keep only last 100 entries to prevent file from growing too large
        if len(existing_memory["memory_entries"]) > 100:
            existing_memory["memory_entries"] = existing_memory["memory_entries"][-100:]
            existing_memory["total_entries"] = 100
        
        # Save to file
        memory_file.write_text(
            json.dumps(existing_memory, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        
        logger.debug(f"💾 Agent memory logged to {memory_file}")
        
    except Exception as e:
        logger.error(f"Failed to log agent memory: {e}")


def get_current_notebook_context(nb, optimized_client) -> str:
    """Get context with strong loop prevention"""
    
    context = f"Notebook has {len(nb.cells)} cells.\n"
    
    # Count different analysis types
    analysis_counts = {
        'visualizations': 0,
        'statistics': 0, 
        'missing_checks': 0,
        'data_info': 0,
        'correlations': 0
    }
    
    completed_basic = {
        'data_loaded': False,
        'info_shown': False,
        'stats_done': False,
        'viz_created': False
    }
    
    # Analyze existing cells
    for cell in nb.cells:
        if cell.cell_type == "code" and cell.source:
            source = cell.source.lower()
            
            # Count analysis types
            if any(word in source for word in ['plot', 'hist', 'chart', 'show()']):
                analysis_counts['visualizations'] += 1
                completed_basic['viz_created'] = True
                
            if any(word in source for word in ['describe', 'mean', 'std']):
                analysis_counts['statistics'] += 1
                completed_basic['stats_done'] = True
                
            if any(word in source for word in ['isnull', 'missing']):
                analysis_counts['missing_checks'] += 1
                
            if any(word in source for word in ['info()', 'head()', 'shape']):
                analysis_counts['data_info'] += 1
                completed_basic['info_shown'] = True
                
            if 'corr()' in source:
                analysis_counts['correlations'] += 1
                
            if 'pd.read_csv' in source:
                completed_basic['data_loaded'] = True
    
    # Build prevention rules
    context += "\n🛑 PREVENTION RULES:\n"
    for analysis_type, count in analysis_counts.items():
        if count >= 2:
            context += f"- NO MORE {analysis_type.upper()} - Done {count} times already\n"
        elif count >= 1:
            context += f"- Limit {analysis_type} - Already done {count} time(s)\n"
    
    # Suggest next steps
    context += "\n🎯 NEXT PRIORITY:\n"
    if not completed_basic['data_loaded']:
        context += "- Load and inspect dataset\n"
    elif not completed_basic['info_shown']:
        context += "- Show basic dataset information\n"
    elif not completed_basic['stats_done']:
        context += "- Generate statistical summary\n"
    elif not completed_basic['viz_created']:
        context += "- Create one key visualization\n"
    else:
        context += "- Generate final insights and conclusions\n"
    
    # Available variables
    variables = optimized_client.variable_tracker.get_available_variables()
    context += f"\nAvailable variables: {', '.join(variables) if variables else 'None'}\n"
    
    return context


def ask_llm_for_step_estimation(user_request: str, dataset_name: str) -> int:
    """Ask LLM to estimate steps with robust JSON cleaning"""
    
    prompt = """Estimate micro-steps needed for: {user_request}
Dataset: {dataset_name}

Return ONLY JSON: {{"estimated_steps": number_between_10_and_30}}"""

    try:
        response = chat(
            messages=[{"role": "user", "content": prompt}],
            model="gemma3n:e2b",
            options={"temperature": 0.7, "num_ctx": 2000}
        )
        
        content = response.message.content.strip()
        
        # Robust JSON extraction
        if content.startswith('```'):
            content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
        elif content.startswith('```'):
            content = content[3:]
            if content.endswith('```'):
                content = content[:-3]
        
        content = content.strip()
        
        # Handle multi-line responses with explanations
        if '\n' in content:
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('{') and line.endswith('}'):
                    content = line
                    break
        
        data = json.loads(content)
        estimated = data.get("estimated_steps", 15)
        estimated = max(10, min(estimated, 30))  # Cap between 10-30
        
        logger.info(f"🧠 LLM estimated {estimated} micro-steps needed")
        return estimated
        
    except Exception as e:
        logger.error(f"Step estimation failed: {e}")
        return 15  # Default fallback


def ask_llm_for_micro_action(user_request: str, dataset_name: str, current_context: str,
                            step_number: int, estimated_total: int) -> Optional[Dict]:
    """Ask LLM for next action with strong loop prevention"""
    
    prompt = """You are an autonomous data analyst. Choose the NEXT logical action.

USER GOAL: {user_request}
DATASET: {dataset_name}
STEP: {step_number}/{estimated_total}

CURRENT STATE:
{current_context}

CRITICAL RULES:
- NEVER repeat actions marked as "Done X times"
- If analysis basics are complete, focus on insights
- Choose actions that advance toward user's goal
- If step >= 12, consider completing analysis

Good progression:
1. Load data → 2. Basic info → 3. Key statistics → 4. One visualization → 5. Insights → 6. Complete

Return JSON: {{"action": "analyze" or "complete", "description": "specific_next_action", "reasoning": "why_needed"}}"""

    try:
        response = chat(
            messages=[{"role": "user", "content": prompt}],
            model="gemma3n:e2b",
            options={"temperature": 0.7, "num_ctx": 6000}
        )
        
        content = response.message.content.strip()
        
        # Fixed syntax error:
        if content.startswith('```json'):
            content = content[7:-3]
        elif content.startswith('```'):
            content = content[3:-3]
        
        return json.loads(content.strip())
        
    except Exception as e:
        logger.error(f"Action generation failed: {e}")
        return {"action": "complete", "description": "Complete analysis", "reasoning": "Error recovery"}


def get_dynamic_dataset_info(df):
    return {
        'columns': list(df.columns),
        'numeric_columns': df.select_dtypes(include=['number']).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
        'datetime_columns': df.select_dtypes(include=['datetime']).columns.tolist(),
        'shape': df.shape,
        'sample_data': df.head(2).to_dict()
    }


def execute_micro_action(nb, optimized_client, action: Dict, user_request: str,
                        dataset_name: str, chat_id: str) -> Tuple[bool, Dict]:
    """Execute micro-action with RAG memory and efficient processing"""
    
    start_time = time.time()
    
    # Get available variables efficiently
    available_vars = optimized_client.variable_tracker.get_available_variables()
    df_available = 'df' in available_vars
    
    # Quick dataset context if df exists
    dataset_context = ""
    if df_available:
        dataset_context = """
DATASET STATUS: df is loaded and available
CRITICAL: Use existing df variable - NEVER reload data
"""
    else:
        dataset_context = f"DATASET STATUS: Need to load {dataset_name}"
    
    # Check if this is visualization
    is_viz = any(kw in action['description'].lower() 
                 for kw in ['plot', 'visualiz', 'chart', 'graph', 'hist'])
    
    # Create focused prompt based on task type
    if is_viz:
        prompt = """Generate visualization code for: {action['description']}

USER GOAL: {user_request}
{dataset_context}

REQUIREMENTS:
- Use only working matplotlib/seaborn functions
- NO plt.corrplot() or sns.corrplot() (don't exist)
- Include plt.show()
- 4-8 lines maximum

Return JSON: {{"markdown": "## Title", "code": "working_code"}}"""
    else:
        prompt = """Generate analysis code for: {action['description']}

USER GOAL: {user_request}
{dataset_context}

REQUIREMENTS:
- Focus on data analysis operations
- Use existing variables when possible
- Include print statements for output
- 3-6 lines maximum

Return JSON: {{"markdown": "## Title", "code": "working_code"}}"""
    
    try:
        # Call model with appropriate temperature
        temp = 0.8 if is_viz else 0.95
        response = chat(
            messages=[{"role": "user", "content": prompt}],
            model="gemma3n:e2b",
            options={"temperature": temp, "num_ctx": 4000}
        )
        
        content = response.message.content.strip()
        
        # Simple JSON cleaning
        if content.startswith('```'):
            if content.startswith('```json'):
                content = content[7:]
            else:
                content = content[3:]
            if content.endswith('```'):
                content = content[:-3]
        
        # Parse JSON
        try:
            data = json.loads(content.strip())
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse failed: {e}")
            return False, {"error": f"JSON error: {e}", "processing_time": time.time() - start_time}
        
        markdown = data.get('markdown', '## Step')
        code = data.get('code', '')
        
        if not code.strip():
            return False, {"error": "No code generated", "processing_time": time.time() - start_time}
        
        # Fix common syntax errors
        if 'plt.corrplot' in code:
            code = code.replace('plt.corrplot', 'sns.heatmap(df.corr(), annot=True); plt.title("Correlation Matrix")')
        if 'sns.corrplot' in code:
            code = code.replace('sns.corrplot', 'sns.heatmap(df.corr(), annot=True)')
        
        # Prevent data reloading
        if df_available and 'pd.read_csv' in code:
            code = code.replace('pd.read_csv', '# df already loaded\n# pd.read_csv')
        if '.to_markdown()' in code:
            logger.warning("🚫 Replacing df.describe().to_markdown() - tabulate not available")
            code = code.replace('.to_markdown()', '')
            code = code.replace('print(df.describe())', 'print("Statistical Summary:")\nprint(df.describe())')
        
        # Fix other common issues
        if 'df.describe().to_markdown()' in code:
            code = code.replace('df.describe().to_markdown()', 'df.describe()')
        
        # Add to notebook and execute
        append_cells(nb, markdown, code)
        save_notebook(chat_id, nb)
        
        # Execute the cell
        cell_idx = len(nb.cells) - 1
        success, error, outputs = optimized_client.execute_cell_smart(cell_idx, force_execution=True)
        
        processing_time = time.time() - start_time

        
        # Return results
        return success, {
            "markdown": markdown,
            "code": code,
            "execution_success": success,
            "error": error if not success else None,
            "processing_time": processing_time,
            "outputs": [{"content": out.content[:100]} for out in outputs] if outputs else [],
            "visualization": is_viz
        }
        
    except Exception as e:
        logger.error(f"Micro-action failed: {e}")
        return False, {
            "error": str(e), 
            "processing_time": time.time() - start_time,
            "execution_success": False
        }

# In coding_agent.py, replace the existing function
def chat_with_timeout(messages, model, timeout_seconds=150, **kwargs):
    """
    Calls ollama.chat with a timeout. If it times out, it triggers a
    hard restart of the Ollama service and then fails the current attempt.
    """
    import queue

    result_queue = queue.Queue()
    exception_queue = queue.Queue()

    def target():
        try:
            response = chat(messages=messages, model=model, **kwargs)
            result_queue.put(response)
        except Exception as e:
            exception_queue.put(e)

    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout=timeout_seconds)

    if thread.is_alive():
        # --- THIS IS THE NEW LOGIC ---
        logger.error(f"❌ LLM call timed out after {timeout_seconds} seconds. Triggering hard restart...")
        hard_restart_ollama_service()
        # By returning None, we signal that this attempt failed. The agent's
        # main retry loop will then attempt the task again with the fresh service.
        return None
        # --- END OF NEW LOGIC ---

    if not exception_queue.empty():
        raise exception_queue.get()

    if not result_queue.empty():
        return result_queue.get()

    return None

def append_single_markdown_cell(nb, markdown_text: str):
    """Append only a markdown cell to the notebook"""
    from nbformat.v4 import new_markdown_cell
    
    cleaned_markdown = markdown_text.strip()
    md_cell = new_markdown_cell(cleaned_markdown)
    nb.cells.append(md_cell)

    
def initialize_agent_memory(chat_id: str, history: List[Dict[str, Any]], memory_manager: AgentMemoryManager):
    """
    Creates the agent_memory file immediately and populates it with the initial prompt.
    Looks for the "task" object in the history.
    """
    logger.info(f"Initializing agent memory for chat_id: {chat_id}")

    # --- START OF FIX ---
    # Find the last user query and the model's task list from the history
    last_user_query = ""
    model_tasks = []
    for obj in reversed(history):
        if "user" in obj and "query" in obj["user"]:
            last_user_query = obj["user"]["query"]
        # Correctly look for the 'task' object, not 'plan'
        if "task" in obj and "tasks" in obj["task"]:
            model_tasks = obj["task"]["tasks"]
        
        # Stop once we have both pieces of information
        if last_user_query and model_tasks:
            break
    # --- END OF FIX ---

    if not last_user_query or not model_tasks:
        logger.error("Could not find initial user query or task list in history. Cannot initialize memory.")
        return

    # Use the memory manager to save this initial context
    memory_manager.add_interaction(
        step="Step 0: Task Initialization",
        user_input=last_user_query,
        agent_output=f"Plan to execute: {model_tasks}",
        notebook_context={"status": "pending_execution"}
    )
    logger.info(f"✅ Agent memory for {chat_id} created successfully with initial prompt.")

# In coding_agent.py, REPLACE the main function

def main():
    """Main execution loop with the duplicate call removed."""
    logger.info("🚀 Starting Professional Notebook Generator")
    logger.info("🔧 Features: Variable Reuse, Clean Structure, Context Awareness")
    logger.info("🧹 Anti-Redundancy: Enabled")
    
    # This initial file discovery is for logging purposes only.
    discovered_files = discover_data_files()
    if discovered_files.get('csv'):
        logger.info(f"📁 Found initial CSV files: {', '.join(discovered_files['csv'])}")

    while True:
        try:
            file, history, tasks, dataset_name = find_planned_chat()
            if file and tasks and dataset_name:
                chat_id = file.stem

                # --- START OF THE FIX ---
                # 1. Create the Memory Manager ONCE, here at the beginning.
                memory_manager = AgentMemoryManager(session_id=chat_id)

                # 2. Now, pass the created instance to the initialization function.
                initialize_agent_memory(chat_id, history, memory_manager)
                # --- END OF THE FIX ---

                # Mark task as in-progress to prevent re-execution by another thread.
                for obj in reversed(history):
                    if "task" in obj and obj["task"].get("status") == "planned":
                        obj["task"]["status"] = "in_progress"
                        break
                with open(file, "w", encoding="utf-8") as f:
                    json.dump(history, f, indent=2, ensure_ascii=False)

                # --- START OF THE FIX ---
                # 3. Pass the SAME instance to the execution function.
                execute_planned_task_with_memory(chat_id, tasks, dataset_name, memory_manager)
                # --- END OF THE FIX ---
                
                logger.info(f"✅ Full analysis process for chat {chat_id} has concluded.")

            else:
                time.sleep(5) 
        except KeyboardInterrupt:
            logger.info("👋 Shutting down gracefully...")
            break
        except Exception as e:
            logger.error(f"🚨 A fatal, unrecoverable error occurred in the main loop: {e}")
            traceback.print_exc()
            break # 

def test_rag_integration():
    """Test the RAG system integration"""
    try:
        kb = CodeKnowledgeBase()
        solutions = kb.search_solutions("y should be a 1d array", "machine learning")
        if solutions:
            logger.info("✅ RAG system working correctly")
            return True
        else:
            logger.warning("⚠️ RAG system initialized but no solutions found")
            return False
    except Exception as e:
        logger.error(f"❌ RAG system test failed: {e}")
        return False

# Add to main() function for testing
if __name__ == "__main__":
    if test_rag_integration():
        logger.info("🚀 Starting RAG-enhanced Professional Notebook Generator")

    main()

