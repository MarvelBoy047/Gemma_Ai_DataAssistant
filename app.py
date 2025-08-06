# app.py
import streamlit as st
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
import subprocess
import pandas as pd
import json
import time
import threading
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ValidationError
from ollama import chat
from coding_agent import main as coding_agent_main
import ollama
import re
# --- CONFIGURATION & PATHS ---
CHAT_HISTORY_DIR = Path("chat_history")
NOTEBOOKS_DIR = Path("notebooks")
CHAT_HISTORY_DIR.mkdir(exist_ok=True)
NOTEBOOKS_DIR.mkdir(exist_ok=True)


PLANNER_SYSTEM_PROMPT = """
[CONTEXT]
You are a pragmatic and cautious senior data analyst. Your goal is to create a simple, foundational analysis plan to start with. More advanced tasks can be added later if the user requests them.
And your name is Gemma.

[INSTRUCTIONS]
[YOUR TASK]
You will be given a user's request, context about a dataset (including its filename), and a large list of "AVAILABLE TOOLS". Your job is to create a SHORT, foundational analysis plan by selecting a logical sequence of tools from that list.

[RULES FOR THE INITIAL PLAN]
1.  **ALWAYS START WITH LOADING:** The very first task in your plan MUST be a tool for importing initial libraries (eg., "Import essential libraries...") and loading the dataset (e.g., "Automatically load dataset..."). This is because the file is available on disk but is NOT yet loaded into a DataFrame in memory. This is the most important rule.
2.  **BE CONCISE:** The plan MUST contain between 5 and 10 steps. DO NOT exceed 10 steps.
3.  **PRIORITIZE THE BASICS:** After loading, you MUST select foundational tasks. Prioritize tools for:
    - Basic inspection (info, shape, columns, dtypes)
    - Initial data cleaning (handling missing values, duplicates)
    - A simple baseline analysis (like descriptive statistics or a correlation matrix).
4.  **AVOID ADVANCED TASKS:** For this first plan, you MUST NOT select complex or advanced tools.
5.  **STICK TO THE LIST:** You MUST only use the exact descriptions provided in the "AVAILABLE TOOLS" list for your tasks.

[OUTPUT FORMAT]
You MUST respond with ONLY a valid JSON object.
{
  "reply": "A brief, friendly message to the user that MENTIONS the specific dataset filename you are about to analyze.",
  "tasks": ["description_of_selected_tool_1", "description_of_selected_tool_2", ...]
}
"""


class PlannerKnowledgeBase:
    """
    Manages loading, searching, and PERSISTING the tool knowledge base.
    It now saves the FAISS index to disk to avoid re-calculating it on every run.
    """
    def __init__(self, file_path="knowledge_base.json", index_path="planner_kb_index"):
        self.index_path = index_path
        self.tools = self._load_tools(file_path)

        if self.tools:
            self.descriptions = [tool['description'] for tool in self.tools]
            self.embeddings = OllamaEmbeddings(model="nomic-embed-text:v1.5")

            # --- START OF THE FIX ---
            # Check if a saved index already exists on disk
            if os.path.exists(self.index_path):
                # If it exists, load it directly (this is very fast)
                print(f"✅ Loading existing planner knowledge base from '{self.index_path}'...")
                self.vector_store = FAISS.load_local(
                    self.index_path, 
                    self.embeddings, 
                    allow_dangerous_deserialization=True # Required for loading pickled index
                )
            else:
                # If it doesn't exist, create it once and save it for future runs
                print("🧠 No existing index found. Building and saving new planner knowledge base...")
                self.vector_store = FAISS.from_texts(self.descriptions, self.embeddings)
                self.vector_store.save_local(self.index_path)
                print(f"✅ New planner knowledge base saved to '{self.index_path}'.")
            # --- END OF THE FIX ---

    def _load_tools(self, file_path) -> List[Dict]:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            st.error(f"Critical Error: The knowledge base file at '{file_path}' was not found or is corrupted.")
            return []

    def search_relevant_tools(self, query: str, k: int = 20) -> List[str]:
        """
        Searches the vector store for the most relevant tool descriptions.
        """
        if not hasattr(self, 'vector_store'):
            return []
        
        results = self.vector_store.similarity_search(query, k=k)
        return [doc.page_content for doc in results]

class Plan(BaseModel):
    reply: str = Field(description="A friendly, conversational reply to the user.")
    tasks: List[str] = Field(description="A list of 5 to 7 granular, single-action tasks.")

#--- STREAMLIT SESSION STATE INITIALIZATION ---
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "conversation_locked" not in st.session_state:
    st.session_state.conversation_locked = False
if "agent_thread" not in st.session_state:
    st.session_state.agent_thread = None
if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None
if "pending_plan" not in st.session_state:
    st.session_state.pending_plan = None
if "initial_loading_complete" not in st.session_state:
    st.session_state.initial_loading_complete = False

#--- CORE HELPER FUNCTIONS ---
def get_chat_sessions() -> List[str]:
    """Lists all available chat session IDs."""
    return sorted([p.stem for p in CHAT_HISTORY_DIR.glob("*.json")], reverse=True)

def load_chat_history_from_file(chat_id: str) -> List[Dict[str, Any]]:
    """Loads a specific chat history from its JSON file."""
    path = CHAT_HISTORY_DIR / f"{chat_id}.json"
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f: return json.load(f)
        except (json.JSONDecodeError, IOError): pass
    return []

def get_dataset_context() -> str:
    """
    If a dataset is uploaded for the current session, this function reads its
    name and column headers to create a context string for the LLM.
    """
    if st.session_state.get("uploaded_file_name") and st.session_state.get("current_chat_id"):
        file_path = NOTEBOOKS_DIR / st.session_state.current_chat_id / st.session_state.uploaded_file_name
        if file_path.exists():
            try:
                # Read only the first 5 rows to efficiently get columns
                df = pd.read_csv(file_path, nrows=5)
                columns = df.columns.tolist()
                context = (
                    f"The user has uploaded the following dataset. Use this information to create a specific and relevant analysis plan:\n"
                    f"- Filename: `{st.session_state.uploaded_file_name}`\n"
                    f"- Columns: `{columns}`\n\n"
                )
                return context
            except Exception as e:
                # Return an info string if the file can't be read
                return f"Info: A file named {st.session_state.uploaded_file_name} is uploaded, but I could not read its details. Error: {e}\n\n"
    return "" # Return empty string if no file is uploaded

def save_chat_history_to_file():
    """Saves the current chat history to its file if a session is active."""
    if st.session_state.current_chat_id:
        path = CHAT_HISTORY_DIR / f"{st.session_state.current_chat_id}.json"
    with open(path, "w", encoding="utf-8") as f: json.dump(st.session_state.chat_history, f, indent=2, ensure_ascii=False)

def check_system_health() -> tuple[bool, bool]:
    """
Checks if the Ollama server is running and whether GPU is physically available.
"""
    ollama_is_running = False
    gpu_is_active = False

    try:
        ollama.list()
        ollama_is_running = True
    except Exception:
        return False, False

    try:
        result = subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and "NVIDIA-SMI" in result.stdout:
            gpu_is_active = True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass # nvidia-smi not found or timed out, assume no GPU

    return ollama_is_running, gpu_is_active
OLLAMA_RUNNING, GPU_ACTIVE = check_system_health()

def start_new_chat():
    """Resets all state variables to start a new conversation."""
    st.session_state.current_chat_id = datetime.now().strftime('%Y%m%d%H%M%S')
    st.session_state.chat_history = []
    st.session_state.conversation_locked = False
    st.session_state.pending_plan = None
    st.session_state.uploaded_file_name = None
    save_chat_history_to_file()




def switch_chat_session(chat_id: str):
    """Switches the active chat session and resets planning state."""
    st.session_state.current_chat_id = chat_id
    st.session_state.chat_history = load_chat_history_from_file(chat_id)
    st.session_state.conversation_locked = any(obj.get("task", {}).get("status") in ["planned", "in_progress"] for obj in st.session_state.chat_history)
    st.session_state.pending_plan = None
    session_dir = NOTEBOOKS_DIR / chat_id
    if session_dir.exists():
        csv_files = list(session_dir.glob("*.csv"))
    if csv_files: st.session_state.uploaded_file_name = csv_files[0].name

#--- NEW: AGENT INTERACTION & PLANNING LOGIC ---
# In app.py, replace the existing get_llm_response function

def get_llm_response(prompt: str, dataset_context: str, available_tasks: List[str]) -> str:
    """
    Acts as the "Planner" by sending the user request and a list of
    available tools to the LLM to construct a plan.
    """
    # Format the list of available tasks so the LLM can read it
    tasks_formatted = "\n".join(f"- {task}" for task in available_tasks)

    full_prompt = f"""
{dataset_context}
---
**AVAILABLE TOOLS:**
{tasks_formatted}
---
**User Request:** "{prompt}"

Based on the user's request and the available tools, select the appropriate sequence of tasks to form a plan.
"""

    messages = [
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT}, # Using the new prompt
        {"role": "user", "content": full_prompt}
    ]
    try:
        response = ollama.chat(model="gemma3n:e2b", messages=messages)
        return response['message']['content']
    except Exception as e:
        st.error(f"Error communicating with Ollama: {e}")
        # Return a valid JSON with an error message
        return json.dumps({
            "reply": f"Sorry, I encountered an error while creating a plan: {e}",
            "tasks": []
        })

def load_knowledge_base_descriptions() -> List[str]:
    """
    Loads the knowledge_base.json file and returns a list of all
    the task descriptions.
    """
    try:
        with open("knowledge_base.json", "r", encoding="utf-8") as f:
            kb = json.load(f)
        # Return only the description of each tool
        return [item['description'] for item in kb]
    except (FileNotFoundError, json.JSONDecodeError) as e:
        st.error(f"Error loading knowledge base: {e}")
        return []

def extract_plan_from_response(response_text: str) -> Optional[Plan]:
    """
    Tries to find and validate a Plan object from the LLM's raw response text.
    """
    match = re.search(r"json\s*(\{.*?\})\s*", response_text, re.DOTALL)
    if not match:
        return None  # No JSON block found

    json_str = match.group(1)
    try:
        data = json.loads(json_str)
        plan = Plan(**data)
        if plan.tasks:
            return plan
    except (json.JSONDecodeError, ValidationError):
        return None
    return None

def handle_user_input(prompt: str):
    """
    Handles user input by fetching dataset context, getting a response from the LLM,
    and then processing the response for a potential plan.
    """
    if not st.session_state.current_chat_id:
        start_new_chat()

    st.session_state.chat_history.append({"user": {"query": prompt}})

    with st.spinner("Thinking..."):
        # 1. Get the cached knowledge base object
        planner_kb = get_planner_kb()
        if not planner_kb.tools:
            return # Stop if the knowledge base failed to load

        # 2. RAG Step: Search for the most relevant tools for the user's prompt
        relevant_tasks = planner_kb.search_relevant_tools(prompt)
        if not relevant_tasks:
            st.warning("Could not find any relevant tools for your request.")
            relevant_tasks = planner_kb.descriptions # Fallback to all if search fails

        # 3. Get context from the uploaded dataset
        dataset_context = get_dataset_context()

        # 4. Get the plan from the LLM, now providing only the short, relevant list of tasks
        raw_response = get_llm_response(prompt, dataset_context, relevant_tasks)

        # The rest of the function continues as it was before
        st.session_state.chat_history.append({"reply": raw_response})
        plan = extract_plan_from_response(raw_response)
        st.session_state.pending_plan = plan if plan else None

    save_chat_history_to_file()
    st.rerun()

def trigger_coding_agent():
    """
    Locks the conversation and adds the confirmed plan as a 'task' object
    to the chat history, which the background agent will then execute.
    """
    if st.session_state.pending_plan and st.session_state.current_chat_id:
        st.session_state.conversation_locked = True

        task_obj = {
            "task": {
                "tasks": st.session_state.pending_plan.tasks,
                "status": "planned",
                "dataset_name": st.session_state.uploaded_file_name
            }
        }
        st.session_state.chat_history.append(task_obj)
        st.session_state.pending_plan = None
        save_chat_history_to_file()
        st.rerun()


def run_initial_loading():
    """
    Displays a one-time loading bar while initializing backend components.
    """
    st.set_page_config(page_title="Loading...", layout="centered")
    st.title("🤖 Initializing Gemma Data Assistant...")
    
    loading_bar = st.progress(0, text="Waking up the agent...")
    
    # Simulate loading steps for a better user experience
    time.sleep(1)
    loading_bar.progress(33, text="Checking system health...")
    
    # This is where the actual work happens
    try:
        # Initialize the planner's knowledge base (this is the slow part)
        get_planner_kb()
        loading_bar.progress(100, text="Initialization complete!")
        time.sleep(1)
        
        # Set the flag to True so this doesn't run again
        st.session_state.initial_loading_complete = True
        
        # Force the app to rerun from the top, which will now skip the loading screen
        st.rerun()

    except Exception as e:
        st.error(f"Fatal error during initialization: {e}")
        st.error("Please check your Ollama connection and knowledge_base.json file, then refresh the page.")
        st.stop()


@st.cache_resource
def get_planner_kb():
    """Loads and caches the PlannerKnowledgeBase so it runs only once."""
    return PlannerKnowledgeBase()

def run_ui():
    st.set_page_config(page_title="Gemma data assistant", layout="wide", initial_sidebar_state="expanded")

    with st.sidebar:
        st.title("📝 Agent Controls")
        st.success("✅ Ollama Active")
        if GPU_ACTIVE:
            st.info("⚡ GPU Detected")
        else:
            st.warning("🐌 CPU Mode")

        st.markdown("---")

        if st.button("➕ New Chat", use_container_width=True):
            start_new_chat()
            st.rerun()

        st.markdown("---")
        st.subheader("Upload Dataset (Required)")
        uploaded_file = st.file_uploader("Upload a CSV file for this chat session.", type=["csv"])
        if uploaded_file:
            if not st.session_state.current_chat_id:
                start_new_chat()

            session_dir = NOTEBOOKS_DIR / st.session_state.current_chat_id
            session_dir.mkdir(exist_ok=True)
            file_path = session_dir / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            if st.session_state.uploaded_file_name != uploaded_file.name:
                st.session_state.uploaded_file_name = uploaded_file.name
                st.success(f"Loaded `{uploaded_file.name}`.")
                time.sleep(1)
                st.rerun()

        st.markdown("---")
        st.subheader("Chat History")
        for session_id in get_chat_sessions():
            button_type = "primary" if session_id == st.session_state.current_chat_id else "secondary"
            if st.button(session_id, key=session_id, use_container_width=True, type=button_type):
                switch_chat_session(session_id)
                st.rerun()

    st.title("🤖 Gemma data assistant")
    st.caption("An offline AI agent for data analysis.")

    if not st.session_state.current_chat_id:
        st.info("Start a new chat or upload a dataset to begin.")
        return

    # --- START OF UI FIX ---
    for message in st.session_state.chat_history:
        if "user" in message:
            with st.chat_message("user"):
                st.markdown(message["user"]["query"])

        # --- START OF UI FIX ---
        elif "reply" in message:
            with st.chat_message("assistant"):
                reply_content = message["reply"]
                try:
                    # Clean the string if it's wrapped in a markdown code block
                    if "```json" in reply_content:
                        json_str_match = re.search(r"\{.*\}", reply_content, re.DOTALL)
                        if json_str_match:
                            reply_content = json_str_match.group(0)
        
                    # Parse the JSON
                    data = json.loads(reply_content)
                    
                    # Display the conversational part of the reply
                    if "reply" in data:
                        st.markdown(data["reply"])
                    
                    # --- NEW: Display the list of tasks ---
                    if "tasks" in data and data["tasks"]:
                        st.info("Here is the plan I will follow:")
                        task_list = "\n".join(f"1. {task}" for task in data["tasks"])
                        st.markdown(task_list)
        
                except (json.JSONDecodeError, AttributeError):
                    # If it's not a JSON string, display it as is
                    st.markdown(reply_content)
        # --- END OF UI FIX ---
        # --- START OF UI FIX ---
        elif "conclusion" in message:
            with st.chat_message("assistant"):
                st.success("✅ **Analysis Complete!** Here is the final summary:")
                st.markdown(message["conclusion"])
        
                # --- NEW: Show Notebook Expander ---
                notebook_path = NOTEBOOKS_DIR / st.session_state.current_chat_id / f"{st.session_state.current_chat_id}.ipynb"
                if notebook_path.exists():
                    with st.expander("🔬 View Generated Notebook"):
                        try:
                            # Read the notebook file
                            with open(notebook_path, "r", encoding="utf-8") as f:
                                nb_content = json.load(f)
        
                            # Loop through each cell in the notebook
                            for cell in nb_content.get("cells", []):
                                if cell["cell_type"] == "markdown":
                                    # Display markdown cells
                                    st.markdown("".join(cell["source"]), unsafe_allow_html=True)
                                elif cell["cell_type"] == "code":
                                    # Display code cells
                                    st.code("".join(cell["source"]), language="python")
                                    # Display cell outputs
                                    for output in cell.get("outputs", []):
                                        if output["output_type"] == "stream":
                                            st.text("".join(output["text"]))
                                        elif output["output_type"] == "execute_result":
                                            st.text("".join(output["data"].get("text/plain", [])))
                                        elif output["output_type"] == "error":
                                            st.error("".join(output["traceback"]))
                            st.info(f"Notebook file path: {notebook_path}")
                        except Exception as e:
                            st.error(f"Could not display notebook. Error: {e}")

        elif "task" in message and message["task"]["status"] in ["planned", "in_progress"]:
            with st.chat_message("assistant"):
                st.info("The coding agent is now working on the approved plan.")

    if st.session_state.pending_plan:
        st.info("A plan has been detected. Do you want to proceed?")
        col1, col2, _ = st.columns([1, 1, 3])
        with col1:
            if st.button("✅ Proceed", use_container_width=True, type="primary"):
                trigger_coding_agent()
        with col2:
            if st.button("❌ Cancel", use_container_width=True):
                st.session_state.pending_plan = None
                st.rerun()

    # --- START OF FINAL UI FIX ---
    if st.session_state.conversation_locked:
        # First, check the latest status from the file to see if the agent is already done.
        history = load_chat_history_from_file(st.session_state.current_chat_id)
        is_done = False
        final_status_text = ""
        for obj in reversed(history):
            if "conclusion" in obj or (obj.get("progress", 0) == 100):
                is_done = True
                final_status_text = obj.get("status", "completed: Analysis finished.")
                break
            
        # If the agent is NOT done, display the progress bar and poll for updates.
        if not is_done:
            progress_bar = st.progress(0, text="Waiting for agent to start...")
            
            # This loop will now continue until the agent is finished.
            while True:
                # Load the latest history from the file
                latest_history = load_chat_history_from_file(st.session_state.current_chat_id)
                
                # --- THIS IS THE CRITICAL FIX ---
                # Update the session state directly with the new history
                st.session_state.chat_history = latest_history
                
                is_done_in_loop = False
                latest_progress = 0
                latest_status = "working..."
    
                for obj in reversed(latest_history):
                    if "progress" in obj:
                        latest_progress = obj.get("progress", 0)
                        latest_status = obj.get("status", "working...")
                        if "completed" in latest_status.lower() or "failed" in latest_status.lower():
                            is_done_in_loop = True
                        break
                    elif "conclusion" in obj:
                        is_done_in_loop = True
                        break
                    
                progress_bar.progress(latest_progress, text=f"Step: {latest_status}")
                
                if is_done_in_loop:
                    # Force one final rerun to draw the conclusion message
                    st.rerun()
                
                time.sleep(1)
        
        # If the agent IS done, the main chat display will now show the conclusion.
        # We just need to display the final status message and the unlock button.
        else:
            if "failed" in final_status_text.lower():
                error_details = final_status_text.split(":", 1)[-1].strip()
                st.error(f"❌ Analysis Failed: {error_details}")
            else:
                st.success("✅ Analysis Complete!")
    
            if st.button("Acknowledge and Unlock Chat"):
                st.session_state.conversation_locked = False
                st.rerun()
    # --- END OF FINAL UI FIX ---

    is_disabled = st.session_state.conversation_locked
    if prompt := st.chat_input("Ask for an analysis...", disabled=is_disabled):
        handle_user_input(prompt)


if __name__ == "__main__":
    # First, check if the one-time initial loading is complete.
    if not st.session_state.get("initial_loading_complete", False):
        # If not, run the loading function. This function will set the flag
        # and then rerun the app, so the code below won't execute this time.
        run_initial_loading()
    
    # The code below will only run AFTER the initial loading is complete.
    if not OLLAMA_RUNNING:
        st.set_page_config(page_title="Fatal Error", layout="centered")
        st.title("❌ Connection Error")
        st.error(
            """
            **Could not connect to the Ollama service.**
            Please ensure that the Ollama application is running on your system.
            You can start it by running `ollama serve` in your terminal.
            Once the service is active, please refresh this page.
            """
        )
        st.stop()

    if "agent_thread" not in st.session_state:
        agent_thread = threading.Thread(target=coding_agent_main, daemon=True)
        agent_thread.start()
        st.session_state.agent_thread = agent_thread

    run_ui()

# --- END OF STEP 2 ---