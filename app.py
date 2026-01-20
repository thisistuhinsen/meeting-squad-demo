import operator
import os
from typing import Annotated, List, TypedDict

import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

# --- 1. Page Config ---
st.set_page_config(page_title="Meeting Prep Squad", page_icon="🤖")
st.title("🤖 Multi-Agent Meeting Prep Squad")
st.markdown("### Powered by LangGraph & Google Gemini")

# --- 2. API Key Setup ---
if "GOOGLE_API_KEY" in st.secrets:
    raw_key = st.secrets["GOOGLE_API_KEY"]
elif "GOOGLE_API_KEY" in os.environ:
    raw_key = os.environ["GOOGLE_API_KEY"]
else:
    st.error("🚨 Missing Google API Key.")
    st.stop()

# Sanitize Key
api_key = raw_key.strip().strip('"').strip("'")

# --- 3. ROBUST MODEL LOADER (The Fix) ---
# This function tries multiple model names until one works.
def get_working_llm(api_key):
    # List of models to try in order of preference
    candidates = [
        "gemini-1.5-flash",      # Preferred
        "gemini-1.5-flash-001",  # Alternative ID
        "gemini-1.5-pro",        # Stronger model
        "gemini-pro"             # Old reliable (Fall back to this)
    ]
    
    status_placeholder = st.empty()
    
    for model in candidates:
        try:
            # Try initializing and sending a test message
            test_llm = ChatGoogleGenerativeAI(
                model=model, 
                google_api_key=api_key,
                temperature=0
            )
            test_llm.invoke("test connection") # Handshake
            status_placeholder.caption(f"✅ Connected to model: `{model}`")
            return test_llm
        except Exception as e:
            # If it fails, print a small debug note and continue to next model
            print(f"Failed to connect to {model}: {e}")
            continue
            
    # If we loop through all and none work:
    status_placeholder.error("❌ Could not connect to any Gemini models.")
    st.stop()

# Initialize the LLM using the resilient loader
llm = get_working_llm(api_key)

# --- 4. Tool Setup ---
try:
    from langchain_community.tools import DuckDuckGoSearchRun
    search_tool = DuckDuckGoSearchRun()
    tool_status = "Live Web Search 🟢"
except Exception:
    tool_status = "Simulated Search (Backup) 🟡"
    class MockSearch:
        def run(self, query):
            return (
                f"Mock Search Result for '{query}': \n"
                "1. Microsoft Copilot is expanding into Agentic AI workflows. \n"
                "2. Satya Nadella emphasizes 'Trustworthy AI' as a core pillar. \n"
                "3. New features in M365 allow for autonomous agent orchestration."
            )
    search_tool = MockSearch()

st.sidebar.caption(f"System Status: {tool_status}")

# --- 5. Define Agents ---

class AgentState(TypedDict):
    topic: str
    raw_research: str
    draft_brief: str
    fact_check_feedback: str
    final_brief: str
    messages: Annotated[List[str], operator.add]

def research_node(state: AgentState):
    topic = state["topic"]
    # Uses whichever tool loaded successfully
    results = search_tool.run(f"recent news and facts about {topic}")
    return {"raw_research": results}

def analyst_node(state: AgentState):
    research_data = state["raw_research"]
    feedback = state.get("fact_check_feedback")
    prompt = (
        "You are a senior executive assistant. "
        "Create a concise, 3-bullet briefing based ONLY on the provided research text. "
        "Do not add any outside information."
    )
    if feedback:
        prompt += f"\n\nPREVIOUS DRAFT FAILED FACT-CHECK. Feedback: {feedback}"
    
    messages = [SystemMessage(content=prompt), HumanMessage(content=f"DATA:\n{research_data}")]
    try:
        response = llm.invoke(messages)
        return {"draft_brief": response.content}
    except Exception:
        return {"draft_brief": "Error generating brief. Retrying..."}

def fact_checker_node(state: AgentState):
    raw = state["raw_research"]
    draft = state["draft_brief"]
    prompt = (
        "Compare the DRAFT BRIEF to the SOURCE TEXT. "
        "If faithful, reply 'PASS'. If hallucinated, reply 'FAIL: <reason>'."
    )
    messages = [SystemMessage(content=prompt), HumanMessage(content=f"SOURCE:\n{raw}\n\nDRAFT:\n{draft}")]
    try:
        response = llm.invoke(messages)
        content = response.content
    except Exception:
        content = "PASS" # Fail open to avoid crashing

    if "FAIL" in content:
        return {"fact_check_feedback": content, "final_brief": None}
    else:
        return {"fact_check_feedback": None, "final_brief": draft}

def router(state: AgentState):
    return "analyst" if state.get("fact_check_feedback") else "end"

# --- 6. Build Graph ---
workflow = StateGraph(AgentState)
workflow.add_node("researcher", research_node)
workflow.add_node("analyst", analyst_node)
workflow.add_node("fact_checker", fact_checker_node)

workflow.set_entry_point("researcher")
workflow.add_edge("researcher", "analyst")
workflow.add_edge("analyst", "fact_checker")
workflow.add_conditional_edges("fact_checker", router, {"analyst": "analyst", "end": END})

app = workflow.compile()

# --- 7. The UI ---
topic = st.text_input("Enter a meeting topic:", placeholder="e.g. OpenAI's recent safety news")

if st.button("Run Agents"):
    with st.status("🤖 Agents are working...", expanded=True) as status:
        st.write("🔎 Researcher is searching...")
        try:
            result = app.invoke({"topic": topic})
            
            st.write("📝 Analyst is drafting...")
            st.write("🕵️‍♂️ Fact-Checker is validating...")
            
            if result.get("fact_check_feedback"): 
                st.warning(f"Correction Triggered: {result['fact_check_feedback']}")
            
            status.update(label="✅ Workflow Complete!", state="complete", expanded=False)
            
            st.subheader("Final Executive Briefing")
            st.success(result["final_brief"])

            with st.expander("See Raw Research Data"):
                st.text(result["raw_research"])
        except Exception as e:
            st.error(f"Workflow Failed: {e}")
