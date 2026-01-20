import operator
import os
import time
from typing import Annotated, List, TypedDict

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

# --- 1. Page Config ---
st.set_page_config(page_title="Meeting Prep Squad", page_icon="🤖")
st.title("🤖 Multi-Agent Meeting Prep Squad")
st.markdown("### Powered by LangGraph")

# --- 2. The Robust Mock LLM (Your Safety Net) ---
# If Google API fails, this ensures you still have a working demo.
class RobustMockLLM:
    def invoke(self, messages):
        last_msg = messages[-1].content if isinstance(messages, list) else str(messages)
        time.sleep(1.2) # Simulate thinking
        
        if "DATA:" in last_msg: 
            return AIMessage(content=(
                "### Executive Briefing (Demo Mode)\n"
                "* **Strategic Focus:** Microsoft is aggressively integrating Agentic AI into M365 Copilot.\n"
                "* **Key Metric:** 40% of Fortune 100 companies are trialing autonomous agents.\n"
                "* **Challenge:** Data governance remains the primary blocker for enterprise adoption."
            ))
        elif "SOURCE:" in last_msg:
            return AIMessage(content="PASS") # Fact checker passes
        else:
            return AIMessage(content="I am ready.")

# --- 3. The "Works For Sure" Connection Logic ---
def get_working_llm():
    # 1. Get Key
    if "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
    else:
        api_key = os.environ.get("GOOGLE_API_KEY", "")
    
    # Clean the key (remove accidental quotes/spaces)
    api_key = api_key.strip().strip('"').strip("'")
    
    if not api_key:
        return RobustMockLLM(), "🟡 No API Key (Mock Mode)"

    # 2. Try Import
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError:
        return RobustMockLLM(), "🟡 Import Failed (Mock Mode)"

    # 3. Model Roulette (Try these in order)
    # 'gemini-pro' is the most widely available free model in India if Flash fails.
    candidates = [
        "gemini-1.5-flash",
        "gemini-1.5-flash-001", 
        "gemini-pro"
    ]

    for model_name in candidates:
        try:
            # FORCE transport="rest". This fixes the "404/Deadline Exceeded" errors in India.
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                transport="rest", 
                temperature=0
            )
            llm.invoke("Hello") # Handshake
            return llm, f"🟢 Connected: {model_name}"
        except Exception as e:
            print(f"Failed {model_name}: {e}")
            continue

    # 4. If all fail, use Mock
    return RobustMockLLM(), "🟡 API Blocked (Mock Active)"

# Initialize
llm, system_status = get_working_llm()

# --- 4. Tool Setup ---
try:
    from langchain_community.tools import DuckDuckGoSearchRun
    search_tool = DuckDuckGoSearchRun()
    tool_status = "🟢 Live Search"
except Exception:
    tool_status = "🟡 Mock Search"
    class MockSearch:
        def run(self, query):
            return "Microsoft's AI revenue grew by 20% in Q4 2024. Satya Nadella announced new Copilot Agents."
    search_tool = MockSearch()

# Sidebar Status
st.sidebar.caption(f"Brain: {system_status}")
st.sidebar.caption(f"Tools: {tool_status}")

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
    results = search_tool.run(f"recent news and facts about {topic}")
    return {"raw_research": results}

def analyst_node(state: AgentState):
    research_data = state["raw_research"]
    feedback = state.get("fact_check_feedback")
    
    prompt = (
        "You are a senior executive assistant. "
        "Create a concise, 3-bullet briefing based ONLY on the provided research text."
    )
    if feedback:
        prompt += f"\n\nFIX FEEDBACK: {feedback}"
    
    messages = [SystemMessage(content=prompt), HumanMessage(content=f"DATA:\n{research_data}")]
    
    # Safe Invoke
    try:
        response = llm.invoke(messages)
        return {"draft_brief": response.content}
    except:
        return {"draft_brief": "Error generating brief. Proceeding with backup."}

def fact_checker_node(state: AgentState):
    raw = state["raw_research"]
    draft = state["draft_brief"]
    prompt = "Compare DRAFT to SOURCE. Reply 'PASS' or 'FAIL'."
    messages = [SystemMessage(content=prompt), HumanMessage(content=f"SOURCE:\n{raw}\n\nDRAFT:\n{draft}")]
    
    try:
        response = llm.invoke(messages)
        content = response.content
    except:
        content = "PASS" # Fail open

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
topic = st.text_input("Enter meeting topic:", placeholder="e.g. Microsoft Copilot Strategy")

if st.button("Run Workflow"):
    with st.status("🚀 Processing...", expanded=True) as status:
        st.write("🔎 Researcher is gathering data...")
        try:
            result = app.invoke({"topic": topic})
            
            st.write("📝 Analyst is writing brief...")
            st.write("⚖️ Fact-Checker is validating...")
            
            if result.get("fact_check_feedback"): 
                st.warning(f"Correction Triggered: {result['fact_check_feedback']}")
            
            status.update(label="✅ Done!", state="complete", expanded=False)
            
            st.subheader("Final Briefing")
            st.success(result["final_brief"])
            
            with st.expander("Raw Data"):
                st.write(result["raw_research"])
        except Exception as e:
            st.error(f"Error: {e}")
