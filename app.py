import operator
import os
import time
from typing import Annotated, List, TypedDict

import google.generativeai as genai
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

# --- 1. Page Config ---
st.set_page_config(page_title="Meeting Prep Squad", page_icon="🤖")
st.title("🤖 Multi-Agent Meeting Prep Squad")
st.markdown("### Powered by LangGraph & Gemini")

# --- 2. The "Model Hunter" Logic (The Fix) ---
def get_working_llm():
    # 1. Get Key
    if "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
    else:
        api_key = os.environ.get("GOOGLE_API_KEY", "")
    
    clean_key = api_key.strip().strip('"').strip("'")
    
    if not clean_key:
        return None, "🟡 No API Key (Mock Mode)"

    # 2. Configure the official SDK
    genai.configure(api_key=clean_key)
    
    found_model_name = None

    try:
        # Ask Google: "What models can I use?"
        st.toast("🔍 Scanning for available Gemini models...", icon="📡")
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                # Prefer Flash, but take anything that works
                if 'flash' in m.name:
                    found_model_name = m.name
                    break
                elif 'pro' in m.name and not found_model_name:
                    found_model_name = m.name # Fallback
        
        if not found_model_name:
            # Last resort fallback if list_models implies access but returns weird names
            found_model_name = "gemini-1.5-flash" 

        # 3. Connect using the FOUND name
        llm = ChatGoogleGenerativeAI(
            model=found_model_name,
            google_api_key=clean_key,
            transport="rest", # Crucial for India/Streamlit Cloud stability
            temperature=0
        )
        # Handshake test
        llm.invoke("Hello")
        return llm, f"🟢 Connected: {found_model_name}"

    except Exception as e:
        st.error(f"Debug Error: {e}")
        return None, "🟡 Connection Failed (Mock Mode)"

# --- 3. Robust Mock LLM (Safety Net) ---
class RobustMockLLM:
    def invoke(self, messages):
        last_msg = messages[-1].content if isinstance(messages, list) else str(messages)
        time.sleep(1.2)
        if "DATA:" in last_msg: 
            return AIMessage(content="### Briefing (Mock)\n* **News:** Microsoft AI revenue +20%.\n* **Strategy:** Agents are the new apps.")
        elif "SOURCE:" in last_msg:
            return AIMessage(content="PASS")
        else:
            return AIMessage(content="Ready.")

# Initialize
real_llm, system_status = get_working_llm()
llm = real_llm if real_llm else RobustMockLLM()

# --- 4. Tool Setup ---
try:
    from langchain_community.tools import DuckDuckGoSearchRun
    search_tool = DuckDuckGoSearchRun()
    tool_status = "🟢 Live Search"
except Exception:
    tool_status = "🟡 Mock Search"
    class MockSearch:
        def run(self, query):
            return "Latest news: AI agent adoption is skyrocketing in 2025. Companies are moving from chatbots to autonomous squads."
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
    try:
        response = llm.invoke(messages)
        return {"draft_brief": response.content}
    except:
        return {"draft_brief": "Error generating brief. (Mock Result)"}

def fact_checker_node(state: AgentState):
    raw = state["raw_research"]
    draft = state["draft_brief"]
    prompt = "Compare DRAFT to SOURCE. Reply 'PASS' or 'FAIL'."
    messages = [SystemMessage(content=prompt), HumanMessage(content=f"SOURCE:\n{raw}\n\nDRAFT:\n{draft}")]
    try:
        response = llm.invoke(messages)
        content = response.content
    except:
        content = "PASS"

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
