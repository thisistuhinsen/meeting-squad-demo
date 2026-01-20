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
st.markdown("### Powered by LangGraph & Gemini (Real Data Mode)")

# --- 2. Safe Imports ---
try:
    import google.generativeai as genai
    from langchain_google_genai import (ChatGoogleGenerativeAI,
                                        HarmBlockThreshold, HarmCategory)
    LIBS_INSTALLED = True
except ImportError:
    st.error("⚠️ Libraries missing. Please Reboot App.")
    st.stop()

# --- 3. The Logic to Find a Working Model ---
def get_working_llm():
    # 1. Get Key
    if "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
    else:
        api_key = os.environ.get("GOOGLE_API_KEY", "")
    
    clean_key = api_key.strip().strip('"').strip("'")
    
    if not clean_key:
        st.error("🚨 No API Key found.")
        st.stop()

    # 2. Configure & Connect
    try:
        genai.configure(api_key=clean_key)
        
        # Hunt for a working model name
        found_model = "gemini-1.5-flash" # Default
        try:
            # Quick scan for valid models
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    if 'flash' in m.name:
                        found_model = m.name
                        break
        except:
            pass 

        # 3. SAFETY SETTINGS (The Fix for "Silent Failures")
        # We allow ALL content so the model doesn't block innocent news summaries.
        safety_config = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        llm = ChatGoogleGenerativeAI(
            model=found_model,
            google_api_key=clean_key,
            transport="rest", # REST is stable
            temperature=0,
            safety_settings=safety_config
        )
        # Real Handshake
        llm.invoke("Hello") 
        return llm, f"🟢 Connected: {found_model}"

    except Exception as e:
        return None, f"🔴 Error: {e}"

# Initialize System
llm, system_status = get_working_llm()

# --- 4. Tool Setup ---
try:
    from langchain_community.tools import DuckDuckGoSearchRun
    search_tool = DuckDuckGoSearchRun()
    tool_status = "🟢 Live Search"
except Exception:
    tool_status = "🔴 Search Failed"
    # Fallback only if import fails
    class MockSearch:
        def run(self, query): return "Search Tool Broken."
    search_tool = MockSearch()

st.sidebar.caption(f"Brain: {system_status}")
st.sidebar.caption(f"Tools: {tool_status}")

# --- 5. Define Agents (NO MOCKING ALLOWED) ---
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
        "Create a concise, 3-bullet briefing based ONLY on the provided research text. "
        "Do not add any outside information."
    )
    if feedback:
        prompt += f"\n\nFIX FEEDBACK: {feedback}"
    
    messages = [SystemMessage(content=prompt), HumanMessage(content=f"DATA:\n{research_data}")]
    
    # DIRECT CALL - No Try/Except to hide errors
    response = llm.invoke(messages)
    return {"draft_brief": response.content}

def fact_checker_node(state: AgentState):
    raw = state["raw_research"]
    draft = state["draft_brief"]
    prompt = "Compare DRAFT to SOURCE. Reply 'PASS' or 'FAIL'."
    messages = [SystemMessage(content=prompt), HumanMessage(content=f"SOURCE:\n{raw}\n\nDRAFT:\n{draft}")]
    
    # DIRECT CALL - No Try/Except
    response = llm.invoke(messages)
    content = response.content

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
topic = st.text_input("Enter meeting topic:", placeholder="e.g. OpenAI's recent safety news")

if st.button("Run Workflow"):
    if not llm:
        st.error("Cannot run: LLM not connected.")
        st.stop()

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
            st.error(f"Execution Error: {e}")
            st.info("If this error mentions 'Safety', the model blocked the news content.")
