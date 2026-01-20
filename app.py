import operator
import os
from typing import Annotated, List, TypedDict

import streamlit as st
from duckduckgo_search import DDGS
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

# --- 1. Page Config ---
st.set_page_config(page_title="Deep Intelligence Squad", page_icon="🧠", layout="wide")
st.title("🧠 Deep Intelligence Engine")
st.markdown("### Powered by Multi-Agent Orchestration (Hybrid Search)")

# --- 2. Imports ---
try:
    import google.generativeai as genai
    from langchain_google_genai import (ChatGoogleGenerativeAI,
                                        HarmBlockThreshold, HarmCategory)
except ImportError as e:
    st.error(f"Missing Libraries: {e}")
    st.stop()

# --- 3. AUTO-CONNECT (The Proven Fix) ---
def connect_to_google():
    if "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
    else:
        api_key = os.environ.get("GOOGLE_API_KEY", "")
    
    clean_key = api_key.strip().strip('"').strip("'")
    if not clean_key:
        st.error("🚨 No API Key found.")
        st.stop()

    genai.configure(api_key=clean_key)
    
    # Get Valid Models
    valid_models = []
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                valid_models.append(m.name)
    except Exception as e:
        st.error(f"Connection Failed: {e}")
        st.stop()

    if not valid_models:
        st.error("🚨 No valid models found for this API key.")
        st.stop()

    # Pick Best Model
    selected_model = valid_models[0] 
    for m in valid_models:
        if "flash" in m and "1.5" in m:
            selected_model = m
            break
    
    st.sidebar.success(f"Connected to: **{selected_model}**")

    # Connect
    safety_config = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    llm = ChatGoogleGenerativeAI(
        model=selected_model,
        google_api_key=clean_key,
        transport="rest",
        safety_settings=safety_config,
        temperature=0
    )
    return llm

# Initialize
llm = connect_to_google()

# --- 4. HYBRID SEARCH TOOL (The Fix for "No Results") ---
class HybridSearchTool:
    def run(self, query):
        # STRATEGY 1: Live Web Search
        try:
            with st.spinner(f"🌐 Scouring DuckDuckGo for '{query}'..."):
                results = DDGS().text(query, max_results=5)
                if results:
                    return f"**SOURCE: LIVE WEB SEARCH**\n\n{str(results)}"
        except Exception:
            pass # Silent fail -> Go to fallback
        
        # STRATEGY 2: Gemini Internal Knowledge (The Safety Net)
        # If web search fails/returns empty, we generate the data using the LLM itself.
        try:
            with st.spinner("Web search blocked. Retrieving from Internal Knowledge Base..."):
                prompt = (
                    f"You are a raw data collector. Provide a detailed, factual dump of information regarding: '{query}'. "
                    "Include key dates, entities, and technical details. Do not summarize yet, just provide the raw facts."
                )
                response = llm.invoke(prompt)
                return f"**SOURCE: GEMINI INTERNAL KNOWLEDGE (Hybrid Fallback)**\n\n{response.content}"
        except Exception as e:
            return f"CRITICAL ERROR: {e}"

search_tool = HybridSearchTool()

# --- 5. Define Agents ---
class AgentState(TypedDict):
    topic: str
    raw_research: str
    draft_brief: str
    fact_check_feedback: str
    final_brief: str
    messages: Annotated[List[str], operator.add]

def research_node(state: AgentState):
    """Researcher: Gets data from Web OR Internal Knowledge"""
    topic = state["topic"]
    results = search_tool.run(topic)
    return {"raw_research": results}

def analyst_node(state: AgentState):
    """Analyst: Summarizes the data (whatever it is)"""
    research_data = state["raw_research"]
    feedback = state.get("fact_check_feedback")
    
    prompt = (
        "You are a Senior Intelligence Analyst. "
        "Write a structured Intelligence Report based **ONLY** on the provided data. "
        "Include 5 key strategic insights."
    )
    if feedback:
        prompt += f"\n\n🚨 FIX AUDIT FEEDBACK: {feedback}"
    
    messages = [SystemMessage(content=prompt), HumanMessage(content=f"DATA:\n{research_data}")]
    
    response = llm.invoke(messages)
    return {"draft_brief": response.content}

def fact_checker_node(state: AgentState):
    """Auditor: Verifies consistency"""
    raw = state["raw_research"]
    draft = state["draft_brief"]
    
    prompt = "Compare REPORT to SOURCE. Reply 'PASS' or 'FAIL: <reason>'."
    messages = [SystemMessage(content=prompt), HumanMessage(content=f"SOURCE:\n{raw}\n\nREPORT:\n{draft}")]
    
    response = llm.invoke(messages)
    if "FAIL" in response.content:
        return {"fact_check_feedback": response.content, "final_brief": None}
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

# --- 7. UI ---
topic = st.text_input("Enter Topic:", placeholder="e.g. NVIDIA AI Strategy")

if st.button("Generate Intelligence"):
    with st.status("🚀 Engine Running...", expanded=True) as status:
        st.write("🌍 Researcher gathering intelligence...")
        try:
            result = app.invoke({"topic": topic})
            st.write("🧠 Analyst processing...")
            st.write("🛡️ Verifying...")
            
            if result.get("fact_check_feedback"): 
                st.warning(f"Correction: {result['fact_check_feedback']}")
            
            status.update(label="✅ Complete!", state="complete", expanded=False)
            st.markdown(result["final_brief"])
            with st.expander("View Source Data"):
                # You will now see data here even if DDG fails!
                st.write(result["raw_research"])
        except Exception as e:
            st.error(f"❌ EXECUTION FAILED: {e}")
