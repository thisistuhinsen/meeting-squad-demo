import operator
import os
import time
from typing import Annotated, List, TypedDict

import streamlit as st
from duckduckgo_search import DDGS
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

# --- 1. Page Config ---
st.set_page_config(page_title="Deep Intelligence Squad", page_icon="🧠", layout="wide")
st.title("🧠 Deep Intelligence Engine")
st.markdown("### Powered by Multi-Agent Orchestration")

# --- 2. Safe Imports ---
try:
    import google.generativeai as genai
    from langchain_google_genai import (ChatGoogleGenerativeAI,
                                        HarmBlockThreshold, HarmCategory)
    LIBS_INSTALLED = True
except ImportError as e:
    st.error(f"⚠️ Missing Libraries: {e}. Please update requirements.txt and Reboot.")
    st.stop()

# --- 3. THE FIX: Dynamic Model Discovery ---
def get_dynamic_llm():
    """
    Instead of guessing 'gemini-1.5-flash', this function asks Google:
    'What models are actually available to this API Key?' and picks the best one.
    """
    # 1. Get Key
    if "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
    else:
        api_key = os.environ.get("GOOGLE_API_KEY", "")
    
    clean_key = api_key.strip().strip('"').strip("'")
    if not clean_key:
        st.error("🚨 No API Key found in Secrets.")
        st.stop()

    try:
        # 2. Configure GenAI
        genai.configure(api_key=clean_key)
        
        # 3. List Available Models (The Truth Source)
        available_models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                available_models.append(m.name)
        
        # 4. Pick the Best Match
        selected_model = None
        # Priority 1: Flash (Fastest)
        for m in available_models:
            if "flash" in m and "1.5" in m:
                selected_model = m
                break
        
        # Priority 2: Pro (Standard)
        if not selected_model:
            for m in available_models:
                if "pro" in m and "1.5" in m:
                    selected_model = m
                    break
        
        # Priority 3: Any Gemin Pro
        if not selected_model:
            for m in available_models:
                if "gemini-pro" in m:
                    selected_model = m
                    break
        
        # Priority 4: Whatever is first in the list
        if not selected_model and available_models:
            selected_model = available_models[0]
            
        if not selected_model:
            return None, "🔴 No Models Found (Check API Key Quota)"

        # 5. Connect to the SPECIFIC found model
        # Disable safety to prevent blocking news
        safety_config = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        llm = ChatGoogleGenerativeAI(
            model=selected_model,
            google_api_key=clean_key,
            transport="rest", # REST is safer for Cloud
            safety_settings=safety_config
        )
        llm.invoke("Hello") 
        return llm, f"🟢 Connected: {selected_model}"

    except Exception as e:
        return None, f"🔴 Connection Error: {e}"

# Initialize
llm, system_status = get_dynamic_llm()

# --- 4. Deep Search Tool ---
class DeepSearchTool:
    def run(self, query):
        # 1. Try DuckDuckGo
        try:
            with st.spinner(f"🌐 Scouring the web for '{query}'..."):
                results = DDGS().text(query, max_results=10)
                if results:
                    return f"**SOURCE: LIVE WEB SEARCH**\n\n{str(results)}"
        except Exception:
            pass # Silent fail to fallback
        
        # 2. Fallback to Gemini Internal Knowledge
        try:
            with st.spinner("Live search unavailable. Accessing Internal Knowledge..."):
                prompt = f"Provide a detailed, factual intelligence report on: '{query}'. Output raw facts only."
                response = llm.invoke(prompt)
                return f"**SOURCE: GEMINI INTERNAL KNOWLEDGE**\n\n{response.content}"
        except Exception as e:
            return f"Error: {e}"

search_tool = DeepSearchTool()

# Sidebar Status
st.sidebar.caption(f"Brain: {system_status}")

# --- 5. Define Agents ---
class AgentState(TypedDict):
    topic: str
    raw_research: str
    draft_brief: str
    fact_check_feedback: str
    final_brief: str
    messages: Annotated[List[str], operator.add]

def research_node(state: AgentState):
    """Agent 1: Researcher"""
    topic = state["topic"]
    results = search_tool.run(topic)
    return {"raw_research": results}

def analyst_node(state: AgentState):
    """Agent 2: Analyst"""
    research_data = state["raw_research"]
    feedback = state.get("fact_check_feedback")
    
    prompt = (
        "You are a Chief Intelligence Officer. "
        "Produce a **Comprehensive Intelligence Report** (Markdown). "
        "Include at least **10 key insights** categorized logically. "
        "Strictly adhere to the facts provided."
    )
    if feedback:
        prompt += f"\n\n🚨 FIX AUDIT ISSUES: {feedback}"
    
    messages = [SystemMessage(content=prompt), HumanMessage(content=f"DATA:\n{research_data}")]
    response = llm.invoke(messages)
    return {"draft_brief": response.content}

def fact_checker_node(state: AgentState):
    """Agent 3: Auditor"""
    raw = state["raw_research"]
    draft = state["draft_brief"]
    
    prompt = (
        "Compare the REPORT to the SOURCE. "
        "If it contains hallucinations, reply 'FAIL: <reason>'. "
        "If faithful, reply 'PASS'."
    )
    
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

if st.button("Generate Report"):
    if not llm:
        st.error(f"Cannot run. Error: {system_status}")
        st.stop()

    with st.status("🚀 Engine Running...", expanded=True) as status:
        st.write("🌍 Researcher gathering intelligence...")
        try:
            result = app.invoke({"topic": topic})
            st.write("🧠 Analyst compiling report...")
            st.write("🛡️ Auditor verifying integrity...")
            
            if result.get("fact_check_feedback"): 
                st.warning(f"Correction Applied: {result['fact_check_feedback']}")
            
            status.update(label="✅ Complete!", state="complete", expanded=False)
            st.markdown(result["final_brief"])
        except Exception as e:
            st.error(f"Workflow Failed: {e}")
