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
st.markdown("### Powered by Multi-Agent Orchestration & Gemini")

# --- 2. Safe Imports ---
try:
    import google.generativeai as genai
    from langchain_google_genai import (ChatGoogleGenerativeAI,
                                        HarmBlockThreshold, HarmCategory)
    LIBS_INSTALLED = True
except ImportError as e:
    st.error(f"⚠️ Missing Libraries: {e}. Please update requirements.txt and Reboot.")
    st.stop()

# --- 3. The Logic to Find a Working Model ---
def get_working_llm():
    if "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
    else:
        api_key = os.environ.get("GOOGLE_API_KEY", "")
    
    clean_key = api_key.strip().strip('"').strip("'")
    if not clean_key:
        st.error("🚨 No API Key found.")
        st.stop()

    try:
        genai.configure(api_key=clean_key)
        
        # Disable Safety Filters for deep analysis
        safety_config = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=clean_key,
            transport="rest",
            temperature=0,
            safety_settings=safety_config
        )
        llm.invoke("Hello") 
        return llm, "🟢 Connected: Gemini 1.5 Flash"

    except Exception as e:
        return None, f"🔴 Connection Error: {e}"

llm, system_status = get_working_llm()

# --- 4. Deep Search Tool ---
class DeepSearchTool:
    def run(self, query):
        # STRATEGY 1: Deep Web Search (DuckDuckGo)
        try:
            with st.spinner(f"🌐 Scouring the web for deep insights on '{query}'..."):
                # Increased results to 10 for "Intelligence Engine" depth
                results = DDGS().text(query, max_results=10)
                if results:
                    return f"**SOURCE: LIVE DEEP SEARCH**\n\n{str(results)}"
        except Exception as e:
            print(f"DDG Failed: {e}")
        
        # STRATEGY 2: Internal Knowledge Deep Dive
        try:
            with st.spinner("Live search limited. Accessing Deep Internal Archives..."):
                prompt = (
                    f"You are a comprehensive intelligence database. "
                    f"Provide an exhaustive, multi-faceted report on: '{query}'. "
                    f"Include history, current state, key players, and future outlook. "
                    f"Output raw facts only."
                )
                response = llm.invoke(prompt)
                return f"**SOURCE: GEMINI INTERNAL KNOWLEDGE**\n\n{response.content}"
        except Exception as e:
            return f"Error: Could not retrieve data. Details: {e}"

search_tool = DeepSearchTool()

# Sidebar Status
st.sidebar.caption(f"Brain: {system_status}")
st.sidebar.caption("Search: Deep Scan (10+ Sources)")

# --- 5. Define Agents ---
class AgentState(TypedDict):
    topic: str
    raw_research: str
    draft_brief: str
    fact_check_feedback: str
    final_brief: str
    messages: Annotated[List[str], operator.add]

def research_node(state: AgentState):
    """Agent 1: The Researcher (Deep Dive)"""
    topic = state["topic"]
    results = search_tool.run(topic)
    return {"raw_research": results}

def analyst_node(state: AgentState):
    """Agent 2: The Analyst (Comprehensive Report)"""
    research_data = state["raw_research"]
    feedback = state.get("fact_check_feedback")
    
    # UPGRADED PROMPT FOR 10 POINTS
    prompt = (
        "You are a Chief Intelligence Officer. "
        "Your goal is to produce a **Comprehensive Intelligence Report** based on the provided raw data. "
        "\n\n"
        "**Requirements:**\n"
        "1. Identify at least **10 distinct, high-value insights**.\n"
        "2. Categorize them logically (e.g., Strategic, Financial, Technical, Risks).\n"
        "3. Use professional markdown formatting.\n"
        "4. **Strictly adhere to the facts provided** in the source data. Do not invent details."
    )
    if feedback:
        prompt += f"\n\n🚨 CRITICAL: PREVIOUS DRAFT FAILED AUDIT. \n**Auditor Feedback:** {feedback}\n\nFix these specific issues immediately."
    
    messages = [SystemMessage(content=prompt), HumanMessage(content=f"RAW INTELLIGENCE DATA:\n{research_data}")]
    response = llm.invoke(messages)
    return {"draft_brief": response.content}

def fact_checker_node(state: AgentState):
    """Agent 3: The Auditor (Strict Verification)"""
    raw = state["raw_research"]
    draft = state["draft_brief"]
    
    prompt = (
        "You are a Quality Assurance Auditor. "
        "Your job is to compare the INTELLIGENCE REPORT against the SOURCE DATA."
        "\n1. Verify that the 10+ points generated are supported by the source text."
        "\n2. If the report contains hallucinated facts, reply 'FAIL: <explanation of what is wrong>'."
        "\n3. If the report is faithful to the source, reply 'PASS'."
    )
    
    messages = [SystemMessage(content=prompt), HumanMessage(content=f"SOURCE TEXT:\n{raw}\n\nDRAFT REPORT:\n{draft}")]
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
topic = st.text_input("Enter Topic for Deep Analysis:", placeholder="e.g. NVIDIA's AI Chip Strategy")

if st.button("Generate Intelligence Report"):
    if not llm:
        st.error("LLM not connected. Check API Key.")
        st.stop()

    with st.status("🚀 Engine Running...", expanded=True) as status:
        
        st.write("🌍 **Researcher** is conducting a deep web scan...")
        try:
            result = app.invoke({"topic": topic})
            
            st.write("🧠 **Chief Analyst** is compiling the 10-point report...")
            st.write("🛡️ **Auditor** is verifying data integrity...")
            
            if result.get("fact_check_feedback"): 
                st.warning(f"⚠️ Discrepancy Found! \nAudit Log: {result['fact_check_feedback']}")
            
            status.update(label="✅ Report Ready!", state="complete", expanded=False)
            
            st.subheader("📑 Comprehensive Intelligence Report")
            st.markdown(result["final_brief"])

            with st.expander("📂 View Raw Source Data"):
                st.info(result["raw_research"])
                
        except Exception as e:
            st.error(f"Workflow Failed: {e}")
