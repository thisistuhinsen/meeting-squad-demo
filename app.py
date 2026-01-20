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
st.markdown("### Powered by LangGraph & Gemini Flash")

# --- 2. API Key Setup ---
# Checks for key in Streamlit Secrets (Cloud) or Environment (Local)
if "GOOGLE_API_KEY" not in st.secrets and "GOOGLE_API_KEY" not in os.environ:
    st.error("🚨 Missing Google API Key.")
    st.info("To fix: Go to your Streamlit App -> Settings -> Secrets and add `GOOGLE_API_KEY = 'your_key'`")
    st.stop()

# Set key for the library
api_key = st.secrets.get("GOOGLE_API_KEY", os.environ.get("GOOGLE_API_KEY"))
os.environ["GOOGLE_API_KEY"] = api_key

# --- 3. Resilient Tool Setup (The "PM Fix") ---
# This block prevents the app from crashing if the search library has issues.
try:
    from langchain_community.tools import DuckDuckGoSearchRun
    search_tool = DuckDuckGoSearchRun()
    tool_status = "Live Web Search 🟢"
except Exception as e:
    # Fallback to a mock tool if the library fails to load
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

# --- 4. Define Logic ---
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

class AgentState(TypedDict):
    topic: str
    raw_research: str
    draft_brief: str
    fact_check_feedback: str
    final_brief: str
    messages: Annotated[List[str], operator.add]

def research_node(state: AgentState):
    topic = state["topic"]
    # Uses whichever tool loaded successfully (Live or Mock)
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
    response = llm.invoke(messages)
    return {"draft_brief": response.content}

def fact_checker_node(state: AgentState):
    raw = state["raw_research"]
    draft = state["draft_brief"]
    prompt = (
        "Compare the DRAFT BRIEF to the SOURCE TEXT. "
        "If faithful, reply 'PASS'. If hallucinated, reply 'FAIL: <reason>'."
    )
    messages = [SystemMessage(content=prompt), HumanMessage(content=f"SOURCE:\n{raw}\n\nDRAFT:\n{draft}")]
    response = llm.invoke(messages)
    
    # Simple logic: If the LLM says "FAIL", we loop back.
    if "FAIL" in response.content:
        return {"fact_check_feedback": response.content, "final_brief": None}
    else:
        return {"fact_check_feedback": None, "final_brief": draft}

def router(state: AgentState):
    # If feedback exists, go back to analyst. Otherwise end.
    return "analyst" if state.get("fact_check_feedback") else "end"

# Build Graph
workflow = StateGraph(AgentState)
workflow.add_node("researcher", research_node)
workflow.add_node("analyst", analyst_node)
workflow.add_node("fact_checker", fact_checker_node)

workflow.set_entry_point("researcher")
workflow.add_edge("researcher", "analyst")
workflow.add_edge("analyst", "fact_checker")
workflow.add_conditional_edges("fact_checker", router, {"analyst": "analyst", "end": END})

app = workflow.compile()

# --- 5. The UI ---
topic = st.text_input("Enter a meeting topic:", placeholder="e.g. OpenAI's recent safety news")

if st.button("Run Agents"):
    with st.status("🤖 Agents are working...", expanded=True) as status:
        st.write("🔎 Researcher is searching...")
        result = app.invoke({"topic": topic})
        
        st.write("📝 Analyst is drafting...")
        st.write("🕵️‍♂️ Fact-Checker is validating...")
        
        if result.get("fact_check_feedback"): 
            st.warning(f"Fact Check Failed initially! Retrying... Feedback: {result['fact_check_feedback']}")
        
        status.update(label="✅ Workflow Complete!", state="complete", expanded=False)

    st.subheader("Final Executive Briefing")
    st.success(result["final_brief"])

    with st.expander("See Raw Research Data"):
        st.text(result["raw_research"])
