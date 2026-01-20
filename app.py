import streamlit as st
import os
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
import operator
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.tools import DuckDuckGoSearchRun

# --- 1. Page Config ---
st.set_page_config(page_title="Meeting Prep Squad", page_icon="🤖")
st.title("🤖 Multi-Agent Meeting Prep Squad")
st.markdown("### Powered by LangGraph & Gemini Flash")

# --- 2. API Key Setup ---
# Checks for key in Streamlit Secrets (for Cloud) or Environment (for Local)
if "GOOGLE_API_KEY" not in st.secrets and "GOOGLE_API_KEY" not in os.environ:
    st.error("Missing Google API Key. Please set it in Streamlit Secrets.")
    st.stop()

# Set key for the library
api_key = st.secrets.get("GOOGLE_API_KEY", os.environ.get("GOOGLE_API_KEY"))
os.environ["GOOGLE_API_KEY"] = api_key

# --- 3. Define Logic (Same as before, slightly optimized) ---
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
search_tool = DuckDuckGoSearchRun()

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
    return {"fact_check_feedback": response.content if "FAIL" in response.content else None, 
            "final_brief": draft if "FAIL" not in response.content else None}

def router(state: AgentState):
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

# --- 4. The UI ---
topic = st.text_input("Enter a meeting topic:", placeholder="e.g. Microsoft's Q4 AI Strategy")

if st.button("Run Agents"):
    with st.status("🤖 Agents are working...", expanded=True) as status:
        st.write("🔎 Researcher is searching the web...")
        # Run the graph
        result = app.invoke({"topic": topic})
        
        st.write("📝 Analyst is drafting...")
        st.write("🕵️‍♂️ Fact-Checker is validating...")
        
        # Show retry if it happened
        if result.get("fact_check_feedback"): 
            st.warning(f"Fact Check Failed initially! Retrying... Feedback: {result['fact_check_feedback']}")
        
        status.update(label="✅ Workflow Complete!", state="complete", expanded=False)

    # Display Results
    st.subheader("Final Executive Briefing")
    st.success(result["final_brief"])

    with st.expander("See Raw Research Data"):
        st.text(result["raw_research"])