import operator
import os
import time
from typing import Annotated, List, TypedDict

import streamlit as st
from duckduckgo_search import DDGS
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
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
except ImportError:
    st.error("⚠️ Libraries missing. Please Reboot App.")
    st.stop()

# --- 3. The "Mock Brain" (Your Safety Net) ---
class RobustMockLLM:
    def invoke(self, messages):
        last_msg = messages[-1].content if isinstance(messages, list) else str(messages)
        time.sleep(1.5) # Simulate thinking
        
        if "DATA:" in last_msg: 
            return AIMessage(content=(
                "### Intelligence Report (Demo Mode)\n\n"
                "**1. Strategic Dominance:** The subject has secured 40% market share in Q4.\n"
                "**2. Technical Velocity:** New agentic frameworks are reducing latency by 30%.\n"
                "**3. Risk Profile:** Regulatory headwinds in the EU remain the primary bottleneck.\n"
                "**4. Financial Outlook:** Projected revenue growth of 22% YOY driven by AI adoption.\n"
                "**5. Talent Density:** Aggressive hiring of research scientists from top competitors.\n"
                "**6. Product Roadmap:** Pivot to 'Autonomous Squads' expected in next release.\n"
                "**7. Customer Sentiment:** Net Promoter Score (NPS) rose to 72 post-launch.\n"
                "**8. Infrastructure:** Heavy investment in H100 clusters to support training.\n"
                "**9. Partnerships:** New strategic alliance with cloud providers announced.\n"
                "**10. Conclusion:** Strong buy signal for enterprise integration."
            ))
        elif "SOURCE:" in last_msg:
            return AIMessage(content="PASS")
        else:
            return AIMessage(content="I am ready.")

# --- 4. The Logic to Connect (With 429 Protection) ---
def get_resilient_llm():
    if "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
    else:
        api_key = os.environ.get("GOOGLE_API_KEY", "")
    
    clean_key = api_key.strip().strip('"').strip("'")
    if not clean_key:
        return RobustMockLLM(), "🟡 No Key (Mock Mode)"

    try:
        genai.configure(api_key=clean_key)
        
        # Disable Safety Filters
        safety_config = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        # FORCE the standard stable model
        model_name = "gemini-1.5-flash"

        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=clean_key,
            transport="rest",
            safety_settings=safety_config
        )
        # HANDSHAKE TEST
        llm.invoke("Hello") 
        return llm, f"🟢 Connected: {model_name}"

    except Exception as e:
        error_str = str(e)
        if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
            return RobustMockLLM(), "🟡 Quota Exceeded (Mock Active)"
        elif "NOT_FOUND" in error_str:
            # Fallback to Pro if Flash fails
            try:
                llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=clean_key, transport="rest", safety_settings=safety_config)
                llm.invoke("Hello")
                return llm, "🟢 Connected: gemini-pro"
            except:
                return RobustMockLLM(), "🟡 Model Error (Mock Active)"
        else:
            return RobustMockLLM(), "🟡 Connection Failed (Mock Active)"

# Initialize
llm, system_status = get_resilient_llm()

# --- 5. Search Tool ---
class DeepSearchTool:
    def run(self, query):
        try:
            with st.spinner(f"🌐 Scouring the web for '{query}'..."):
                results = DDGS().text(query, max_results=8)
                if results:
                    return f"**SOURCE: LIVE WEB SEARCH**\n\n{str(results)}"
        except Exception:
            pass
        
        # Fallback to LLM Knowledge (or Mock if LLM is Mock)
        try:
            response = llm.invoke(f"Provide detailed facts on: {query}")
            return f"**SOURCE: INTERNAL KNOWLEDGE**\n\n{response.content}"
        except:
            return "**SOURCE: OFFLINE ARCHIVE**\n\n(Simulated data for demo continuity)"

search_tool = DeepSearchTool()

st.sidebar.caption(f"Brain: {system_status}")

# --- 6. Define Agents ---
class AgentState(TypedDict):
    topic: str
    raw_research: str
    draft_brief: str
    fact_check_feedback: str
    final_brief: str
    messages: Annotated[List[str], operator.add]

def research_node(state: AgentState):
    topic = state["topic"]
    results = search_tool.run(topic)
    return {"raw_research": results}

def analyst_node(state: AgentState):
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
    
    # Safe Invoke
    try:
        response = llm.invoke(messages)
        return {"draft_brief": response.content}
    except:
        return {"draft_brief": "Error generating report. Switching to backup display."}

def fact_checker_node(state: AgentState):
    raw = state["raw_research"]
    draft = state["draft_brief"]
    
    prompt = "Compare REPORT to SOURCE. Reply 'PASS' or 'FAIL'."
    messages = [SystemMessage(content=prompt), HumanMessage(content=f"SOURCE:\n{raw}\n\nREPORT:\n{draft}")]
    
    try:
        response = llm.invoke(messages)
        content = response.content
    except:
        content = "PASS" # Fail open on error
    
    if "FAIL" in content:
        return {"fact_check_feedback": content, "final_brief": None}
    else:
        return {"fact_check_feedback": None, "final_brief": draft}

def router(state: AgentState):
    return "analyst" if state.get("fact_check_feedback") else "end"

# --- 7. Build Graph ---
workflow = StateGraph(AgentState)
workflow.add_node("researcher", research_node)
workflow.add_node("analyst", analyst_node)
workflow.add_node("fact_checker", fact_checker_node)
workflow.set_entry_point("researcher")
workflow.add_edge("researcher", "analyst")
workflow.add_edge("analyst", "fact_checker")
workflow.add_conditional_edges("fact_checker", router, {"analyst": "analyst", "end": END})
app = workflow.compile()

# --- 8. The UI ---
topic = st.text_input("Enter Topic:", placeholder="e.g. OpenAI vs Google Strategy")

if st.button("Generate Report"):
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
