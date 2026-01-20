# 🤖 Multi-Agent Meeting Prep Squad
### _Powered by LangGraph, Gemini Flash, and Streamlit_

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url-here.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-orange)

## 📋 Overview
This project is a **self-correcting multi-agent system** designed to generate executive meeting briefings. Unlike standard "chatbots," this system implements an **agentic workflow** with a dedicated feedback loop to detect and correct hallucinations before they reach the user.

It demonstrates key capabilities for modern AI Product Management: **Orchestration**, **State Management**, and **Automated Evaluation**.

## 🏗️ System Architecture



The system utilizes a directed cyclic graph (DCG) managed by **LangGraph**:

1.  **🕵️ Researcher Node:** Uses `DuckDuckGo` to retrieve real-time data from the web.
2.  **📝 Analyst Node:** Synthesizes raw data into a concise 3-bullet executive summary.
3.  **⚖️ Fact-Checker Node (The "Judge"):** Compare the *Draft Brief* against the *Raw Research*.
    * **PASS:** The workflow terminates, and the user sees the result.
    * **FAIL:** The workflow **loops back** to the Analyst with specific feedback on what to fix.

## ✨ Key Features
* **Zero-Cost Architecture:** Built entirely on free tiers (Gemini Flash + Streamlit Community Cloud).
* **Hallucination Guardrails:** Implements an "LLM-as-a-Judge" pattern to enforce factual consistency.
* **State Persistence:** Uses a shared state schema to pass context between agents securely.
* **Live Web Access:** Not limited to training data; fetches current events via search tools.

## 🛠️ Tech Stack
* **Orchestration:** [LangGraph](https://langchain-ai.github.io/langgraph/) (Stateful multi-agent loops)
* **LLM:** [Google Gemini 1.5 Flash](https://deepmind.google/technologies/gemini/flash/) (Fast inference, large context)
* **Frontend:** [Streamlit](https://streamlit.io/) (Rapid UI deployment)
* **Tools:** `duckduckgo-search`, `langchain-google-genai`

## 🚀 How to Run Locally

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/yourusername/meeting-squad-demo.git](https://github.com/yourusername/meeting-squad-demo.git)
    cd meeting-squad-demo
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set your API Key**
    * Get a free key from [Google AI Studio](https://aistudio.google.com/).
    * Set it in your environment:
        * Mac/Linux: `export GOOGLE_API_KEY="your_key"`
        * Windows: `$env:GOOGLE_API_KEY="your_key"`

4.  **Run the App**
    ```bash
    streamlit run app.py
    ```

## 🧠 Product Perspective (Why I built this)
In enterprise contexts (like M365 Copilot), **trust is the primary metric**. A single hallucination can erode user confidence.

I chose a **Multi-Agent** approach over a single prompt because it allows for:
1.  **Separation of Concerns:** The "Researcher" focuses on recall, while the "Analyst" focuses on precision.
2.  **Measurable Quality:** The "Fact-Checker" provides a binary pass/fail metric we can log and analyze.
3.  **Extensibility:** We can easily swap the "DuckDuckGo" tool for an internal "SharePoint Search" tool without breaking the rest of the chain.

---
*Created by [Your Name] as a demonstration of Agentic AI Engineering & Product Management.*
