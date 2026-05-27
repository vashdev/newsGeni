
# NewsGeni Agent

NewsGeni is an intelligent agent built with LangGraph and Streamlit that can answer general questions and search for the latest news using the Tavily API. It incorporates guardrails to ensure safe and relevant interactions.

## Features

-   **Conversational AI**: Interact with the agent through a Streamlit chat interface.
-   **Web Search Capabilities**: Utilizes the Tavily API to fetch current news and information.
-   **Guardrails**:
    -   **Input Guardrails**: Detects prompt injection attempts and checks for domain relevance (news-related topics).
    -   **Output Guardrails**: Verifies faithfulness of LLM responses against retrieved context and flags low-confidence answers.
-   **LangGraph Workflow**: Manages complex multi-turn interactions, including tool use and reasoning.

## Setup

### Prerequisites

-   Python 3.13+
-   `uv` (recommended package installer and virtual environment manager)

### 1. Clone the repository

```bash
git clone https://github.com/your-username/newsgeni.git # Replace with your actual repo URL
cd newsgeni
```

### 2. Set up the virtual environment and install dependencies

NewsGeni uses `uv` for dependency management.

```bash
uv venv
source .venv/bin/activate # On Windows, use `.venv\Scripts\activate`
uv sync
```

### 3. Configure API Keys

Create a `.env` file in the root directory of the project and add your API keys:

```
TAVILY_API_KEY="your_tavily_api_key_here"
ANTHROPIC_API_KEY="your_anthropic_api_key_here"
```

## How to Run

### 1. Run the Streamlit Chat App (Recommended)

This will launch the interactive chat interface in your web browser.

```bash
uv run streamlit run app.py
```

### 2. Run the Agent Directly (for testing/development)

This executes a predefined query and prints the agent's response to the console.

```bash
uv run main.py
```
