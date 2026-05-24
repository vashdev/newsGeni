import os
import json
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.func import task, entrypoint
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage , ToolMessage, AnyMessage
from tavily import TavilyClient
from langchain.tools import tool
from typing_extensions import TypedDict, Annotated
import operator
from typing import Literal
from langgraph.graph import StateGraph, START, END
from guardrails import run_input_guardrails, run_output_guardrails, GuardrailStatus

from IPython.display import Image, display
from pydantic import BaseModel, Field
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int


class GuardrailsOutput(BaseModel):
    decision: Literal["end", "planner"] = Field(
        description="Decision on whether the question is related to the news ."
    )

    # Initialize LLM at module level or pass it through tasks

model = ChatAnthropic(
    model="claude-haiku-4-5",  
    max_tokens=300,
    temperature=0,  
    )

@tool("web_search", description="A tool to perform a web search for current news and information. Input should be a search query string.")
def websearch(search):
    TAVILY_API_KEY=os.getenv("TAVILY_API_KEY")
    if not TAVILY_API_KEY:
        return "Error: TAVILY_API_KEY is missing from environment variables."

    try:
        tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
        response = tavily_client.search(
            query=search,
            include_domains=["npr.org","news.google.com","nytimes.com"],
            max_results=5
        )
        results = response.get("results", [])
        if not results:
            return f"No results found for search: '{search}'"

        for result in results:
            print(f"[{result.get('score', 0):.2f}] {result.get('title', 'No Title')}")
            print(f"  {result.get('url', 'No URL')}\n")

        return results
    except Exception as e:
        return f"An error occurred while accessing the search API: {str(e)}"

# Augment the LLM with tools
tools = [websearch]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)

def llm_call(state: MessagesState):
    """LLM decides whether to call a tool or not"""

    # 1. Run Input Guardrails (only on the first turn)
    if state.get("llm_calls", 0) == 0:
        user_query = state["messages"][-1].content
        input_check = run_input_guardrails(user_query)
        if input_check.status == GuardrailStatus.BLOCK:
            return {
                "messages": [AIMessage(content=input_check.reason)],
                "llm_calls": 1 # Exit turn
            }

    # 2. Invoke the LLM
    response = model_with_tools.invoke(
        [
            SystemMessage(
                content="You are a helpful assistant answering user questions  and using tools when necessary"
            )
        ]
        + state["messages"]
    )

    # 3. Run Output Guardrails (only on final text responses, not tool calls)
    if not response.tool_calls and state.get("llm_calls", 0) > 0:
        context_chunks = []
        retrieval_scores = []
        # Gather context from previous tool results in state
        for m in state["messages"]:
            if isinstance(m, ToolMessage):
                try:
                    data = json.loads(m.content)
                    for item in data:
                        context_chunks.append(f"{item.get('title', '')}: {item.get('url', '')}")
                        retrieval_scores.append(item.get("score", 0.0))
                except (json.JSONDecodeError, TypeError):
                    continue
        
        if context_chunks:
            output_check = run_output_guardrails(response.content, context_chunks, retrieval_scores)
            if output_check.requires_human_review or output_check.status != GuardrailStatus.PASS:
                response.content = output_check.answer

    return {
        "messages": [response],
        "llm_calls": state.get('llm_calls', 0) + 1
    }

def tool_node(state: dict):
    """Performs the tool call"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        # Convert tool output to a string to avoid ValueError in message formatting
        result.append(ToolMessage(content=json.dumps(observation), tool_call_id=tool_call["id"]))
    return {"messages": result}



def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    if state.get("llm_calls", 0) >= 5:
        return END

    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        return "tool_node"

    # Otherwise, we stop (reply to the user)
    return END


def get_agent():
    agent_builder = StateGraph(MessagesState)
    agent_builder.add_node("llm_call", llm_call)
    agent_builder.add_node("tool_node", tool_node)
    agent_builder.add_edge(START, "llm_call")
    agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END]
    )
    agent_builder.add_edge("tool_node", "llm_call")
    return agent_builder.compile()

def main():
    # Configuration
    agent = get_agent()
    display(Image(agent.get_graph(xray=True).draw_mermaid_png()))
    
    # Invoke
    messages = [HumanMessage(content="latest news today")]
    messages = agent.invoke({"messages": messages})
    #for m in messages["messages"]:
    #    m.pretty_print()

if __name__ == "__main__":
    main()
