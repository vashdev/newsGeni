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

from IPython.display import Image, display
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int

    # Initialize LLM at module level or pass it through tasks

model = ChatAnthropic(
    model="claude-haiku-4-5",  
    max_tokens=300,
    temperature=0,  
    )

@tool("web_search", description="A tool to perform a web search for current news and information. Input should be a search query string.")
def websearch(search):
    TAVILY_API_KEY=os.getenv("TAVILY_API_KEY")

    tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
    response = tavily_client.search(
    query=search,
    include_domains=["npr.org","news.google.com","nytimes.com"],
    max_results=5) or {"results": []}

    results = response.get("results", [])
    for result in results:
        print(f"[{result['score']:.2f}] {result['title']}")
        print(f"  {result['url']}\n")

    return results

# Augment the LLM with tools
tools = [websearch]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)

def llm_call(state: dict):
    """LLM decides whether to call a tool or not"""

    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content="You are a helpful assistant answering user questions  and using tools when necessary"
                    )

                ]
                + state["messages"]
            )
        ],
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
    for m in messages["messages"]:
        m.pretty_print()

if __name__ == "__main__":
    main()
