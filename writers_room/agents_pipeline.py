from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolExecutor
from langchain_openai import ChatOpenAI
from langchain.tools.render import format_tool_to_openai_function
from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage
from writers_room.langgraph_utils import (
    agent_node,
    create_agent,
    create_openai_functions_agent,
    tool_node,
    tavily_tool,
    python_repl,
    AgentState,
)
from langchain_core.messages import (
    HumanMessage,
)

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph
from langgraph.prebuilt.tool_executor import ToolExecutor, ToolInvocation
import functools


def _setup_tools():
    tools = [TavilySearchResults(max_results=1)]
    return tools


def _setup_model():
    tools = _setup_tools()
    tool_executor = ToolExecutor(tools)

    model = ChatOpenAI(temperature=0, streaming=True)
    functions = [format_tool_to_openai_function(t) for t in tools]
    model.bind_functions(functions)


llm = ChatOpenAI(model="gpt-4-1106-preview")

# Research agent and node
research_agent = create_agent(
    llm,
    [tavily_tool],
    system_message="You should provide accurate data for the chart generator to use.",
)
research_node = functools.partial(agent_node, agent=research_agent, name="Researcher")

# Chart Generator
chart_agent = create_agent(
    llm,
    [python_repl],
    system_message="Any charts you display will be visible by the user.",
)
chart_node = functools.partial(agent_node, agent=chart_agent, name="Chart Generator")

writer_agent = create_agent(
    llm,
    [],
    system_message="You should write for a newsletter using the best copywriting frameworks available.",
)
writer_node = functools.partial(agent_node, agent=writer_agent, name="Writer")

critique_agent = create_agent(
    llm,
    [],
    system_message="You will critique the newsletter draft to make it more readable, coherent and grammatically correct.",
)
critique_node = functools.partial(agent_node, agent=critique_agent, name="Critique")


def router(state):
    # This is the router
    messages = state["messages"]
    last_message = messages[-1]
    if "function_call" in last_message.additional_kwargs:
        # The previus agent is invoking a tool
        return "call_tool"
    if "FINAL ANSWER" in last_message.content:
        # Any agent decided the work is done
        return "end"
    return "continue"


workflow = StateGraph(AgentState)

workflow.add_node("Researcher", research_node)
workflow.add_node("Writer", writer_node)
workflow.add_node("Critique", critique_node)
# workflow.add_node("Chart Generator", chart_node)
workflow.add_node("call_tool", tool_node)

# workflow.add_conditional_edges(
#     "Researcher",
#     router,
#     {"continue": "Chart Generator", "call_tool": "call_tool", "end": END},
# )
workflow.add_edge("ResearchNode", "WriteNode")
workflow.add_edge("WriteNode", "CritiqueNode")
# workflow.add_conditional_edges("CritiqueNode", critique_decision, {"revise": "WriteNode", "approve": "TweetNode"})

# workflow.add_conditional_edges(
#     "Chart Generator",
#     router,
#     {"continue": "Researcher", "call_tool": "call_tool", "end": END},
# )

workflow.add_conditional_edges(
    "call_tool",
    # Each agent node updates the 'sender' field
    # the tool calling node does not, meaning
    # this edge will route back to the original agent
    # who invoked the tool
    lambda x: x["sender"],
    {
        "Researcher": "Researcher",
        # "Chart Generator": "Chart Generator",
    },
)
workflow.set_entry_point("Researcher")
graph = workflow.compile()


def run_agent_pipeline(uer_input):
    steps = []
    for s in graph.stream(
        {
            "messages": [
                HumanMessage(
                    content=f"Create a newsletter draft about the following topic. \n TOPIC: {uer_input}",
                )
            ],
        },
        # Maximum number of steps to take in the graph
        {"recursion_limit": 150},
    ):
        print(s)
        print("---")
        steps.append(s)

    return steps
