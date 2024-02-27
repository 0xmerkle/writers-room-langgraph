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
    repurpose_content_to_tweets,
    AgentState,
)
from langchain_core.messages import (
    HumanMessage,
)

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph
from langgraph.prebuilt.tool_executor import ToolExecutor, ToolInvocation
import functools
from langchain.tools import tool
from typing import List, Dict
from openai import OpenAI
import os
import dotenv

dotenv.load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


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
    system_message="You should provide accurate data for the writer to use for writing a newsletter about the topic.",
)
research_node = functools.partial(agent_node, agent=research_agent, name="Researcher")


writer_agent = create_agent(
    llm,
    [],
    system_message="You should write for a newsletter article using the best copywriting frameworks available.",
)
writer_node = functools.partial(agent_node, agent=writer_agent, name="Writer")

critique_agent = create_agent(
    llm,
    [],
    system_message="You will critique the newsletter draft to make it more readable, coherent and grammatically correct.",
)
critique_node = functools.partial(agent_node, agent=critique_agent, name="Critique")


content_repurposer_agent = create_agent(
    llm,
    [repurpose_content_to_tweets],
    system_message="You will use the best content repurposing frameworks to find the best content for your newsletter.",
)
content_repurposer_node = functools.partial(
    agent_node, name="ContentRepurposer", agent=content_repurposer_agent
)


def router(state):
    # This is the router
    messages = state["messages"]
    print("STATE", state)
    print("MESSAGES", messages)
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
workflow.add_node("ContentRepurposer", content_repurposer_node)
# workflow.add_node("Chart Generator", chart_node)
workflow.add_node("call_tool", tool_node)

workflow.add_conditional_edges(
    "Researcher",
    router,
    {"continue": "Writer", "call_tool": "call_tool", "end": END},
)
# workflow.add_edge("Researcher", "Writer")
workflow.add_edge("Writer", "Critique")

workflow.add_edge("Critique", "ContentRepurposer")
# workflow.add_edge("ContentRepurposer", END)
workflow.add_conditional_edges(
    "ContentRepurposer",
    router,
    {"call_tool": "call_tool", "end": END},
)


# workflow.add_edge("Researcher", "Writer")
# workflow.add_edge("Writer", "Critique")
# workflow.add_conditional_edges("CritiqueNode", critique_decision, {"revise": "WriteNode", "approve": "TweetNode"})

workflow.add_conditional_edges(
    "call_tool",
    # Each agent node updates the 'sender' field
    # the tool calling node does not, meaning
    # this edge will route back to the original agent
    # who invoked the tool
    lambda x: x["sender"],
    {
        "Researcher": "Researcher",
        "ContentRepurposer": "ContentRepurposer",
    },
)
workflow.set_entry_point("Researcher")
graph = workflow.compile()

import json


def save_steps_to_file(steps, file_name="test_outputs.txt"):
    # Convert the steps list to a JSON string for readable storage
    steps_json = json.dumps(steps, indent=4)

    # Open the file in append mode, creating it if it doesn't exist
    with open(file_name, "a") as file:
        file.write(steps_json)
        file.write("\n\n---\n\n")


def run_agent_pipeline(user_input):
    steps = []
    for s in graph.stream(
        {
            "messages": [
                HumanMessage(
                    content=f"Find relevant information for the topic below that we can use to write in a newsletter article. Then write the newsletter draft. Then critique it. Then repurpose the article draft into tweets. Once you've done all of that, finish. \n TOPIC: {user_input}",
                )
            ],
        },
        # Maximum number of steps to take in the graph
        {"recursion_limit": 150},
    ):
        print(s)
        print("---")
        steps.append(s)
    save_steps_to_file(steps=steps, file_name="test_outputs.txt")

    return steps
