from typing import Annotated, List, Tuple, Union

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langsmith import trace
from typing import Any, Callable, List, Optional, TypedDict, Union

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from langgraph.graph import END, StateGraph
import functools
import operator

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai.chat_models import ChatOpenAI
import functools
from tempfile import TemporaryDirectory
from pathlib import Path
from openai import OpenAI
import os
import dotenv
from uuid import uuid4

dotenv.load_dotenv()


def create_unique_dir(base_dir="pipeline_outputs"):
    """Create and return a unique directory path for storing pipeline outputs."""
    base_path = Path(base_dir)
    unique_dir_path = base_path / str(uuid4())
    unique_dir_path.mkdir(parents=True, exist_ok=False)
    return unique_dir_path


openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
_TEMP_DIRECTORY = TemporaryDirectory()
WORKING_DIRECTORY = create_unique_dir()
tavily_tool = TavilySearchResults(max_results=5)


@tool
def scrape_webpages(urls: List[str]) -> str:
    """Use requests and bs4 to scrape the provided web pages for detailed information."""
    loader = WebBaseLoader(urls)
    docs = loader.load()
    return "\n\n".join(
        [
            f'<Document name="{doc.metadata.get("title", "")}">\n{doc.page_content}\n</Document>'
            for doc in docs
        ]
    )


@tool
def repurpose_content_to_tweets(
    article_content: str,
    # output_dir: Annotated[Path, "output directory of the draft"],
) -> List[str]:
    """Repurpose the content of the article into an array tweets returned in JSON format."""
    print(f"REPUPPOSE CONTENT TO TWEETS: ===============")

    user_prompt = f"""
        Please repurpose the content of the article into an array tweets returned in JSON format.
        # ARTICLE:
        {article_content}
        # END OF ARTICLE
        ---
        The format should be like this:
        {{
        "tweets": [<tweet1>, <tweet2>, <tweet3>],
        }}
    """
    formatted_message = {"role": "user", "content": user_prompt}
    print(f"USER PROMPT: {formatted_message}")
    response = openai_client.chat.completions.create(
        model="gpt-4-turbo-preview",
        response_format={"type": "json_object"},
        messages=[formatted_message],
    )
    tweets_path = WORKING_DIRECTORY / "tweets.json"
    # Logic to generate tweets...
    tweets = response.choices[0].message.content
    with tweets_path.open("w") as file:
        json.dump({"tweets": tweets}, file)
    return {
        "status": "Successfully repurposed tweets. Done.",
        "repurposed_tweets": response.choices[0].message.model_dump_json(),
        "tweets_path": tweets_path,
    }


@tool
def write_draft(
    topic: Annotated[str, "topic of the article"],
    research_results: Annotated[str, "research results of the topic"],
    # output_dir: Annotated[Path, "output directory of the draft"],
) -> str:
    """Use this to write a draft of the article."""
    print(f"WRITE DRAFT: ===============")
    user_prompt = f"""
    Please write a draft of the article based on the topic and research results.
    # TOPIC:
    {topic}
    # END OF TOPIC
    # RESEARCH RESULTS:
    {research_results}
    # END OF RESEARCH RESULTS
    ---
    Only write the article draft and nothing else. Begin!
    """
    formatted_message = {"role": "user", "content": user_prompt}
    print(f"USER PROMPT: {formatted_message}")
    response = openai_client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[formatted_message],
    )
    draft_path = WORKING_DIRECTORY / "newsletter_draft.json"
    # Logic to generate the draft...
    draft = response.choices[0].message.content
    with draft_path.open("w") as file:
        json.dump({"draft": draft}, file)
    return {
        "newsletter_draft": response.choices[0].message.content,
        "status": "Newsletter draft completed. Done.",
        "draft_path": draft_path,
    }


@tool
def critique_content(content: str) -> str:
    """Use this to critique the content of the article."""
    print(f"CRITIQUE CONTENT: ===============")
    user_prompt = f"""
    Please critique the content of the article on a scale of 1-5 for each of the following aspects:
    1. Coherence
    2. Grammar
    3. Style
    ---
    Please find the content of the article below:
    # ARTICLE:
    {content}
    # END OF ARTICLE
    ---
    Return your critique in the JSON format:
    {{
    "critiques": {{
        "coherence": <int>,
        "grammar": <int>,
        "style": <string>,
        }}
    }}
   
]
    """
    formatted_message = {"role": "user", "content": user_prompt}
    print(f"USER PROMPT: {formatted_message}")
    response = openai_client.chat.completions.create(
        model="gpt-4-turbo-preview",
        response_format={"type": "json_object"},
        messages=[formatted_message],
    )
    json_response = response.choices[0].message.model_dump_json()
    print(f"JSON RESPONSE: {json_response}")
    return json_response


# ===========
def create_agent(
    llm: ChatOpenAI,
    tools: list,
    system_prompt: str,
) -> str:
    """Create a function-calling agent and add it to the graph."""
    system_prompt += "\nWork autonomously according to your specialty, using the tools available to you."
    " Do not ask for clarification."
    " Your other team members (and other teams) will collaborate with you with their own specialties."
    " You are chosen for a reason! You are one of the following team members: {team_members}."
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_functions_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor


def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}


def create_team_supervisor(llm: ChatOpenAI, system_prompt, members) -> str:
    """An LLM-based router."""
    options = ["FINISH"] + members
    function_def = {
        "name": "route",
        "description": "Select the next role.",
        "parameters": {
            "title": "routeSchema",
            "type": "object",
            "properties": {
                "next": {
                    "title": "Next",
                    "anyOf": [
                        {"enum": options},
                    ],
                },
            },
            "required": ["next"],
        },
    }
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                "Given the conversation above, who should act next?"
                " Or should we FINISH? Select one of: {options}",
            ),
        ]
    ).partial(options=str(options), team_members=", ".join(members))
    return (
        prompt
        | llm.bind_functions(functions=[function_def], function_call="route")
        | JsonOutputFunctionsParser()
    )


# ======= Research Team =======
class ResearchTeamState(TypedDict):
    # A message is added after each team member finishes
    messages: Annotated[List[BaseMessage], operator.add]
    # The team members are tracked so they are aware of
    # the others' skill-sets
    team_members: List[str]
    # Used to route work. The supervisor calls a function
    # that will update this every time it makes a decision
    next: str


llm = ChatOpenAI(model="gpt-4-turbo-preview")

search_agent = create_agent(
    llm,
    [tavily_tool],
    "You are a research assistant who can search for up-to-date info using the tavily search engine.",
)
search_node = functools.partial(agent_node, agent=search_agent, name="Search")

research_agent = create_agent(
    llm,
    [scrape_webpages],
    "You are a research assistant who can scrape specified urls for more detailed information using the scrape_webpages function.",
)
research_node = functools.partial(agent_node, agent=research_agent, name="Web Scraper")

supervisor_agent = create_team_supervisor(
    llm,
    "You are a supervisor tasked with managing a conversation between the"
    " following workers:  Search, Web Scraper. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH.",
    ["Search", "Web Scraper"],
)

research_graph = StateGraph(ResearchTeamState)
research_graph.add_node("Search", search_node)
research_graph.add_node("Web Scraper", research_node)
research_graph.add_node("supervisor", supervisor_agent)

# Define the control flow
research_graph.add_edge("Search", "supervisor")
research_graph.add_edge("Web Scraper", "supervisor")
research_graph.add_conditional_edges(
    "supervisor",
    lambda x: x["next"],
    {"Search": "Search", "Web Scraper": "Web Scraper", "FINISH": END},
)


research_graph.set_entry_point("supervisor")
chain = research_graph.compile()


def enter_chain(message: str):
    results = {
        "messages": [HumanMessage(content=message)],
    }
    return results


research_chain = enter_chain | chain
# ======= End of Research Team =======


class WritingTeamState(TypedDict):
    # A message is added after each team member finishes
    messages: Annotated[List[BaseMessage], operator.add]
    # The team members are tracked so they are aware of
    # the others' skill-sets
    team_members: List[str]
    # Used to route work. The supervisor calls a function
    next: str


def prelude(state):
    written_files = []
    if not WORKING_DIRECTORY.exists():
        WORKING_DIRECTORY.mkdir()
    try:
        written_files = [
            f.relative_to(WORKING_DIRECTORY) for f in WORKING_DIRECTORY.rglob("*")
        ]
    except:
        pass
    if not written_files:
        return {**state, "current_files": "No files written."}
    return {
        **state,
        "current_files": "\nBelow are files your team has written to the directory:\n"
        + "\n".join([f" - {f}" for f in written_files]),
    }


llm = ChatOpenAI(model="gpt-4-1106-preview")

draft_writer_agent = create_agent(
    llm,
    [write_draft],
    "You are an expert writing a research document.\n"
    # The {current_files} value is populated automatically by the graph state
    "Below are files currently in your directory:\n{current_files}",
)
# Injects current directory working state before each call
context_aware_draft_writer_agent = prelude | draft_writer_agent
draft_writing_node = functools.partial(
    agent_node, agent=context_aware_draft_writer_agent, name="Draft Writer"
)


content_repurposing_agent = create_agent(
    llm,
    [repurpose_content_to_tweets],
    "You are an expert in social media tasked with re-purposing content",
)
context_aware_repurposing_agent = prelude | content_repurposing_agent
repurposing_node = functools.partial(
    agent_node, agent=context_aware_repurposing_agent, name="Repurposing Agent"
)
draft_writing_supervisor = create_team_supervisor(
    llm,
    "You are a supervisor tasked with managing a conversation between the"
    " following workers:  {team_members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH.",
    ["Draft Writer", "Repurposing Agent"],
)

authoring_graph = StateGraph(WritingTeamState)
authoring_graph.add_node("Draft Writer", draft_writing_node)
authoring_graph.add_node("Repurposing Agent", repurposing_node)
authoring_graph.add_node("supervisor", draft_writing_supervisor)

# Add the edges that always occur
authoring_graph.add_edge("Draft Writer", "supervisor")
authoring_graph.add_edge("Repurposing Agent", "supervisor")

# Add the edges where routing applies
authoring_graph.add_conditional_edges(
    "supervisor",
    lambda x: x["next"],
    {
        "Draft Writer": "Draft Writer",
        "Repurposing Agent": "Repurposing Agent",
        "FINISH": END,
    },
)
authoring_graph.set_entry_point("supervisor")
chain = research_graph.compile()


# The following functions interoperate between the top level graph state
# and the state of the research sub-graph
# this makes it so that the states of each graph don't get intermixed
def enter_chain(message: str, members: List[str]):
    results = {
        "messages": [HumanMessage(content=message)],
        "team_members": ", ".join(members),
    }
    return results


# We re-use the enter/exit functions to wrap the graph
authoring_chain = (
    functools.partial(enter_chain, members=authoring_graph.nodes)
    | authoring_graph.compile()
)

#  ===== Add Layers =====

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai.chat_models import ChatOpenAI


llm = ChatOpenAI(model="gpt-4-1106-preview")

supervisor_node = create_team_supervisor(
    llm,
    "You are a supervisor tasked with managing a conversation between the"
    " following teams: {team_members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH.",
    ["Research team", "Paper writing team"],
)


class State(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    next: str


def get_last_message(state: State) -> str:
    return state["messages"][-1].content


def join_graph(response: dict):
    return {"messages": [response["messages"][-1]]}


super_graph = StateGraph(State)
super_graph.add_node("Research team", get_last_message | research_chain | join_graph)
super_graph.add_node(
    "Paper writing team", get_last_message | authoring_chain | join_graph
)
super_graph.add_node("supervisor", supervisor_node)

super_graph.add_edge("Research team", "supervisor")
super_graph.add_edge("Paper writing team", "supervisor")
super_graph.add_conditional_edges(
    "supervisor",
    lambda x: x["next"],
    {
        "Paper writing team": "Paper writing team",
        "Research team": "Research team",
        "FINISH": END,
    },
)
super_graph.set_entry_point("supervisor")
super_graph = super_graph.compile()

import json


def save_steps_to_file(steps, file_name="test_outputs.txt"):
    steps_json = json.dumps(steps, indent=4)

    with open(file_name, "a") as file:
        file.write(steps_json)
        file.write("\n\n---\n\n")


def run_agent_pipeline(user_input):
    steps = []
    # unique_dir = create_unique_dir()
    for s in super_graph.stream(
        {
            "messages": [
                HumanMessage(
                    content=f"""Research the following topic. 
                    Once you have the reearch information, you should pass it off to the writing team. 
                    Here is the topic you should research:
                    ---
                    TOPIC: {user_input}
                    ---
                    
                    """,
                )
            ],
        },
        # Maximum number of steps to take in the graph
        {"recursion_limit": 150},
    ):
        print(s)
        print("---")
        steps.append(s)
    newsletter_draft_path = WORKING_DIRECTORY / "newsletter_draft.json"
    # Convert Path to str
    tweets_path = WORKING_DIRECTORY / "tweets.json"

    newsletter_draft, tweets = {}, {}

    if newsletter_draft_path.exists():
        with open(newsletter_draft_path, "r") as nd_file:
            newsletter_draft = json.load(nd_file)

    if tweets_path.exists():
        with open(tweets_path, "r") as t_file:
            tweets = json.load(t_file)

    return {"newsletter_draft": newsletter_draft, "tweets": tweets}
