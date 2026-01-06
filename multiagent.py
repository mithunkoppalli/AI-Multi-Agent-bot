import os
from dotenv import load_dotenv
load_dotenv()

from typing import TypedDict, List

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from tavily import TavilyClient


class AgentState(TypedDict):
    user_input: str
    plan: List[str]
    research: List[str]
    final_answer: str


GROQ_KEY = "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"
TAVILY_KEY = "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"

if not GROQ_KEY:
    raise ValueError("GROQ_API_KEY missing from .env")

if not TAVILY_KEY:
    raise ValueError("TAVILY_API_KEY missing from .env")


llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=GROQ_KEY,
    temperature=0
)

tavily = TavilyClient(api_key=TAVILY_KEY)


def planner_agent(state: AgentState) -> AgentState:
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(
            content="Break the query into 3-5 research tasks. Return ONLY a numbered list."
        ),
        HumanMessage(content=state["user_input"])
    ])

    response = llm.invoke(prompt.format_messages())
    plan = [
        line.split(".", 1)[-1].strip()
        for line in response.content.split("\n")
        if line.strip()
    ]

    return {**state, "plan": plan}


def searcher_agent(state: AgentState) -> AgentState:
    research = []

    for task in state["plan"]:

        # FIX → Skip empty tasks to avoid Tavily "Query is missing" error
        if not task or task.strip() == "":
            continue

        result = tavily.search(
            query=task,
            search_depth="advanced",
            max_results=1,
            include_domains=[
                "scholar.google.com",
                "semanticscholar.org",
                "arxiv.org",
                "ncbi.nlm.nih.gov",
                "pubmed.ncbi.nlm.nih.gov",
                "researchgate.net",
                "jstor.org"
            ]
        )

        for item in result.get("results", []):
            research.append(
                f"Source: {item['url']}\nTitle: {item['title']}\nContent: {item['content']}\n"
            )

    return {**state, "research": research}


def writer_agent(state: AgentState) -> AgentState:
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(
            content="Write a clear answer from the research. Do NOT mention agents or tools."
        ),
        HumanMessage(
            content=f"Question:\n{state['user_input']}\n\nResearch:\n" +
                    "\n".join(state["research"])
        )
    ])

    response = llm.invoke(prompt.format_messages())
    return {**state, "final_answer": response.content}

graph = StateGraph(AgentState)

graph.add_node("planner", planner_agent)
graph.add_node("searcher", searcher_agent)
graph.add_node("writer", writer_agent)

graph.set_entry_point("planner")
graph.add_edge("planner", "searcher")
graph.add_edge("searcher", "writer")
graph.add_edge("writer", END)


app = graph.compile()
