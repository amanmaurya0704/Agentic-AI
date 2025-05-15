from typing import Annotated
from typing_extensions import TypedDict
from langchain_groq import ChatGroq
from langgraph.graph import END, START, MessageGraph
from langgraph.graph.state import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
import os
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper  
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun
from dotenv import load_dotenv

load_dotenv()

os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")

os.environ["LANGSMITH_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    
model = ChatGroq(model = "llama3-70b-8192", temperature=0)

def make_default_graph():
    """Make a simple LLM agent"""
    graph_workflow = StateGraph(State)
    def call_model(state):
        return {"messages": [model.invoke(state["messages"])]}

    graph_workflow.add_node("agent", call_model)
    graph_workflow.add_edge("agent", END)
    graph_workflow.add_edge(START, "agent")

    agent = graph_workflow.compile()
    return agent

def make_alternative_graph():
    """Make a tool-calling agent"""

    @tool
    def add(a: float, b: float):
        """Adds two numbers."""
        return a + b
    @tool
    def wiki_search(query: str):
        """A wrapper around Wikipedia. Useful for when you need to answer general questions about people, places, companies, facts, historical events, or other subjects. Input should be a search query."""
        api_wrapper_wiki = WikipediaAPIWrapper(top_k_results = 1, doc_content_chars_max = 250)
        wiki = WikipediaQueryRun(api_wrapper = api_wrapper_wiki)
        return wiki.run(query)
    @tool
    def arxiv_search(query: str):
        """Searches Arxiv for a Research papers."""
        api_wrapper_arxiv = ArxivAPIWrapper(top_k_results = 1, doc_content_chars_max = 250)
        arxiv = ArxivQueryRun(api_wrapper = api_wrapper_arxiv)
        return arxiv.run(query)
    tool_node = ToolNode([add,wiki_search,arxiv_search])
    model_with_tools = model.bind_tools([add,wiki_search,arxiv_search])
    #model_with_tools = model.bind_tools([add])
    def call_model(state):
        return {"messages": [model_with_tools.invoke(state["messages"])]}

    def should_continue(state: State):
        if state["messages"][-1].tool_calls:
            return "tools"
        else:
            return END

    graph_workflow = StateGraph(State)

    graph_workflow.add_node("agent", call_model)
    graph_workflow.add_node("tools", tool_node)
    graph_workflow.add_edge("tools", "agent")
    graph_workflow.add_edge(START, "agent")
    graph_workflow.add_conditional_edges("agent", should_continue)

    agent = graph_workflow.compile()
    return agent

agent=make_alternative_graph()