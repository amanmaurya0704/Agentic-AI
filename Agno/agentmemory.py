from agno.agent import Agent
from agno.models.groq import Groq
from agno.embedder.ollama import OllamaEmbedder
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.vectordb.lancedb import LanceDb, SearchType

agent = Agent(
    model = Groq(id="llama3-70b-8192"),
    description = "You are Thai cuiseine expert",
    instructions=[
        "Search you knowledge base for Thai recipes.",
        "If the question is better suited for web search, search the web to fill in Gaps.",
        "Prefer the information in your knowledge base over the web results",
    ],
    knowledge=PDFUrlKnowledgeBase(
        urls = ["https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
        vector_db= LanceDb(
            uri = "tmp/lancedb",
            table_name= "recipes",
            search_type=SearchType.hybrid,
            embedder=OllamaEmbedder(id="granite-embedding:30m") 
            ),
    ),
    tools = [DuckDuckGoTools()],
    show_tool_calls=True,
    markdown=True
)

if agent.knowledge is not None:
    agent.knowledge.load()

agent.print_response("How do I make chicken and galangal in coconut milk soup.",stream=True)

agent.print_response("What is the history of Thai curry?",stream=True)
