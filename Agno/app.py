from agno.agent import Agent
from agno.models.groq import Groq
from dotenv import load_dotenv
from agno.tools.duckduckgo import DuckDuckGoTool
import os
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
agent = Agent(
    model = Groq(id = "llama3-70b-8192"),
    description = "You are a helpful assistant that can answer questions based on user query.",
    tools = [DuckDuckGoTool()]
    markdown = True
    )
agent.print_response("What is the capital of France?")