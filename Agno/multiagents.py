from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools

web_agent = Agent(
    name = "Web Agent",
    role = "Search the web for the information",
    model = Groq(id = "llama-3.3-70b-versatile"),
    tools = [DuckDuckGoTools()],
    instructions = "Always include sources",
    show_tool_calls=True,
    markdown=True,
)

finance_agent = Agent(
    name = "Finance Agent",
    role = "Get finance data",
    model = Groq(id = "llama-3.3-70b-versatile"),
    tools = [YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True)],
    instructions = "Use tables to display data",
    show_tool_calls=True,
    markdown=True,
)
agent_team = Agent(
    team = [finance_agent,web_agent],
    model = Groq(id = "llama-3.3-70b-versatile"),
    instructions= ["Always include sources", "Use table to display data"],
    show_tool_calls=True,
    markdown=True,
    

)
agent_team.print_response("Whats the market outlook and performance of AI semiconductor compnies like NVDA",stream = True)