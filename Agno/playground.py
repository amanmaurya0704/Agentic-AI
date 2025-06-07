from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
from agno.playground import Playground,serve_playground_app
from agno.storage.sqlite import SqliteStorage

agent_storage:str = "tmp/agent.db"

web_agent = Agent(
    name = "Web Agent",
    role = "Search the web for the information",
    model = Groq(id = "llama-3.3-70b-versatile"),
    storage=SqliteStorage(table_name="web_agent", db_file=agent_storage),
    add_datetime_to_instructions=True,
    add_history_to_messages=True,
    num_history_responses=5,
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
    storage=SqliteStorage(table_name="finance_agent", db_file=agent_storage),
    add_datetime_to_instructions=True,
    add_history_to_messages=True,
    num_history_responses=5,
    show_tool_calls=True,
    markdown=True,
)
agent_team = Agent(
    name = "Agent Team",
    role = "User provided agents to give best answer to user query",
    team = [finance_agent,web_agent],
    model = Groq(id = "llama-3.3-70b-versatile"),
    instructions= ["Always include sources", "Use table to display data"],
    show_tool_calls=True,
    markdown=True,
    

)
app = Playground(agents=[web_agent, finance_agent,agent_team]).get_app()
if __name__ == "__main__":
    serve_playground_app("playground:app", reload=True)
