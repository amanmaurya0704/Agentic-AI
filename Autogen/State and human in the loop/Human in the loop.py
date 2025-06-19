from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.ui import Console
import asyncio
import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
openai_model_client = OpenAIChatCompletionClient(model='gpt-4o',api_key=api_key)


assistant = AssistantAgent(
    name = 'Assistant',
    model_client=openai_model_client,
    description="You are a great assitant.",
    system_message="You are a really helpful assistant who help on the task given."
)

user_agent = UserProxyAgent(
    name= "UserPoxy",
    description="A proxy agent that represent a user.",
    input_func=input
)

termination_condition = TextMentionTermination('APPROVE')

agent_team = RoundRobinGroupChat([assistant,user_agent],termination_condition=termination_condition)

stream = agent_team.run_stream(task = "Write a 4 line poem about India.")

async def main():
    await Console(stream)

if(__name__=="__main__"):
    asyncio.run(main())

