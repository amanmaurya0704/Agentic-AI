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


writerAgent = AssistantAgent(
    name = 'Witer',
    model_client=openai_model_client,
    description="You are a great writer.",
    system_message="You are a really helpful writer who wites in less than 30 words."
)

reviewerAgent = AssistantAgent(
    name = 'Reviewer',
    model_client=openai_model_client,
    description="You are a great reviewer.",
    system_message="You are a really helpful reviewer review the content and suggest changes in less then 30 words."
)

editorAgent = AssistantAgent(
    name = 'Editor',
    model_client=openai_model_client,
    description="You are a great editor.",
    system_message="You are a really helpful editor who edits the content based on suggestion in less then 30 words."
)


team_agent = RoundRobinGroupChat(
    participants=[writerAgent,reviewerAgent,editorAgent],
    max_turns=3
)


async def main():
    task = "Write a article about India and its culture"

    while True:
        stream = team_agent.run_stream(task=task)
        await Console(stream)
        feedback = input("Please provide your feedback (type 'exit' to stop)")
        if (feedback.lower().strip()=='exit'):
            break
        task = feedback

    

if(__name__=="__main__"):
    asyncio.run(main())