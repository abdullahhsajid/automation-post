from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from typing import TypedDict, List
import tweepy
import os
from fastapi import FastAPI
from dotenv import load_dotenv
load_dotenv()

consumer_key = os.getenv("CONSUMER_KEY")
consumer_secret = os.getenv("CONSUMER_SECRET")

access_token = os.getenv("ACCESS_TOKEN")
access_secret = os.getenv("ACCESS_SECRET")

api_key = os.getenv('OPENAI_API_KEY')

app = FastAPI(title="Post Automation Bot", description="API to run Crew AI Bot", version="1.0.0")
# nest_asyncio.apply()


class State(TypedDict):
    messages: List
    next_steps: List[str]


def researcher(state: State) -> State:
    messages = state["messages"]
    llm = ChatOpenAI(model="gpt-4", api_key=api_key)
    response = llm.invoke(messages + [HumanMessage(content="your researcher agent so identify accurate topics")])
    return {
        "messages": messages + [response],
        "next_steps": ['writer']
    }


def writer(state: State) -> State:
    messages = state["messages"]
    llm = ChatOpenAI(model="gpt-4", api_key=api_key)
    response = llm.invoke(messages + [HumanMessage(content="your writer agent so write comprehensive base on topic")])
    return {
        "messages": messages + [response],
        "next_steps": [END]
    }


def PostTweet():
    workflow = StateGraph(State)
    workflow.add_node("researcher", researcher)
    workflow.add_node("writer", writer)
    workflow.add_edge("researcher", "writer")
    workflow.set_entry_point("researcher")
    workflow.set_finish_point("writer")
    app = workflow.compile()

    print("\nStarting workflow...")
    final_state = app.invoke({
        "messages": [HumanMessage(content="talk about mars")],
        "next_steps": []
    })
    print(f"\nFinal messages: {final_state['messages']}")
    response = ""
    for msg in final_state["messages"]:
        if isinstance(msg, AIMessage):
            response += msg.content

    print(f'response => ${response}')
    client = tweepy.Client(
            consumer_key = consumer_key, consumer_secret=consumer_secret,
            access_token=access_token, access_token_secret=access_secret)

    text_data = response.strip()[:200]

    response = client.create_tweet(text=text_data)


@app.get("/")
async def root():
    return {"message": "Crew AI Bot API is running"}


@app.get("/post")
async def post():
    PostTweet()
    return {"message": "Post created"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)