from typing import Literal

from langchain_classic.hub import pull
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel
from retrieval_graph.configuration import Configuration
from retrieval_graph.state import AgentState
from retrieval_graph.utils import format_docs, load_chat_model
from shared.retrieval import make_retriever


class Schema(BaseModel):
    route: str = Literal["retrieve", "direct"]
    direct_answer: str


async def check_query_type(state: AgentState, *, config: RunnableConfig):
    configuration = Configuration.from_runnable_config(config)
    structured_llm = load_chat_model(configuration.query_model).with_structured_output(
        Schema
    )
    routing_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a routing assistant. Your job is to determine if a question needs document retrieval or can be answered directly.\n\nRespond with either:\n'retrieve' - if the question requires retrieving documents\n'direct' - if the question can be answered directly AND your direct answer.\n\n"
                "You MUST respond with a VALID JSON object ONLY.\n"
                "STRICT FORMAT:\n"
                "{{\n"
                '  "route": "retrieve" | "direct",\n'
                '  "direct_answer": "text"\n'
                "}}\n"
                "Do NOT write markdown. Do NOT add explanations. Only JSON.",
            ),
            ("human", "{query}"),
        ]
    )

    formatted_prompt = routing_prompt.invoke({"query": state["query"]})
    response = structured_llm.invoke(formatted_prompt)

    route = response.route

    if route == "retrieve":
        return {"route": "retrieve_documents"}
    else:
        direct_answer = response.direct_answer
        return {"route": END, "messages": [HumanMessage(content=direct_answer)]}


async def route_query(
    state: AgentState, *, config: RunnableConfig
) -> Literal["retrieve_documents", END]:
    route = state["route"]
    if not route:
        raise ValueError("Route is not set")

    if route == "retrieve_documents":
        return "retrieve_documents"
    else:
        return END


async def retrieve_documents(state: AgentState, *, config: RunnableConfig):

    with make_retriever(config) as retriever:
        response = retriever.invoke(state["query"])

    return {"documents": response}


async def generate_response(state: AgentState, *, config: RunnableConfig):
    configuration = Configuration.from_runnable_config(config)
    context = format_docs(state["documents"])
    prompt_template = pull("rlm/rag-prompt")
    formatted_prompt = prompt_template.invoke(
        {"context": context, "question": state["query"]}
    )
    messages = formatted_prompt.messages + state["messages"]
    response = load_chat_model(configuration.query_model).invoke(messages)
    return {"messages": response}


builder = StateGraph(AgentState, config_schema=Configuration)
builder.add_node("check_query_type", check_query_type)
builder.add_node("retrieve_documents", retrieve_documents)
builder.add_node("generate_response", generate_response)
builder.add_edge(START, "check_query_type")
builder.add_conditional_edges("check_query_type", route_query)
builder.add_edge("retrieve_documents", "generate_response")
builder.add_edge("generate_response", END)

# Compile into a graph object that you can invoke and deploy.
graph = builder.compile()
graph.name = "RetrievalGraph"
