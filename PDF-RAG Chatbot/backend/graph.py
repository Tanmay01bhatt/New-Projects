from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, List
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.messages import BaseMessage,HumanMessage,AIMessage
from typing import TypedDict, Annotated, List
from langgraph.graph.message import add_messages



load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)
llm2=ChatGroq(model="llama-3.1-8b-instant")



def get_chains(retriever):


    # multi-query generation

    prompt = """You are a helpful assistant that generates multiple search queries based on a single input query. \n

    Generate multiple search queries related to: {question} \n

    Output (4 queries):"""

    multi_prompt = ChatPromptTemplate.from_template(prompt)

    generate_queries = (
        multi_prompt
        | llm
        | StrOutputParser()
        | RunnableLambda(lambda x: x.split("\n"))
    )

    # Re-Ranking

    def reciprocal_rank_fusion(results, k=60):

        fused_scores = {}
        doc_map = {}

        for docs in results:
            for rank, doc in enumerate(docs):

                doc_id = doc.page_content

                if doc_id not in fused_scores:
                    fused_scores[doc_id] = 0
                    doc_map[doc_id] = doc

                fused_scores[doc_id] += 1 / (rank + k) # reciprocal rank fusion formula

        reranked = sorted(
            fused_scores.keys(),
            key=lambda x: fused_scores[x],
            reverse=True
        )

        return [doc_map[x] for x in reranked]


    # answer generation chain

    answer_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the question based only on the context: {context}"),
    MessagesPlaceholder(variable_name="messages"),
    ("human", "{question}"),
    ])
    answer_chain = (
        answer_prompt 
        | llm 
        | StrOutputParser()
    )

    # hallucination check chain

    hallucination_prompt = ChatPromptTemplate.from_template("""
    You are a hallucination detection system.

    Context:
    {context}

    Answer:
    {answer}

    Check if the answer is fully supported by the context.

    Return:
    - "GROUNDED" if supported
    - "HALLUCINATED" if not supported

    """)

    hallucination_chain = (
        hallucination_prompt
        | llm
        | StrOutputParser()
    )

    # relevance chain

    relevance_prompt = ChatPromptTemplate.from_template("""
        You are a strict classifier.

        Determine whether the user's question can be answered
        using the provided document context.

        Question:
        {question}

        Context:
        {context}

        Answer ONLY:
        RELEVANT
        or
        NOT_RELEVANT
        """)

    relevance_chain = (
        relevance_prompt
        | llm2
        | StrOutputParser()
    )
    def relevance_retrieval_chain(question):

        docs = retriever.invoke(question)

        context = "\n\n".join(
            doc.page_content for doc in docs[:3]
        )

        result = relevance_chain.invoke({
            "question": question,
            "context": context
        })

        return result

    return generate_queries,answer_chain, hallucination_chain, relevance_retrieval_chain,reciprocal_rank_fusion

class AgentState(TypedDict):
    question: str
    queries: List[str]
    context: List[Document]
    answer: str
    hallucination_status: str
    messages: Annotated[List[BaseMessage], add_messages]
    results: List[List[Document]] 


def build_graph(retriever):
    generate_queries,answer_chain, hallucination_chain, relevance_retrieval_chain, reciprocal_rank_fusion = get_chains(retriever)

    def multi_query_node(state: AgentState):

        queries = generate_queries.invoke({
            "question": state["question"]
        })

        return {
            "queries": queries
        }

    def retrieve_node(state: AgentState):
        results = retriever.map().invoke(state["queries"])
        return {"results": results}

    def reciprocal_rank_fusion_node(state: AgentState):

        fused_docs = reciprocal_rank_fusion(state["results"])
        return {"context": fused_docs}

    def answer_node(state: AgentState):

        fused_docs = state["context"] 
        context_text = "\n\n".join(d.page_content for d in fused_docs)


        answer = answer_chain.invoke({
            "context": context_text,
            "question": state["question"],
            "messages": state["messages"]
        })
        return {"answer": answer,"messages": [HumanMessage(content=state["question"]),AIMessage(content=answer)]}

    def hallucination_node(state: AgentState):

        context_text = "\n\n".join(
            doc.page_content for doc in state["context"]
        )

        status = hallucination_chain.invoke({
            "context": context_text,
            "answer": state["answer"]
        })

        return {
            "hallucination_status": status
        }

    def fail_node(state: AgentState):

        return {
            "answer": (
                "I cannot answer this question based on the provided PDF. "
                "The information is not present in the document."
            )
        }

    def relevance_node(state: AgentState):

        result = relevance_retrieval_chain(state["question"])

        if "NOT_RELEVANT" in result.strip().upper():
            return {
                "hallucination_status": "OUT_OF_SCOPE"
            }

        return {
            "hallucination_status": "RELEVANT"
        }

    def route_after_relevance(state: AgentState):

        if state["hallucination_status"] == "OUT_OF_SCOPE":
            return "fail"

        return "generate_queries"

    def route_after_hallucination(state: AgentState):

        if state["hallucination_status"] == "GROUNDED":
            return "end"

        return "fail"

    # graph

    graph = StateGraph(AgentState)
    graph.add_node("generate_queries", multi_query_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("rerank", reciprocal_rank_fusion_node)
    graph.add_node("answer", answer_node)
    graph.add_node("hallucination", hallucination_node)
    graph.add_node("fail", fail_node)
    graph.add_node("relevance", relevance_node)
    graph.set_entry_point("relevance")
    graph.add_conditional_edges(
        "relevance",
        route_after_relevance,
        {
            "generate_queries": "generate_queries",
            "fail": "fail"
        }
    )
    graph.add_edge("generate_queries", "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "answer")
    graph.add_edge("answer", "hallucination")
    graph.add_conditional_edges(
        "hallucination",
        route_after_hallucination,
        {
            "fail": "fail",
            "end":END
        }
    )
    graph.add_edge("fail", END)

    memory = MemorySaver()

    app = graph.compile(checkpointer=memory)

    return app
