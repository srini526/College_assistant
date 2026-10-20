import os
from typing import TypedDict, Annotated
from langchain_groq import ChatGroq
from langchain_cloudflare import ChatCloudflareWorkersAI
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_community.embeddings.cloudflare_workersai import CloudflareWorkersAIEmbeddings
from langchain_cloudflare import CloudflareWorkersAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv


load_dotenv()

# step 1 Model Setup

embedding = CloudflareWorkersAIEmbeddings(model_name="@cf/baai/bge-m3")

def build_retrive(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    document = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size = 800, chunk_overlap = 100)

    chunks = splitter.split_documents(document)

    vectorstore = FAISS.from_documents(chunks,embedding)

    return vectorstore.as_retriever(search_kwargs = {"k":4})

acadamic_retrival = build_retrive("Academic-Regulations-2023.pdf")
fee_retrival = build_retrive("Fee_Structure_AY_2026-27.pdf") 

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.4)


#step 2 state

class State(TypedDict):
    programme_level : str
    programme : str
    messages : Annotated[list,add_messages]
    query_type : str
    retrieved_context : str
    
# Step 3 Node

def classifier_node (state : State) -> dict:
    """It Look at the latest human message and decide which path to take."""
    #print ("\n Classification\n")

    last_message = state["messages"][-1].content

    prompt = (
        "Classify the following student query into exactly one category: "
        "'academic', 'fee', or 'general'.\n\n"
        "Use 'academic' for questions about programmes, degrees, "
        "admission requirements, eligibility, campus offerings,attendance, "
        "exams, grading, credits, promotion, course structure, "
        "academic calendar, summer training, or degree requirements.\n"
        "Use 'fee' for questions about tuition, payment, refund, late charges, "
        "scholarships, or any money-related topic.\n"
        "Use 'general' for greetings, casual talk, or anything not related to "
        "the college rules or fee.\n\n"
        f"Query: {last_message}\n\n"
        "Return only one word: academic, fee, or general."
    )
    response = llm.invoke(prompt)
    category = response.content.strip().lower()

    if "academic" in category:
        category = 'academic'
    elif "fee" in category:
        category = "fee"
    else:
        category = "general"

    return {"query_type": category}

def acadamic_rag_node(state : State) -> dict:
    """Retrives relevant chunks from acadamic handbook"""
    query = state["messages"][-1].content
    docs = acadamic_retrival.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"retrieved_context":context}

def fee_rag_node(state : State) -> dict:
    """Retrives relevant chunks from Fee Structure"""
    query = state["messages"][-1].content
    docs = fee_retrival.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"retrieved_context":context}

def general_node(state : State) -> dict:
    """Answer using the LLM's own Knowledge"""
    return {"retrieved_context": "No_Retrievel_Needed"}

def response_node(state : State) -> dict:
    """Generates the answer using student's personalized programme."""
    query = state["messages"][-1].content
    programme_level = state.get("programme_level", "Unknown")
    programme = state.get("programme","Unknown")
    context = state["retrieved_context"]

    if context == "No_Retrievel_Needed":
        prompt = (
            f"You are a friendly BITS Pilani student assistant helping a {programme_level} student enrolled in {programme}.\n"
            f"Answer this question using your own knowledge:\n\n{query}"
        )
    else:
        prompt = (
            f"You are a BITS Pilani student assistant helping a "
            f"{programme_level} student enrolled in {programme}.\n"
            "Use the provided official-document excerpts as the primary source. "
            "Do not invent programmes, eligibility requirements, policies, "
            "dates, or fees. If the context does not contain the answer, "
            "say that you could not verify it from the available documents. "
            "Programme availability may vary by campus and admission year.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{query}\n\n"
            "Give a clear, friendly and precise answer."
        )
    response = llm.invoke(prompt)
    return {"messages": [("ai", response.content.strip())]}

# step 4 Router function

def router_query(state: State):
    if state["query_type"] == 'academic':
        return "academic_rag"
    elif state["query_type"] == "fee":
        return "fee_rag"
    else:
        return "general"

# step 5 Building Graph

graph = StateGraph(State)

graph.add_node("Classifier", classifier_node)
graph.add_node("academic_rag", acadamic_rag_node)
graph.add_node("fee_rag", fee_rag_node)
graph.add_node("general", general_node)
graph.add_node("response", response_node)

#edges

graph.add_edge(START, "Classifier")

graph.add_conditional_edges(
    "Classifier", router_query
)
graph.add_edge("academic_rag","response")
graph.add_edge("fee_rag","response")
graph.add_edge("general","response")

graph.add_edge("response", END)

app = graph.compile()

def create_assistant():
    return app

# Step 6 run the code

# print("welcome Back\n\n")
# print("which Programme You want know about..??\n")

# print("1. BCA")
# print("2. BBA")
# print("3. B.Com (H)")

# choice = input("\nEnter 1, 2 or 3\n")

# programme_map = {
#     "1": "BCA",
#     "2": "BBA",
#     "3": "B.Com (H)"
# }

# student_programme = programme_map.get(choice,"BCA")

# print(f"\n Great! You are set as a {student_programme} student.")

# while True:
#     user_query = input("you: ")
#     if user_query.lower() in ["exit","quit"]:
#         break

#     result = app.invoke({
#         "programme": student_programme,
#         "messages": [("human", user_query)]
#     })

#     print(f"Assistant: {result["messages"][-1].content} ")

