import gradio as gr
import pandas as pd
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import faiss
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

# --- Advanced RAG Query Transformation Functions ---
from langchain.prompts import PromptTemplate

def get_query_rewriter():
    re_write_llm = ChatOpenAI(temperature=0, model_name="gpt-4o", max_tokens=4000)
    query_rewrite_template = """You are an AI assistant tasked with reformulating user queries to improve retrieval in a RAG system.\nGiven the original query, rewrite it to be more specific, detailed, and likely to retrieve relevant information.\n\nOriginal query: {original_query}\n\nRewritten query:"""
    query_rewrite_prompt = PromptTemplate(
        input_variables=["original_query"],
        template=query_rewrite_template
    )
    return query_rewrite_prompt | re_write_llm

def get_step_back_chain():
    step_back_llm = ChatOpenAI(temperature=0, model_name="gpt-4o", max_tokens=4000)
    step_back_template = """You are an AI assistant tasked with generating broader, more general queries to improve context retrieval in a RAG system.\nGiven the original query, generate a step-back query that is more general and can help retrieve relevant background information.\n\nOriginal query: {original_query}\n\nStep-back query:"""
    step_back_prompt = PromptTemplate(
        input_variables=["original_query"],
        template=step_back_template
    )
    return step_back_prompt | step_back_llm

def get_subquery_decomposer():
    sub_query_llm = ChatOpenAI(temperature=0, model_name="gpt-4o", max_tokens=4000)
    subquery_decomposition_template = """You are an AI assistant tasked with breaking down complex queries into simpler sub-queries for a RAG system.\nGiven the original query, decompose it into 2-4 simpler sub-queries that, when answered together, would provide a comprehensive response to the original query.\n\nOriginal query: {original_query}\n\nexample: What are the impacts of climate change on the environment?\n\nSub-queries:\n1. What are the impacts of climate change on biodiversity?\n2. How does climate change affect the oceans?\n3. What are the effects of climate change on agriculture?\n4. What are the impacts of climate change on human health?"""
    subquery_decomposition_prompt = PromptTemplate(
        input_variables=["original_query"],
        template=subquery_decomposition_template
    )
    return subquery_decomposition_prompt | sub_query_llm

def rewrite_query(original_query):
    chain = get_query_rewriter()
    response = chain.invoke({"original_query": original_query})
    return response.content

def generate_step_back_query(original_query):
    chain = get_step_back_chain()
    response = chain.invoke({"original_query": original_query})
    return response.content

def decompose_query(original_query):
    chain = get_subquery_decomposer()
    response = chain.invoke({"original_query": original_query}).content
    sub_queries = [q.strip() for q in response.split('\n') if q.strip() and not q.strip().startswith('Sub-queries:')]
    return sub_queries

# --- RAG Pipeline ---
def build_rag(csv_path):
    loader = CSVLoader(file_path=csv_path)
    docs = loader.load_and_split()
    embeddings = OpenAIEmbeddings()
    index = faiss.IndexFlatL2(len(embeddings.embed_query(" ")))
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )
    vector_store.add_documents(documents=docs)
    retriever = vector_store.as_retriever()
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise.\n\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    return rag_chain

rag_chain = None
csv_loaded = False

# --- Gradio App ---
def upload_csv(file):
    global rag_chain, csv_loaded
    rag_chain = build_rag(file.name)
    csv_loaded = True
    return "CSV loaded and indexed! You can now ask questions."

def ask_question(question, transform):
    if not csv_loaded:
        return "Please upload a CSV file first."
    try:
        # Apply query transformation if selected
        if transform == "Rewrite Query":
            question = rewrite_query(question)
        elif transform == "Step-back Prompt":
            question = generate_step_back_query(question)
        elif transform == "Sub-query Decomposition":
            sub_queries = decompose_query(question)
            answers = []
            for sub_q in sub_queries:
                answer = rag_chain.invoke({"input": sub_q})
                answers.append(f"Q: {sub_q}\nA: {answer['answer']}")
            return "\n\n".join(answers)
        answer = rag_chain.invoke({"input": question})
        return answer['answer']
    except Exception as e:
        return f"Error: {e}"

with gr.Blocks() as demo:
    gr.Markdown("# Advanced CSV RAG Q&A Bot\nUpload a CSV and ask questions about its content! Choose advanced query transformation techniques for better answers.")
    with gr.Row():
        csv_input = gr.File(label="Upload CSV", file_types=[".csv"])
        upload_btn = gr.Button("Load CSV")
    status = gr.Textbox(label="Status", interactive=False)
    question = gr.Textbox(label="Ask a question about your CSV")
    transform = gr.Radio(["None", "Rewrite Query", "Step-back Prompt", "Sub-query Decomposition"], label="Query Transformation", value="None")
    answer = gr.Textbox(label="Answer", interactive=False)
    ask_btn = gr.Button("Ask")

    upload_btn.click(upload_csv, inputs=csv_input, outputs=status)
    ask_btn.click(ask_question, inputs=[question, transform], outputs=answer)

demo.launch()
