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

# Set your OpenAI API key here or use a secret in HF Spaces
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

# Build the RAG pipeline from a CSV file
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

def upload_csv(file):
    global rag_chain
    rag_chain = build_rag(file.name)
    return "CSV loaded and indexed! You can now ask questions."

def ask_question(question):
    if rag_chain is None:
        return "Please upload a CSV file first."
    try:
        answer = rag_chain.invoke({"input": question})
        return answer['answer']
    except Exception as e:
        return f"Error: {e}"

with gr.Blocks() as demo:
    gr.Markdown("# Simple CSV RAG Q&A Bot\nUpload a CSV and ask questions about its content!")
    with gr.Row():
        csv_input = gr.File(label="Upload CSV", file_types=[".csv"])
        upload_btn = gr.Button("Load CSV")
    status = gr.Textbox(label="Status", interactive=False)
    question = gr.Textbox(label="Ask a question about your CSV")
    answer = gr.Textbox(label="Answer", interactive=False)
    ask_btn = gr.Button("Ask")

    upload_btn.click(upload_csv, inputs=csv_input, outputs=status)
    ask_btn.click(ask_question, inputs=question, outputs=answer)

demo.launch()
