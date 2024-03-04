from llama_index.core import SimpleDirectoryReader

from llama_index.core import VectorStoreIndex

from llama_index.llms.gemini import Gemini

from IPython.display import Markdown, display

from llama_index.core import ServiceContext

from llama_index.core import StorageContext, load_index_from_storage

import google.generativeai as genai

from llama_index.embeddings.gemini import GeminiEmbedding

from llama_index.core import PromptTemplate

from llama_index.core.prompts.prompt_type import PromptType

import streamlit as st

import os

from dotenv import load_dotenv


load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def load_model():
    model=Gemini(models='gemini-pro',api_key=google_api_key)
    return model

def get_pdf_corpus(doc):
    loader = SimpleDirectoryReader(doc)
    document = loader.load_data()
    return document

def embed_corpus(model, document):
    gemini_embed_model = GeminiEmbedding(model_name="models/embedding-001")
    service_context = ServiceContext.from_defaults(llm=model,embed_model=gemini_embed_model, chunk_size=1024, chunk_overlap=64)
    vector_db = VectorStoreIndex.from_documents(document,service_context=service_context)
    vector_db_index = vector_db.storage_context.persist()
    return vector_db_index

def query_engine(user_question, vector_db_index):
    template = (
    "You are an experienced document question-answering system. We have provided context information below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given this information, please answer the question without any fail: {query_str}\n"
    )
    qa_template = PromptTemplate(template)
    chain_of_thought = vector_db_index.as_query_engine(text_qa_template=qa_template)
    response = chain_of_thought.query(user_question)
    print(response)
    return response

def main():
    st.set_page_config("Document Retriever")
    
    doc=st.file_uploader("UPLOAD YOUR DOCUMENT")
    
    st.header("CHAT WITH DOCUMENT")
    
    user_question= st.text_input("What's your query?")
    
    if st.button("Ask"):
        with st.spinner("Retrieving..."):
            document = get_pdf_corpus(doc)
            model = load_model()
            index = embed_corpus(model, document)
            response = query_engine(user_question, index) 
            st.write("Reply: ", response.response)
            st.success("Done")

                
                
if __name__=="__main__":
    main()    




