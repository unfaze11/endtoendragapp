import json
import os
import sys
import boto3
import streamlit as st


# we will be using titan embedding model to generate embedding

from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.llms.bedrock import Bedrock

# Data Ingestion

import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

# vector embedding and vector store

from langchain.vectorstores import FAISS

# LLM Models
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

## Bedrock clients

bedrock=boto3.client(service_name="bedrock-runtime")
bedrock_embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",client=bedrock)

#Data Ingestion

def data_ingestion():
    loader=PyPDFDirectoryLoader("data")
    documents=loader.load()

    # - in our testing Character split works better with this pdf data set
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000,
                                                 chunk_overlap=1000)
    
    docs=text_splitter.split_documents(documents)
    return docs

# Vector embedding and vector store

def get_vector_store(docs):
    vectorstore_faiss=FAISS.from_documents(
        docs,
        bedrock_embeddings
    )
    vectorstore_faiss.save_local("faiss_index")

def get_titan_llm():
    llm=Bedrock(model_id="amazon.titan-text-premier-v1:0", client=bedrock)
    
    return llm

prompt_template = """

Human: Use the following pieces of context to provide a concise 
answer to the question at the end but use atleast summarise with
250 words with detailed explanations. If you don't know the answer,
just say that you don't know, don't try to make up an answer
<context>
{context}
</context

Question: {question}

Assistant: """

PROMPT=PromptTemplate(
    template=prompt_template, input_variables=["context","question"]
)

def get_response_llm(llm,vectorstor_faiss,query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstor_faiss.as_retriever(
            search_type="similarity", search_kwargs={"k":3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt":PROMPT}
    )
    answer=qa({"query":query})
    return answer['result']

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using AWS Bedrock")

    user_question=st.text_input("Ask a Question from the PDF Files")

    with st.sidebar:
        st.title("Update or create vector store")

        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs=data_ingestion()
                get_vector_store(docs)
                st.success("Done")
    
    if st.button("Titan Output"):
        with st.spinner("Processing...."):
            faiss_index=FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            llm=get_titan_llm()

            st.write(get_response_llm(llm,faiss_index,user_question))
            st.success("Done")

if __name__ == "__main__":
    main()