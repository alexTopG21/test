import os
import streamlit as st
from dotenv import load_dotenv
import openai

from llama_index.core import ServiceContext, VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def main():
    st.header("Ask any question about robots")
    reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
    docs = reader.load_data()
    service_context = ServiceContext.from_defaults(
        llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5,
                   system_prompt="You are an expert on industrial robots, your task is to give users URL to the website where the robot adapted to their use case is. The urls are in the document provided")
    )
    index = VectorStoreIndex.from_documents(docs, service_context=service_context)
    query = st.text_input("Ask a question we can solve with a robot")
    if query:
        chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)
        response = chat_engine.chat(query)
        url = response.response.strip()
        if "http" in url:  # Check if the response contains a URL
            robot_name = url.split('/')[-1].replace('-', ' ').title()  # Extracting and formatting robot name from URL
            st.markdown(f"**Recommended Robot:**")
            st.markdown(f"[{robot_name}]({url}) - This robot is designed to assist with {query.lower()}. Click the link to learn more about how it can meet your specific needs.")
        else:
            st.write("No specific robot could be found for your query. Please try another question.")

if __name__ == '__main__':
    main()
