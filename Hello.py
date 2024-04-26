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
                   system_prompt="You are an expert on industrial robots. Provide URLs to websites where robots adapted to user cases are listed.")
    )
    index = VectorStoreIndex.from_documents(docs, service_context=service_context)
    query = st.text_input("Ask a question we can solve with a robot")
    if query:
        chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)
        response = chat_engine.chat(query)
        urls = response.response.strip().split()  # Assuming multiple URLs could be returned, separated by spaces
        urls = [url for url in urls if "http" in url]  # Filter to include only valid URLs

        if urls:
            st.markdown("**Recommended Robots:**")
            for url in urls[:3]:  # Limit to top 3 relevant robots
                robot_name = url.split('/')[-1].replace('-', ' ').title()  # Extracting and formatting robot name from URL
                st.markdown(f"- [{robot_name}]({url}) - This robot is designed to meet specialized requirements. Click the link for more details.")
        else:
            st.write("No specific robots could be found for your query. Please try another question or refine your search.")

if __name__ == '__main__':
    main()

