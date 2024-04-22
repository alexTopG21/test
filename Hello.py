import os
import streamlit as st
from dotenv import load_dotenv
import openai

# Import API key directly from config.py
from config import OPENAI_API_KEY
from llama_index.core import ServiceContext, VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI

# Load environment variables from .env file
load_dotenv()
# Get API key from environment, or use the one from config.py if not found
openai.api_key = os.getenv("OPENAI_API_KEY", OPENAI_API_KEY)

def main():
    st.header("Ask any question about robots")
    reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
    docs = reader.load_data()

    # Since ServiceContext.from_defaults might be deprecated or incorrect, check the latest usage in the llama_index documentation
    # Assuming it's still correct, use it here:
    service_context = ServiceContext.from_defaults(
        llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5,
                   system_prompt="You are an expert on industrial robots, tasked with helping users find the ideal robot for their needs from our catalog. Focus on specifications like payload, reach, and application suitability. Answer only questions about industrial robots, ignoring unrelated topics.")
    )
    index = VectorStoreIndex.from_documents(docs, service_context=service_context)
    query = st.text_input("Ask a question we can solve with a robot")
    if query:
        chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)
        response = chat_engine.chat(query)
        st.write(response.response)

if __name__ == '__main__':
    main()
