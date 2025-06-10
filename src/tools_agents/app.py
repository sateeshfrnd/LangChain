"""
app.py

A Streamlit UI for interacting with a LangChain-powered agent that uses:
- Wikipedia for general knowledge
- Arxiv for academic research
- DuckDuckGo for broader web search

This app provides a text input for users to ask questions.
The question is passed to the agent, which selects the appropriate tool
and returns a smart response.

Run the app:
    streamlit run app.py
"""
import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun, DuckDuckGoSearchRun
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent,AgentType,Tool
from langchain.callbacks import StreamlitCallbackHandler

# Function to create a Wikipedia tool for the LangChain agent.
def wikipedia_tool():    
    # Create wrapper for Wikipedia API
    wiki_api_wrapper = WikipediaAPIWrapper(
        top_k_results=1,
        doc_content_chars_max=250
    )
    # Create function to query Wikipedia
    wiki_func = WikipediaQueryRun(api_wrapper=wiki_api_wrapper)
    return wiki_func

# Function to create an Arxiv tool for the LangChain agent.
def arxiv_tool():
    # Create wrapper for Arxiv API
    arxiv_api_wrapper = ArxivAPIWrapper(
        top_k_results=1,
        doc_content_chars_max=250
    )
    # Create function to query Arxiv
    arxiv_func = ArxivQueryRun(api_wrapper=arxiv_api_wrapper)
    return arxiv_func

# Function to create DuckDuckGo search tool
def duckduckgo_search_tool():
    # Create DuckDuckGo search tool
    search_tool = DuckDuckGoSearchRun(name="Search")
    return search_tool

# Function to setup tools for the LangChain agent.
def setup_tools():
    try:
        wiki = wikipedia_tool()
        arxiv = arxiv_tool()
        search = duckduckgo_search_tool()
        tools = [wiki, arxiv, search]
        if not tools:
            raise ValueError("No tools could be initialized.")
    except Exception as e:
        st.error(f"Error initializing tools: {e}")
        return None
    return tools

# Funtion to render sidebar for configuration and information.
def render_sidebar():
    with st.sidebar:
        st.header("Configuration")
        # TODO: Remove hardcoded API Key
        groq_api_key = st.text_input("Enter your GROQ API Key", type="password", value="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")
        # groq_api_key = st.text_input("Enter your GROQ API Key", type="password")

         # Add a divider
        st.markdown("---")

        # Add About section at the bottom of sidebar
        st.header("About")
        st.markdown("""
        This is a LangChain-powered agent that can search and retrieve information from multiple sources:
        - Look up general knowledge using Wikipedia
        - Fetch academic research from Arxiv        
        - Use DuckDuckGo for broader web search
                    
        You can ask questions in the chat input, and the agent will respond with relevant information from these sources.
        """)

        # Add Reference Links section
        st.header("Reference Links")
        st.markdown("""
        - [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)        
        - [GROQ API](https://console.groq.com/)        
        - [LangChain X Streamlit Agent Examples](https://github.com/langchain-ai/streamlit-agent)
        """)
        return groq_api_key

# Function to render the Streamlit UI.
def render_ui():
    st.set_page_config(page_title="LangChain Search Agent", page_icon=":robot:", layout="wide")
    st.title("LangChain Search Agent")
    st.caption("Ask anything — I can search Wikipedia, Arxiv, LangSmith Docs, and DuckDuckGo search for you!")
    
    # === Render sidebar ===
    api_key = render_sidebar()
    # Pass the key into session_state (optional)
    if api_key:
        st.session_state["groq_api_key"] = api_key
    else:
        st.warning("Please enter your GROQ API key in the sidebar.")
        return   

    # === Initialize chat history ===
    if "messages" not in st.session_state:
        st.session_state["messages"]=[
            # {"role":"assisstant","content":"Hi,I'm a chatbot who can can search Wikipedia, Arxiv, LangSmith Docs, and DuckDuckGo search for you. How can I help you?"}
        ]

    # === Display chat history ===
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg['content'])

    # === Handle user prompts ===
    if prompt:=st.chat_input(placeholder="What is machine learning?"):
        # Add user message to chat history
        st.session_state.messages.append({"role":"user","content":prompt})

        # Display user message in chat
        st.chat_message("user").write(prompt)

        try:           
            # Initialize the LLM with the provided API key
            llm = ChatGroq(groq_api_key=st.session_state["groq_api_key"], model_name="Llama3-8b-8192", streaming=True)

            # Setup tools for the agent
            tools = setup_tools()
            if not tools:
                return

            # Initialize the LangChain agent with the tools
            agent = initialize_agent(
                tools=tools, 
                llm=llm, 
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
                handling_parsing_errors=True
            )

            # Display response from the agent
            with st.chat_message("assistant"):
                st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
                # Pass the user prompt to the agent
                response = agent.run(st.session_state.messages, callbacks=[st_cb])
                st.session_state.messages.append({'role': 'assistant', "content": response})                
                st.write(response)
        except Exception as e:
            st.error(f"An error occurred while processing your request: {e}")
            st.session_state.messages.append({'role': 'assistant', "content": f"Error: {e}"})


if __name__ == "__main__":
    render_ui()
