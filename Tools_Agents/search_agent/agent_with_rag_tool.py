"""
app.py

A Streamlit UI for interacting with a LangChain-powered agent that uses:
- Wikipedia for general knowledge
- Arxiv for academic research
- A custom RAG tool for LangSmith documentation

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

# Function to get the tools for the LangChain agent.
def get_tools():
    tools = [
        wikipedia_tool(),
        arxiv_tool(),
        duckduckgo_search_tool(),        
    ]
    try:
        # Get the RAG tool from session state
        langsmith_tool = initialize_rag_tool()
        if langsmith_tool:
            tools.append(langsmith_tool)
    except Exception as e:
        st.error(f"Error initializing LangSmith RAG tool: {str(e)}")       
            
    return tools

# Function to render the sidebar with configuration options and information.
def render_sidebar():
    with st.sidebar:
        st.header("Configuration")
        groq_api_key = st.text_input("Enter your GROQ API Key", type="password", value="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")

         # Add a divider
        st.markdown("---")

        # Add About section at the bottom of sidebar
        st.header("About")
        st.markdown("""
        This is a LangChain-powered agent that can search and retrieve information from multiple sources:
        - Look up general knowledge using Wikipedia
        - Fetch academic research from Arxiv
        - Search LangSmith documentation using a custom RAG tool
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

# Function to create a RAG tool for LangSmith documentation.
def create_rag_tool():
    print(f"===== Creating LangSmith RAG tool =====")    
    try:
        # Create a web loader for a specific URL
        web_loader = WebBaseLoader(
            web_paths=["https://docs.smith.langchain.com/"],
            verify_ssl=False,            
        )
        docs = web_loader.load()

        if not docs:
            raise ValueError("No documents loaded from LangSmith docs")
        
        # Print the loaded documents
        for doc in docs:
            print(doc.page_content[:50])  
            print(doc.metadata)  

        # Create a text splitter to split documents into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False
        )
        
        # Split the loaded documents into smaller chunks
        split_docs = text_splitter.split_documents(docs)
        print(f"Split into {len(split_docs)} chunks")

        if not split_docs:
            raise ValueError("No documents could be split")

        print("Creating embeddings model...")
        # Create an embeddings model using Hugging Face
        embeddings_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False}
        )
        print("Embeddings model created.")

        # Create a vector store using FAISS with the embeddings model
        try:
            vector_store = FAISS.from_documents(
                documents=split_docs,
                embedding=embeddings_model
            )
            print(f"Vector store created with {len(vector_store)} documents.")
        except Exception as e:
            print(f"Error creating vector store: {str(e)}")
            #Try persisting and re-loading the vector store
            FAISS.save_local(vector_store, "langsmith_vector_store")
            vector_store = FAISS.load_local("langsmith_vector_store", embeddings_model)

        # Create a retriever from the vector store
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 2}
        )

        # Create a retriever tool using the retriever
        retriever_tool = Tool(
            name="LangSmithSearch",
            func=lambda q: retriever.get_relevant_documents(q),
            description="A tool to search and retrieve information from the LangSmith documentation. Use this when you need to find information about LangSmith features, setup, or usage."
        )
        return retriever_tool
    except Exception as e:
        st.error(f"Error creating LangSmith RAG tool: {str(e)}")
        raise

# Function to initialize the RAG tool and store it in session state
def initialize_rag_tool():
    """Initialize the RAG tool once and store it in session state"""
    if "langsmith_rag_tool" not in st.session_state:
        try:
            retriever_tool = create_rag_tool()
            st.session_state["langsmith_rag_tool"] = retriever_tool
            return retriever_tool
        except Exception as e:
            st.error(f"Error initializing LangSmith RAG tool: {str(e)}")
            return None
    return st.session_state["langsmith_rag_tool"]


def render_ui():    
    st.set_page_config(page_title="LangChain Search Agent", page_icon=":robot:", layout="wide")
    st.title("LangChain Search Agent")
    st.caption("Ask anything — I can search Wikipedia, Arxiv, and DuckDuckGo search for you!")   

    # === Render sidebar ===
    api_key = render_sidebar()
    # Pass the key into session_state (optional)
    if api_key:
        st.session_state["groq_api_key"] = api_key
    else:
        st.warning("Please enter your GROQ API key in the sidebar.")
        return

    # === Initialize the RAG tool ===
    # Check if the RAG tool is already initialized in session state
    if "langsmith_rag_tool" not in st.session_state:
        with st.spinner("Initializing LangSmith RAG tool..."):
            initialize_rag_tool()

    # === Initialize chat history ===
    # Check if the messages session state exists, if not, initialize it
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
        ]

    # === Display chat messages ===
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg['content'])

    # === Chat input for user prompt ===
    if prompt := st.chat_input(placeholder="What is machine learning?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat
        st.chat_message("user").write(prompt)

        try:
            # Initialize the LLM with the provided API key
            llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)
            # Setup tools for the agent
            tools = get_tools()
            
            if not tools:
                st.error("No tools available. Please check the configuration.")
                return
                
            print(f'Using tools: {[tool.name for tool in tools]}')

            # Initialize the agent with explicit parameters
            search_agent = initialize_agent(
                tools=tools,
                llm=llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                handle_parsing_errors=True,
                verbose=True,
                max_iterations=3,
                early_stopping_method="generate"
            )

            with st.chat_message("assistant"):
                st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
                try:
                    # Pass the user prompt to the agent
                    response = search_agent.invoke(
                        {"input": prompt}, callbacks=[st_cb], include_run_info=True
                    )
                    # Handle the response
                    # TODO: Remove debug print statement
                    print(f'Response: {response}')
                    # Check if the response is a dictionary and extract the output
                    if isinstance(response, dict) and "output" in response:
                        response_text = response["output"]
                    else:
                        response_text = str(response)
                    # Append the response to session state                   
                    st.session_state.messages.append({'role': 'assistant', "content": response_text})
                    # Display the response in the chat
                    st.write(response_text)
                except Exception as e:
                    error_message = f"Error during agent execution: {str(e)}"
                    st.error(error_message)
                    st.session_state.messages.append({'role': 'assistant', "content": error_message})
        except Exception as e:
            error_message = f"Error initializing agent: {str(e)}"
            st.error(error_message)
            st.session_state.messages.append({'role': 'assistant', "content": error_message})


if __name__ == "__main__":
    render_ui()
