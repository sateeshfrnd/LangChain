# Module to create a SQL agent that can execute SQL queries and return results.

import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.callbacks import StreamlitCallbackHandler
from sqlalchemy import create_engine
import sqlite3
from urllib.parse import quote_plus

# Create ENUM for database types.
class DatabaseType:
    SQLITE = "sqlite"
    MYSQL = "mysql"

# Function to render the sidebar for configuration and information.
def render_sidebar():
    with st.sidebar:
        st.header("Configuration")
        st.caption("This SQL agent can execute SQL queries on a SQLite database.")

        groq_api_key = st.text_input("Enter your GROQ API Key", type="password")

        # Provide options for selecting database type
        db_type = st.selectbox(
            "Select Database Type",
            options=[DatabaseType.SQLITE, DatabaseType.MYSQL],
            index=0
        )

        
        # Get additional configuration based on selected database type
        db_url = None
        if db_type == DatabaseType.MYSQL:
            host = st.text_input("Enter MySQL Host", value="localhost")
            port = st.number_input("Enter MySQL Port", value=3306, min_value=1, max_value=65535)
            user = st.text_input("Enter MySQL User", value="root")
            password = st.text_input("Enter MySQL Password", type="password")
            database = st.text_input("Enter MySQL Database Name", value="test_db")
            # validate inputs
            if not host or not port or not user or not password or not database:
                st.warning("Please provide valid MySQL connection details.")
            else:
                db_url = f"mysql+mysqlconnector://{user}:{quote_plus(password)}@{host}:{port}/{database}"
                # print(f">>>>> Using MySQL database URL: {db_url}")
        elif db_type == DatabaseType.SQLITE:
            db_file = st.text_input("Enter SQLite Database File Path", value="company.db")
            if db_file:
                db_url = f"sqlite:///{db_file}"
            else:
                st.warning("Please provide a valid SQLite database file path.")


        # Add a divider
        st.markdown("---")
        st.markdown("### About")
        st.markdown(
            "This agent uses LangChain and Groq to execute SQL queries and return results."
        )
        return groq_api_key, db_url

@st.cache_resource(ttl="2h")
def connect_to_database(db_url):
    return SQLDatabase(create_engine(db_url))
    

# Function to render the Streamlit UI for the SQL agent.
def render_ui():
    st.set_page_config(page_title="SQL Agent: Chat with SQL DB", page_icon=":robot_face:", layout="wide")
    st.title("SQL Agent: Chat with SQL DB")
    st.caption("Ask anything about the SQL database, and I will execute SQL queries to find the answers for you!")

    # === Render sidebar ===
    api_key, db_url = render_sidebar()
    # Pass the key into session_state (optional)
    if api_key:
        st.session_state["groq_api_key"] = api_key
    else:
        st.warning("Please enter your GROQ API key in the sidebar.")
        return   
    
    if db_url is None:
        st.warning("Please provide a valid database details in the sidebar.")
        return
    
    
    # Connect to the database
    try:
        db = connect_to_database(db_url)
    except Exception as e:
        st.error(f"Error connecting to the database: {e}")
        return
    
    # Create LLM model
    try:
        llm = ChatGroq(groq_api_key=api_key,model_name="Llama3-8b-8192",streaming=True)
    except Exception as e:
        st.error(f"Error initializing LLM: {e}")
        return
    
    # Create the SQL database toolkit
    try:
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    except Exception as e:
        st.error(f"Error creating SQL database toolkit: {e}")
        return
    
    # Create the SQL agent
    agent = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        verbose=True
    )

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Display chat history
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg['content'])

    # Handle user prompts
    if prompt := st.chat_input(placeholder="show me all the records from the table employees"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message in chat
        st.chat_message("user").write(prompt)

        try:
            # Display response from the agent
            with st.chat_message("assistant"):
                st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
                # Pass the user prompt to the agent
                response = agent.run(prompt, callbacks=[st_cb])
                st.session_state.messages.append({'role': 'assistant', "content": response})                
                st.write(response)
        except Exception as e:
            st.error(f"An error occurred while processing your request: {e}")
            st.session_state.messages.append({'role': 'assistant', "content": f"Error: {e}"})

if __name__ == "__main__":
    render_ui()
