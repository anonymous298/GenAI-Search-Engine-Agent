# Importing Necessary Dependencies
import modulefinder
import os
import webbrowser
import streamlit as st

from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun, DuckDuckGoSearchRun
from langchain_community.document_loaders import  WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import  AgentExecutor, initialize_agent, AgentType
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.callbacks import StreamlitCallbackHandler

# Building our retriever tool
def get_retriever_tool():
    try: 
        loader = WebBaseLoader("https://docs.smith.langchain.com/overview")
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents = text_splitter.split_documents(docs)

        embedding = OllamaEmbeddings(model='mxbai-embed-large:335m')
        vectordb = Chroma.from_documents(documents=documents, embedding=embedding)

        retriever = vectordb.as_retriever()

        retriever_tool = create_retriever_tool(
            retriever,
            'retriever_tool',
            'RAG Vectordb for our agent'
        )

        return retriever_tool

    except Exception as e:
        print(e)

# Building Tools for our agents
def get_tools():
    '''
    This function will build tools for our agents.
    '''
    
    try:
        # Building our Arxiv Tool
        arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
        arxiv_tool = ArxivQueryRun(api_wrapper = arxiv_wrapper)

        # Building our Wikipedia Tool
        wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
        wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)

        # Building our DuckDuckGoSearch Tool
        search = DuckDuckGoSearchRun(name='Search')

        # Returing the tools
        return (
            arxiv_tool,
            wiki_tool,
            search
        )

    except Exception as e:
        print(e)

# Building our Streamlit Application
st.title('Basic Search-Engine Agent Using Langchain')

def initialize_sessions():
    '''
    This will initialize all the steps of building the agents in session so it will not exectue again and again.
    '''

    st.session_state.arxiv_tool, st.session_state.wiki_tool, st.session_state.search_tool = get_tools()
    # st.session_state.retriever_tool = get_retriever_tool()
    st.session_state.tools = [st.session_state.arxiv_tool, st.session_state.wiki_tool, st.session_state.search_tool]

    st.session_state.llm = ChatOllama(
        model = 'llama3.2:3b'
    )

    st.session_state.agent = initialize_agent(
        llm = st.session_state.llm,
        tools = st.session_state.tools,
        agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose = True,
        handling_parsing_errors = True
    )

if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {'role' : 'assistant', 'content' : 'Hello! I am a chatbot who can search the web. How can I help you today?'}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

if prompt := st.chat_input('Enter what you want to do with the agent'):

    st.session_state.messages.append({'role' : 'user', 'content' : prompt}) 

    # Initializing all the streamlit session of creating an agent
    if 'agent' not in st.session_state:
        initialize_sessions()

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        response = st.session_state.agent.invoke(
            st.session_state.messages,
            callbacks = [st_callback]
        )

        st.session_state.messages.append({'role' : 'assistant', 'content' : response['output']})

        st.write(response["output"])
        