from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.document_loaders import PyPDFLoader
import streamlit as st
from dotenv import load_dotenv
import os
load_dotenv()
import chromadb
from chromadb.config import Settings

##

os.environ["OPENAI_API_KEY"]= os.getenv("OPENAI_API_KEY")


embeddings = OpenAIEmbeddings(model= "text-embedding-3-large")

st.title("Conversational RAG with PDF Uploads and Chat History")
st.write("Upload Pdfs and chat with their content")

api_key = st.text_input("Enter your groq api key", type = "password")

if api_key:
    llm = ChatGroq(groq_api_key = api_key, model = "llama-3.1-8b-instant")

    session_id = st.text_input("Session ID",value = "Default_Session_ID" )

    if "store" not in st.session_state:
        st.session_state.store = {}

    uploaded_files = st.file_uploader("Upload a pdf file", type = "pdf", accept_multiple_files= True)

    if uploaded_files:
        documents = []

        for file in uploaded_files:
            temp_path = f"./{file.name}"
            with open(temp_path, "wb") as f:
                f.write(file.getvalue())
            
            loader = PyPDFLoader(temp_path)
            docs = loader.load()
            documents.extend(docs)
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 5000, chunk_overlap = 200)
        splits = text_splitter.split_documents(documents)

        if "vectorstore_ready" not in st.session_state:
            CHROMA_DIR = "/tmp/chroma_db"

            chroma_client = chromadb.PersistentClient(
                path=CHROMA_DIR,
                settings=Settings(anonymized_telemetry=False),
            )

            st.session_state.vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                client=chroma_client,
                collection_name="rag_docs",
            )

            st.session_state["vectorstore_ready"] = True
        
        vectorstore = st.session_state.vectorstore
        retriever = vectorstore.as_retriever()

        contextualize_q_system_prompt = (
            """ Given the chat history and latest user question
            which might reference context in the chat history ,
            formulate a standalone question which can be understood 
            without the chat history , Do not answer the question,
            just reformulate it if needed and otherwise return it as is. """
        )    
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        history_aware_retriever = create_history_aware_retriever(llm,retriever,contextualize_q_prompt)

        system_prompt = (
            """You are an assistant for question answering task , use the following 
            pieces of retrieved context to answer  the question, If you don't know the answer, say
            that you don't know, Use three sentences maximum  and keep the  answer concise. 
            \n\n 
            {context}"""
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ('human', "{input}")
            ]
        )

        question_answer_chain = create_stuff_documents_chain(llm,qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever,question_answer_chain)


        def get_session_history(session_id : str)-> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]


        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain, get_session_history, input_messages_key= "input",
            history_messages_key = "chat_history",
            output_messages_key = "answer"
        )


        user_input = st.text_input("Your Question")
        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config = {"configurable": {"session_id": session_id}}
            )
            st.write(st.session_state.store)
            st.success(f"Assistant: {response['answer']}")
            st.write("Chat History: ", session_history.messages)    


else:
    st.warning("Please Enter the api key")







