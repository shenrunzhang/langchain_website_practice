import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import requests
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.retrievers.merger_retriever import MergerRetriever



# Load api key from .env
load_dotenv()

# Create the vector store for website content
def get_vectorstore_from_urls(urls):
    document_chunks = []

    # Read through all the websites entered
    for url in urls:
        # get the text in document
        loader = WebBaseLoader(url)
        document = loader.load()

        # text splitter splits document up into several chunks
        # each chunk contains meta data about the entire document
        text_splitter = RecursiveCharacterTextSplitter()
        chunks = text_splitter.split_documents(document)

        document_chunks.extend(chunks)

    # Create the vector store for websites
    vector_store =  FAISS.from_documents(document_chunks, OpenAIEmbeddings())

    return vector_store

# Get the titles of all the webpages entered
def get_website_title(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    return soup.title.string if soup.title else "error"


def get_context_retriever_chain(retriever):
    llm = ChatOpenAI()

    # messages placeholder replaces itself with var chat_history if exists
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation.")
    ])

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    return retriever_chain

def get_conversational_rag_chain(retriever_chain):
    llm = ChatOpenAI()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

# LLM will return a response to the user given user input.
def get_response(user_query):
    # create conversation chain

    # Merge the vector db created from website content 
    # and the vector db created from pdf documents
    retrievers = [st.session_state.vector_store.as_retriever(),
                  st.session_state.vector_store_pdf.as_retriever()]
    merged_retriever = MergerRetriever(retrievers=retrievers)

    # Retriever and conversation rag chains are created from the merged retriever
    retriever_chain = get_context_retriever_chain(merged_retriever)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)

    # Return the response
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_query
    })

    return response['answer']

# Update the vector store from websites.
def update_vectorstore():
    if st.session_state.urls == []:
        st.session_state.vector_store = None
        return
    st.session_state.vector_store = get_vectorstore_from_urls(st.session_state.urls)

# Update the vector store from pdf docs.
def update_vectorstore_pdf(pdf_docs):
    
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    embeddings = OpenAIEmbeddings()
    vector_store_pdf = FAISS.from_texts(texts=chunks, embedding=embeddings)

    st.session_state.vector_store_pdf = vector_store_pdf


def main():
    # app config
    st.set_page_config(page_title="Chat with websites and pdf docs", page_icon="W")
    st.title("Chat with websites and pdf docs!")

    if "vector_store" not in st.session_state:    
        embeddings = OpenAIEmbeddings()
        st.session_state.vector_store = FAISS.from_texts(texts=[""], embedding=embeddings)

        
    if "vector_store_pdf" not in st.session_state:    
        embeddings = OpenAIEmbeddings()
        st.session_state.vector_store_pdf = FAISS.from_texts(texts=[""], embedding=embeddings)

    if "urls" not in st.session_state:
        st.session_state.urls = []

    if "website_titles" not in st.session_state:
        st.session_state.website_titles = []

    # side bar
    with st.sidebar:

        # Adding websites
        st.header("Settings")
        website_url = st.text_input("Website URL")

        if st.button("Add website"):
            # Get website url and title of webpage
            st.session_state.urls.append(website_url)
            st.session_state.website_titles.append(get_website_title(website_url))

            # Update vector store with new content
            update_vectorstore()
        
        # Displays the websites added so far and provides option
        # to delete individual websites.
        if st.session_state.website_titles:
            st.subheader("Added Media")
            for index, title in enumerate(st.session_state.website_titles):
                cols = st.columns([3,1])
                cols[0].write(title)
                if cols[1].button("X", key=f"{index}"):
                    st.session_state.urls.pop(index)
                    st.session_state.website_titles.pop(index)

                    print(st.session_state.urls)
                    update_vectorstore()
                    st.rerun()


        # Adding pdf documents.
        pdf_docs = st.file_uploader(
        "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)


        if st.button("Process"):
                with st.spinner("Processing"):

                    update_vectorstore_pdf(pdf_docs)


    if (st.session_state.urls is None or st.session_state.urls == []) and not pdf_docs:
        st.info("Please enter a website URL or upload pdf documents")
    else:
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [
                AIMessage(content="Hello, I am a bot. How can I help you?")
            ]
        

        # user input
        user_query = st.chat_input("Type your message here...")

        if user_query is not None and user_query != "":
            response = get_response(user_query)
            st.write(response)
            st.session_state.chat_history.append(HumanMessage(content=user_query))
            st.session_state.chat_history.append(AIMessage(content=response))

        # Back and forth conversation
        for message in st.session_state.chat_history:
            if isinstance(message, AIMessage):
                with st.chat_message("AI"):
                    st.write(message.content)
            elif isinstance(message, HumanMessage):
                with st.chat_message("Human"):
                    st.write(message.content)


if __name__ == "__main__":
    main()