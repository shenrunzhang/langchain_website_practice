# pip install streamlit langchain lanchain-openai beautifulsoup4 python-dotenv chromadb

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


# For integration with Huggingface
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from langchain.llms import HuggingFaceHub
from transformers import AutoConfig
from huggingfacewrapper import HuggingFaceWrapper
# torch.cuda.set_device(1) 
from langchain.globals import set_debug
from langchain import PromptTemplate, LLMChain
from langchain.chains import ConversationalRetrievalChain


load_dotenv()
set_debug(True)

def init_llm():
    # Initializes huggingface llm
    # model_path = "meta-llama/Llama-2-7b-chat-hf"
    model_path = "nvidia/Llama3-ChatQA-1.5-8B"

    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    pipe = pipeline("text-generation",
                                    model=model,
                                    tokenizer=tokenizer,
                                    torch_dtype=torch.float16,
                                    device="cuda",
                                    max_new_tokens=4096,
                                    temperature =0.1
                                    )
        
    llm = HuggingFaceWrapper(pipeline=pipe)
    # llm = HuggingFacePipeline(pipeline=pipe)

    # llm = HuggingFaceHub(repo_id="meta-llama/Llama-2-13b", model_kwargs={"temperature":0.5, "max_length":512})

    return llm

def get_vectorstore_from_url(url):
    # get the text in document form
    loader = WebBaseLoader(url)
    document = loader.load()
    
    # split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=20,
        )
    document_chunks = text_splitter.split_documents(document)
    
    # create a vectorstore from the chunks
    vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings())

    return vector_store

def format_chat_history_display(chat_history):
    formatted_history = ""

    for msg in chat_history:
        if isinstance(msg, AIMessage):
            formatted_history += f"Assistant: {msg.content}\n\n"
        elif isinstance(msg, HumanMessage):
            formatted_history += f"User: {msg.content}\n\n"

    return formatted_history

# Function to get a response from the chatbot
def get_response(user_input):
    # Contextualize question
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, just "
        "reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        st.session_state.llm, 
        st.session_state.vector_store.as_retriever(), 
        contextualize_q_prompt
    )

    # Answer question
    qa_system_prompt = """Answer the user's questions based on the below context.
        
        {context}
        """
    

    template = "System: " + qa_system_prompt + "\n\n"

    template += format_chat_history_display(st.session_state.chat_history)

    template += "User: {input} \n\n Assistant: "

    prompt_template = PromptTemplate(template=template, input_variables=["context",
                                                                         "chat_history",
                                                                         "input"])

    # Create stuff_documents_chain to feed retrieved context into the llm

    question_answer_chain = create_stuff_documents_chain(st.session_state.llm, prompt_template) 
    rag_chain = create_retrieval_chain(
    history_aware_retriever, question_answer_chain
    )

    response = rag_chain.invoke({"input": user_input, "chat_history": st.session_state.chat_history})



    return response['answer']

if "llm" not in st.session_state:
    st.session_state.llm = init_llm()

# app config
st.set_page_config(page_title="Chat with websites", page_icon="🤖")
st.title("Chat with websites")

# sidebar
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL")

if website_url is None or website_url == "":
    st.info("Please enter a website URL")

else:
    # session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorstore_from_url(website_url)    

    # user input
    user_query = st.chat_input("Type your message here...")
    if user_query is not None and user_query != "":
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))
        
       

    # conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)