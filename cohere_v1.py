#pip install streamlit langchain openai faiss-cpu tiktoken

import streamlit as st
from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.csv_loader import CSVLoader
import tempfile
import pandas as pd
from langchain.agents import create_pandas_dataframe_agent
import os
from langchain.llms import OpenAI



user_api_key = st.sidebar.text_input(
    label="#### Your Cohere API key 👇",
    placeholder="Paste your Cohere API key",
    type="password")

uploaded_file = st.sidebar.file_uploader("upload", type="csv")

if uploaded_file :
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    # df = pd.read_csv(tmp_file_path)
    # data = loader.load()

    # embeddings = OpenAIEmbeddings()
    # vectors = Chroma.from_texts(data, embeddings)

    # chain = ConversationalRetrievalChain.from_llm(llm = ChatOpenAI(temperature=0.0,model_name='gpt-3.5-turbo', openai_api_key=user_api_key),
    #                                                                   retriever=vectors.as_retriever())
    # agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True)
    from langchain.embeddings import CohereEmbeddings
    import cohere
    cohere_api_key = user_api_key

    # Adding the Title of the app
    st.title("Ecommerece MultiLingual Chatbot 🤖")

    # Data loading
    st.write("Loading data...")
    loader = CSVLoader(tmp_file_path, encoding="utf-8")
    data = loader.load()

    # Embeddings
    st.write("Loading embeddings...")
    vec = CohereEmbeddings(model ="multilingual-22-12", cohere_api_key=cohere_api_key)

    # Storing data inside the pinecone index
    st.write("Storing data inside the pinecone index...")
    import pinecone 
    from langchain.vectorstores import Pinecone

    # initialize pinecone
    pinecone.init(
        api_key="cbf39944-82a9-4434-9ced-77c4ef516319",  # find at app.pinecone.io
        environment="us-east1-gcp"  # next to api key in console
    )

    index_name = "cohere"

    # create the index
    st.write("Creating the index...")
    # cache the index
    @st.cache_data(show_spinner=True)
    docsearch = Pinecone.from_documents(data, vec, index_name=index_name)

    # Embeddings

    def conversational_chat(query):
        
        result = agent({"question": query, "chat_history": st.session_state['history']})
        st.session_state['history'].append((query, result["answer"]))
        
        return result["answer"]
    
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello ! Ask me anything about " + uploaded_file.name + " 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey ! 👋"]
        
    #container for the chat history
    response_container = st.container()
    #container for the user's text input
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            
            user_input = st.text_input("Query:", placeholder="Talk about your csv data here (:", key='input')
            submit_button = st.form_submit_button(label='Send')
            
        if submit_button and user_input:
            output = docsearch.similarity_search(user_input, k=1)
            output = output[0].page_content
            # covert the out into JSON serializable format
           # output = output.to_dict()
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")
                
#streamlit run streamlit_app.py