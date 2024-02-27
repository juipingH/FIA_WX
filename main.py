import pickle
import tempfile
import time
from PIL import Image
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import (CharacterTextSplitter,
                                     RecursiveCharacterTextSplitter)
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from core import config
from src import answer_generator, table_generator

embedding_model = config.embedding_model
db_name = config.db_filename
filemap_name = config.filemap_name
input_file = config.main_table
icon = Image.open("fia_image.png")
st.set_page_config(
    page_title="FIA",
    page_icon=icon,
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
    <style>
        .reportview-container {
            margin-top: -2em;
        }
        #MainMenu {visibility: hidden;}
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
        .big-font {
            font-size:300px !important;}
    </style>
""", unsafe_allow_html=True)

hide_streamlit_style = """
                <style>
                div[data-testid="stToolbar"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                div[data-testid="stDecoration"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                div[data-testid="stStatusWidget"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                #MainMenu {
                visibility: hidden;
                height: 0%;
                }
                header {
                visibility: hidden;
                height: 0%;
                }
                footer {
                visibility: hidden;
                height: 0%;
                }
                </style>
                """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
# st.image('fia_image.png')
# st.header("FIA Watsonx Demo")
col1, mid, col2 = st.columns([1,1,20])
with col1:
    st.image('/Users/erinhsu/Documents/GitHub/Pilot-FIA-Taiwan/fia_image.png', width=60)
with col2:
    st.header("FIA Watsonx Demo")
        
@st.cache_data
def read_pdf(uploaded_files,chunk_size =500,chunk_overlap=0):
    file_map = {}
    docs = []
    for uploaded_file in uploaded_files:
      bytes_data = uploaded_file.read()
      with tempfile.NamedTemporaryFile(mode='wb', delete=False) as temp_file:
      # Write content to the temporary file
          temp_file.write(bytes_data)
          filepath = temp_file.name
          print(filepath)
          file_map[filepath] = uploaded_file.name
      with st.spinner('Waiting for the file to upload'):
        loader = PyPDFLoader(filepath)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size= chunk_size, chunk_overlap=chunk_overlap)
        docs += text_splitter.split_documents(data)
        with open(filemap_name, 'wb') as file:
            pickle.dump(file_map,file)
        st.session_state.filemap = file_map
    return docs
          
@st.cache_data        
def read_txt(files, chunk_size =1200,chunk_overlap=200):
    for uploaded_file in files:
      bytes_data = uploaded_file.read()
      with tempfile.NamedTemporaryFile(mode='wb', delete=False) as temp_file:
      # Write content to the temporary file
          temp_file.write(bytes_data)
          filepath = temp_file.name
          with st.spinner('Waiting for the file to upload'):
             with open(filepath) as f:
                raw_docs = f.read()
             text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0, length_function=len)
             docs = text_splitter.create_documents([raw_docs])
             return docs

def read_push_embeddings(docs):
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(db_name)
    return db

if "db" not in st.session_state:
    st.session_state.db = None

if "filemap" not in st.session_state:
    st.session_state.filemap = None
    

# Sidebar contents
with st.sidebar:
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    st.title("稅收徵起查詢")
    css='''
    <style>
    [data-testid="stFileUploadDropzone"] div div::before {color:black; content:"請上傳所需文件"}
    [data-testid="stFileUploadDropzone"] div div span{display:none;}
    [data-testid="stFileUploadDropzone"] div div::after {color:black; font-size: .8em; content:"文件大小限制為200MB"}
    [data-testid="stFileUploadDropzone"] div div small{display:none;}
    </style>
    '''
    st.markdown(css, unsafe_allow_html=True)
    uploaded_files = st.file_uploader(label="請上傳相關文件", accept_multiple_files=True)

    if st.session_state.db is None:
        try:
            st.session_state.db = FAISS.load_local(db_name, embeddings)
            with open(filemap_name,'rb') as filemap_name:
                st.session_state.filemap = pickle.load(filemap_name)
        except:
            starttime = time.time()
            docs = read_pdf(uploaded_files)
            if docs is not None and len(docs) > 0:
                st.session_state.db = read_push_embeddings(docs)
            endtime = time.time()
            print(f"take {endtime-starttime} to ingest the doc to vectordb")
    

def get_chunks_details(chunks, file_map):
    print(file_map)
    all_source = []
    for chunk in chunks:
        source = file_map[chunk.metadata['source']]
        page = chunk.metadata['page']
        all_source.append(f"文檔:{source}, 第{page+1}頁")
    return all_source

def run():
    main_df = pd.read_csv(input_file, encoding='utf-8-sig', thousands=",")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if query := st.chat_input("請輸入你的提問"):
        document_source = []
        sources_string = ''
        with st.chat_message("user"):
            st.markdown(query)

        st.session_state.messages.append({"role": "user", "content": query})
        with st.spinner(text="正在查詢中...", cache=False):
            starttime = time.time()

            question_type = answer_generator.evaluate_question(query)
            if not "否" in question_type:
                filtered_table, pivot_table = table_generator.get_table(main_df, query)
                # chunks = st.session_state.db.similarity_search(query)
                retriever = st.session_state.db.as_retriever()
                years = filtered_table["徵收年"].to_list()
                query_year = query + ",".join(years)
                chunks = retriever.invoke(query_year)
                # chunks = retriever.get_relevant_documents(query_year)
                endtime = time.time()
                print(f"take {endtime-starttime} to search similary")
                print("=====Chunks=====")
                print(chunks)
                all_chunks = [chunk.page_content.replace("\\n"," ") for chunk in chunks]
                starttime = time.time()
                answer = answer_generator.generate_answer(query, pivot_table.to_csv(), all_chunks)
                # translated_answer = answer_generator.translate_answer(answer)
                endtime = time.time() 
                print(f"take {endtime-starttime} to build the answer")
                print(answer)
                # print(translated_answer)
                document_source = get_chunks_details(chunks, st.session_state.filemap)
                for doc in document_source:
                    sources_string += "- " + doc + "\n"
                # print("==========Translated=========")
                # print(translated_answer)
            else:
                answer = "抱歉！我不可以回答這一類問題。"
            st.session_state.messages.append({"role": "agent", "content": f"{answer}\n\n來源:\n{sources_string}"}) 
            
        

            with st.chat_message("agent",avatar="/Users/erinhsu/Documents/GitHub/Pilot-FIA-Taiwan/fia_image.png"):
                st.markdown(answer.strip())
                st.markdown(f"來源:\n{sources_string}")
                if "經濟成長率" in query:
                    table_generator.chart_generator(pivot_table, query)
                    st.image(config.plot_name)

                           
run()