import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma 

def load_document(file):
    import os 
    from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
    _, extension = os.path.splitext(file)
    print(f"Loading {file}...")
    if extension == ".pdf":
        loader = PyPDFLoader(file)
    elif extension == ".docx":
        loader = Docx2txtLoader(file)
    elif extension == ".txt":
        loader = TextLoader(file, encoding="utf-8")
    return loader.load()

def load_from_wikipedia(query, language="en", max_pages=2):
    from langchain.document_loaders import WikipediaLoader
    loader = WikipediaLoader(query=query,lang=language, load_max_docs=max_pages)
    documents = loader.load()
    return documents

def caluculate_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    total_tokens = sum([len(enc.encode(text.page_content)) for text in texts])
    total_cost = round(total_tokens / 1000 * 0.0004,5)
    return total_tokens, total_cost

def chunk_data(data,chunk_size=512):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=20)
    docs = text_splitter.split_documents(data)
    return docs

def insert_embeddings(index_name, chunks):
    import os
    import pinecone
    from langchain.vectorstores import Pinecone 
    from langchain.embeddings.openai import OpenAIEmbeddings
    embeddings = OpenAIEmbeddings()
    pinecone.init(api_key = os.environ.get("PINECONE_API_KEY"), environment = os.environ.get("PINECONE_ENV"))
    
    if index_name in pinecone.list_indexes():
        print(f"Index {index_name} already exists, Loading embeddings...")
        vector_store = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)
        print(f"Done!")
        
    else:
        print(f"Creating index {index_name} and Loading embeddings...")
        pinecone.create_index(index_name, dimension=1536, metric="cosine")
        vector_store = Pinecone.from_documents(chunks, embeddings, index_name=index_name)
        print(f"Done!")
    
    return vector_store

def create_embeddings(chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(chunks, embeddings)
    return vector_store
    
def delete_pinecone_index(index_name='all'):
    import os
    import pinecone
    pinecone.init(api_key = os.environ.get("PINECONE_API_KEY"), environment = os.environ.get("PINECONE_ENV"))
    indexes = pinecone.list_indexes()
    if index_name == 'all':
        for index in indexes:
            print(f"Deleting index {index}")
            pinecone.delete_index(index)
            print(f"Done!")
    elif index_name in indexes:
        print(f"Deleting index {index_name}")
        pinecone.delete_index(index_name)
        print(f"Done!")
    else:
        print(f"Index {index_name} does not exist")

def qa_system(vector_store, q, k=3):
    from langchain.chains import RetrievalQA
    from langchain.chat_models import ChatOpenAI
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k}) 
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    result = qa.run(q) 
    return result

def ask_with_memory(vector_store, question, chat_history =[]):
    from langchain.chains import ConversationalRetrievalChain
    from langchain.chat_models import ChatOpenAI
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    qa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever)
    result = qa({"question": question, "chat_history": chat_history})
    return result, chat_history

def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']

if __name__ == "__main__":
    import os 
    from dotenv import load_dotenv
    load_dotenv()
    
    st.image('qa_bot.jpg')
    st.subheader("LLM Question-Answering Application 🤖🤖!!")
    with st.sidebar:
        api_key = st.text_input("OpenAI API Key: ", type="password")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        
        upload_file = st.file_uploader("Upload a file", type=["pdf", "docx", "txt"])
        chunk_size = st.number_input("Chunk size :", value=512, min_value=100, max_value=2048, on_change=clear_history)
        k = st.number_input('k :', min_value=1, max_value=20, value=3, on_change=clear_history)
        add_data = st.button("Add data", on_click=clear_history)
        
        if upload_file and add_data:
            with st.spinner("Embedding data..."):
                bytes_data = upload_file.read()
                file_name = os.path.join('./', upload_file.name)
                with open(file_name, 'wb') as f:
                    f.write(bytes_data)
                data = load_document(file_name)
                chunks = chunk_data(data, chunk_size=chunk_size)
                st.write(f"Chunks size : {chunk_size}, Chunks_created : {len(chunks)}")
                tokens,cost = caluculate_embedding_cost(chunks)
                st.write(f"Embedding cost : ${cost}")
                vector_store = create_embeddings(chunks)
                st.session_state.vs = vector_store
                st.success("Embeddings created Successfully!")
    q = st.text_input("Ask your question here about the document: ")
    if q:
        if "vs" in st.session_state:
            vector_store = st.session_state.vs
            result = qa_system(vector_store, q, k=k)
            st.text_area('Chatbot Answer: ', value = result)   
                
        st.divider()
        if 'history' not in st.session_state:
            st.session_state.history = ''
        value = f'Q: {q}\nA: {result}'
        st.session_state.history = f'{value} \n {"-" * 100} \n {st.session_state.history}'  
        h = st.session_state.history
        st.text_area(label = "Chat History", value = h, key="history", height = 400) 
        