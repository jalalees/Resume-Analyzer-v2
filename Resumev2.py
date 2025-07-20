# %%
import os
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings  
from langchain_core.runnables import RunnableLambda, RunnableMap
import google.generativeai as genai
from dotenv import load_dotenv
import shutil
import re
import streamlit as st

# Load environment variables
load_dotenv()

# %%
# Configure Google AI API 
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("Please set your GOOGLE_API_KEY in a .env file")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

# Setup embedding model
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# %%
# Create or load Chroma vector store
VECTOR_STORE_DIR = "chroma_store"
if os.path.exists(VECTOR_STORE_DIR):
    vectorstore = Chroma(persist_directory=VECTOR_STORE_DIR, embedding_function=embedding_model)
else:
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    vectorstore = Chroma(persist_directory=VECTOR_STORE_DIR, embedding_function=embedding_model)

# Define the function first
def extract_text_from_resume(file_name):
    file_extension = os.path.splitext(file_name)[1].lower()
    try:
        if file_extension == '.pdf':
            loader = PyPDFLoader(file_name)
        elif file_extension == '.docx':
            loader = Docx2txtLoader(file_name)
        elif file_extension == '.txt':
            loader = TextLoader(file_name)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

        documents = loader.load()
        text = " ".join([doc.page_content for doc in documents])
        return text
    except Exception as e:
        st.error(f"Error extracting text: {e}")
        return None

# Now use the function
uploaded_file = st.file_uploader("Upload your resume", type=["pdf", "docx", "txt"])
resume_text = None

if uploaded_file is not None:
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())
    file_name = uploaded_file.name
    st.write("File uploaded:", file_name)
    resume_text = extract_text_from_resume(file_name)
    if resume_text:
        st.write(resume_text)
else:
    file_name = None


# %%
# Store resume analysis in vector store
def store_resume_analysis(resume_text, analysis, doc_id):
    documents = split_text(analysis)
    vectorstore.add_documents(documents, ids=[f"{doc_id}_chunk_{i}" for i in range(len(documents))])
    vectorstore.persist()

# Extract percentage score from analysis text
def extract_suitability_score(text):
    match = re.search(r"Suitability Score: (\d{1,3})%", text)
    if match:
        return int(match.group(1))
    return None

# %%
# Job requirements upload
job_file = st.file_uploader("Upload job requirements", type=["pdf", "docx", "txt"], key="job_requirements")
job_requirements = None

if job_file is not None:
    with open(job_file.name, "wb") as f:
        f.write(job_file.getbuffer())
    job_file_name = job_file.name
    st.write("Job requirements file uploaded:", job_file_name)
    job_requirements = extract_text_from_resume(job_file_name)
    if job_requirements:
        st.write(job_requirements)
else:
    job_file_name = None



# %%
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2
)

prompt_template = PromptTemplate(
    input_variables=["job_requirements", "resume_text"],
    template="""
    You are an expert HR and recruitment specialist. Analyze the resume below against the job requirements.

    Job Requirements:
    {job_requirements}

    Resume:
    {resume_text}

    Provide a structured analysis of how well the resume matches the job requirements. 
    At the end, clearly state a "Suitability Score" as a percentage (0-100%) based on how well the resume aligns with the job.
    Format: Suitability Score: XX%
    """
)

chain = (
    RunnableMap({
        "job_requirements": lambda x: x["job_requirements"],
        "resume_text": lambda x: x["resume_text"]
    })
    | prompt_template
    | llm
    | StrOutputParser()
)

if resume_text and job_requirements:
    analysis = chain.invoke({
        "job_requirements": job_requirements,
        "resume_text": resume_text
    })
    st.write(analysis)
    suitability_score = extract_suitability_score(analysis)
    st.write(f"Suitability Score: {suitability_score}%")
else:
    st.info("Please upload both a resume and job requirements file to start the analysis.")


# %%

# %%
#suitability_score

def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.create_documents([text])


# %%
# Store resume analysis in vector store
def store_resume_analysis(resume_text, analysis, doc_id):
    documents = split_text(analysis)
    vectorstore.add_documents(documents, ids=[f"{doc_id}_chunk_{i}" for i in range(len(documents))])
