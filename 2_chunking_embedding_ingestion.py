""" #################################################################################################################################################################
###############################   1.  IMPORTING MODULES AND INITIALIZING VARIABLES   ############################################################################
#################################################################################################################################################################

from dotenv import load_dotenv
import os
import json
import pandas as pd
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from uuid import uuid4
import shutil
import time


load_dotenv()

###############################   INITIALIZE EMBEDDINGS MODEL  #################################################################################################

embeddings = OllamaEmbeddings(
    model=os.getenv("EMBEDDING_MODEL"),
)

###############################   DELETE CHROMA DB IF EXISTS AND INITIALIZE   ##################################################################################

if os.path.exists(os.getenv("DATABASE_LOCATION")):
    shutil.rmtree(os.getenv("DATABASE_LOCATION"))

vector_store = Chroma(
    collection_name=os.getenv("COLLECTION_NAME"),
    embedding_function=embeddings,
    persist_directory=os.getenv("DATABASE_LOCATION"), 
)

###############################   INITIALIZE TEXT SPLITTER   ###################################################################################################

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)

#################################################################################################################################################################
###############################   2.  PROCESSING THE JSON RESPONSE LINE BY LINE   ###############################################################################
#################################################################################################################################################################

###############################   FUNCTION TO EXTRACT RESPONSE LINE BY LINE   ###################################################################################



def process_json_lines(file_path):
    ""Process each JSON line and extract relevant information.""
    extracted = []

    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            extracted.append(obj)

    return extracted

file_content = process_json_lines(os.getenv("DATASET_STORAGE_FOLDER")+"data.txt")


#################################################################################################################################################################
###############################   3.  CHUNKING, EMBEDDING AND INGESTION   #######################################################################################
##################################################################################################################################################################

for line in file_content:

    print(line['url'])

    texts = []
    texts = text_splitter.create_documents([line['raw_text']],metadatas=[{"source":line['url'], "title":line['title']}])

    uuids = [str(uuid4()) for _ in range(len(texts))]

    vector_store.add_documents(documents=texts, ids=uuids)


    if len(line) < 10:
        break """


##########################################################################################################
# 1. IMPORTING MODULES AND INITIALIZING VARIABLES
##########################################################################################################

from dotenv import load_dotenv
import os
import glob
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from uuid import uuid4
import shutil

load_dotenv()

##########################################################################################################
# INITIALIZE EMBEDDINGS MODEL
##########################################################################################################

embeddings = OllamaEmbeddings(
    model=os.getenv("EMBEDDING_MODEL", "llama3.2:1b")  # Default to llama2 if not in .env
)

##########################################################################################################
# DELETE CHROMA DB IF EXISTS AND INITIALIZE
##########################################################################################################

db_path = os.getenv("DATABASE_LOCATION", "vector_db")

if os.path.exists(db_path):
    shutil.rmtree(db_path)

vector_store = Chroma(
    collection_name=os.getenv("COLLECTION_NAME", "IT_Testing"),
    embedding_function=embeddings,
    persist_directory=db_path,
)

##########################################################################################################
# INITIALIZE TEXT SPLITTER
##########################################################################################################

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)

##########################################################################################################
# 2. LOAD DOCUMENTS FROM LOCAL FOLDER
##########################################################################################################

from pypdf import PdfReader, PdfWriter
import tempfile

def repaired_copy(src_path: str) -> str | None:
    """Try to rebuild a damaged PDF into a temporary clean copy."""
    try:
        reader = PdfReader(src_path, strict=False)
        writer = PdfWriter()
        for page in reader.pages:
            writer.add_page(page)
        tmp = tempfile.mkstemp(suffix=".pdf")[1]
        with open(tmp, "wb") as f:
            writer.write(f)
        return tmp
    except Exception as e:
        print(f"⚠️  Repair failed for {src_path}: {e}")
        return None


docs_path = os.getenv("DATASET_STORAGE_FOLDER", "Data/MyDocs")
if not os.path.isdir(docs_path):
    raise FileNotFoundError(f"❌ Folder not found: {docs_path}")

pdf_files = glob.glob(os.path.join(docs_path, "*.pdf"))
if not pdf_files:
    raise FileNotFoundError(f"❌ No PDF files found in: {docs_path}")

print(f"📂 Found {len(pdf_files)} PDF(s) in {docs_path}")
all_docs = []

for pdf_file in pdf_files:
    print(f"📄 Loading: {pdf_file}")
    try:
        loader = PyPDFLoader(pdf_file)
        pages = loader.load()
    except Exception:
        repaired = repaired_copy(pdf_file)
        if repaired:
            print(f"🔧 Using repaired copy for: {pdf_file}")
            loader = PyPDFLoader(repaired)
            pages = loader.load()
            os.remove(repaired)
        else:
            print(f"❌ Skipping unrecoverable file: {pdf_file}")
            continue
    all_docs.extend(pages)

print(f"✅ Total pages loaded: {len(all_docs)}")


##########################################################################################################
# 3. CHUNKING, EMBEDDING AND INGESTION
##########################################################################################################

chunks = text_splitter.split_documents(all_docs)
uuids = [str(uuid4()) for _ in range(len(chunks))]

vector_store.add_documents(documents=chunks, ids=uuids)

print("✅ Documents processed and stored successfully!")