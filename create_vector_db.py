# create_vector_db.py
# ============================================================
# Run this file ONCE from Anaconda Prompt:
#   python create_vector_db.py
#
# It will scan the standards_docs/ folder, load ALL .txt and
# .pdf files, split them into chunks, and save a FAISS vector
# database to the faiss_index/ folder.
# ============================================================

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

STANDARDS_FOLDER = "standards_docs"
FAISS_INDEX_PATH = "faiss_index"

print("=" * 50)
print("  RheoFlow AI - Vector Database Builder")
print("=" * 50)

# ---- Step 1: Load all files from standards_docs/ ----
documents = []
found_files = []

for filename in os.listdir(STANDARDS_FOLDER):
    filepath = os.path.join(STANDARDS_FOLDER, filename)
    try:
        if filename.endswith(".txt"):
            loader = TextLoader(filepath, encoding="utf-8")
            docs = loader.load()
            documents.extend(docs)
            found_files.append(filename)
            print(f"  ✅ Loaded TXT: {filename}")

        elif filename.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
            docs = loader.load()
            documents.extend(docs)
            found_files.append(filename)
            print(f"  ✅ Loaded PDF: {filename}")

    except Exception as e:
        print(f"  ❌ Failed to load {filename}: {e}")

if not documents:
    print("No documents found! Please add .txt or .pdf files to standards_docs/")
    exit()

print(f"\nTotal documents loaded: {len(found_files)}")

# ---- Step 2: Split into chunks ----
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,   # Each chunk = ~500 characters
    chunk_overlap=50  # 50 character overlap so context is not lost at edges
)
chunks = text_splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks.")

# ---- Step 3: Create vector embeddings (runs locally, no API needed) ----
print("\nCreating vector embeddings (may take 1-2 min on first run)...")
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# ---- Step 4: Build and save FAISS database ----
vector_db = FAISS.from_documents(chunks, embeddings)
vector_db.save_local(FAISS_INDEX_PATH)

print(f"\n✅ FAISS vector database saved to '{FAISS_INDEX_PATH}/' folder.")
print("You can now run your Streamlit app!")