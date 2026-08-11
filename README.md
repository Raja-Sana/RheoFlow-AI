# 🧪 RheoFlow AI — Multi-Sample Drilling Fluid Analyzer
RheoFlow AI is a high-end web application and AI assistant designed for drilling fluid engineers. It automates calculations, renders interactive Plotly visualizations, compares multiple mud formulations side-by-side, and utilizes Retrieval-Augmented Generation (RAG) to evaluate results against **API RP 13B-1** recommended practices.
## Project Structure
* rheoflow-ai/
* ├── .streamlit/
* │   └── config.toml            (Sleek Obsidian dark mode configuration)
* ├── standards_docs/
* │   └── api_standards.txt     (Local API RP 13B-1 reference documents)
* ├── faiss_index/              (Auto-generated local vector database index)
* ├── app.py                    (Main Streamlit web interface & RAG pipeline)
* ├── calculations.py           (Core math and validation logic)
* ├── visualizations.py         (Plotly interactive charting script)
* ├── create_vector_db.py       (Script to compile documents into FAISS)
* ├── persistent_sessions.db    (SQLite file storing persistent user sessions)
* └── requirements.txt          (Package dependencies for cloud deployment)
## 🚀 Technologies Used
* Frontend/UI: Streamlit, CSS, LaTeX.
* Math & Charts: Pandas, NumPy, Plotly (Interactive charts).
* AI & RAG: LangChain, FAISS (Vector DB), Sentence Transformers, Google GenAI SDK.
* Database: SQLite3, JSON.
