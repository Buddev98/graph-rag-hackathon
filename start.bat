@echo off
echo ========================================================
echo TigerGraph GraphRAG Hackathon - Startup Script
echo ========================================================
echo.

echo [1/4] Installing dependencies (this may take a few minutes)...
call venv\Scripts\activate.bat
pip install -r requirements.txt

if not exist venv\Scripts\streamlit.exe (
    echo [ERROR] Failed to install streamlit. Please check your network connection.
    pause
    exit /b
)

if not exist data\dataset.json (
    echo.
    echo [2/4] Downloading Wikipedia Dataset Snippet ^(2M+ Tokens^)...
    python data_ingestion\prepare_dataset.py
) else (
    echo.
    echo [2/4] Dataset already prepared.
)

if not exist data\chroma_db (
    echo.
    echo [3/4] Ingesting dataset into ChromaDB for Basic RAG...
    python data_ingestion\chroma_ingest.py
) else (
    echo.
    echo [3/4] ChromaDB already ingested.
)

echo.
echo [4/4] Starting Streamlit Dashboard...
streamlit run app.py
