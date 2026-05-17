import streamlit as st
import time
import os
from dotenv import load_dotenv

from pipelines.llm_only import run_llm_only
from pipelines.basic_rag import run_basic_rag
from pipelines.graph_rag import run_graph_rag

load_dotenv()

st.set_page_config(layout="wide", page_title="TigerGraph GraphRAG Hackathon")

st.title("GraphRAG Inference Benchmark")
st.markdown("Compare LLM-Only, Basic RAG, and GraphRAG pipelines side-by-side.")

query = st.text_input("Enter your query:", "What are the key risk factors mentioned in the latest reports?")

if st.button("Run Benchmarks"):
    if not query:
        st.warning("Please enter a query.")
    else:
        st.markdown("### Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("LLM-Only")
            with st.spinner("Running..."):
                start_time = time.time()
                result_1 = run_llm_only(query)
                latency_1 = time.time() - start_time
            
            st.markdown(f"**Latency:** {latency_1:.2f}s | **Tokens:** {result_1['tokens']} | **Cost:** ${result_1['cost']:.6f}")
            st.info(f"**Accuracy (Judge):** {result_1['accuracy']}")
            st.write(result_1['response'])

        with col2:
            st.subheader("Basic RAG")
            with st.spinner("Running..."):
                start_time = time.time()
                result_2 = run_basic_rag(query)
                latency_2 = time.time() - start_time
            
            st.markdown(f"**Latency:** {latency_2:.2f}s | **Tokens:** {result_2['tokens']} | **Cost:** ${result_2['cost']:.6f}")
            st.info(f"**Accuracy (Judge):** {result_2['accuracy']}")
            st.write(result_2['response'])

        with col3:
            st.subheader("GraphRAG")
            with st.spinner("Running..."):
                start_time = time.time()
                result_3 = run_graph_rag(query)
                latency_3 = time.time() - start_time
            
            st.markdown(f"**Latency:** {latency_3:.2f}s | **Tokens:** {result_3['tokens']} | **Cost:** ${result_3['cost']:.6f}")
            st.info(f"**Accuracy (Judge):** {result_3['accuracy']}")
            st.write(result_3['response'])
