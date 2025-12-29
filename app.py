from typing import TypedDict, List, Any
import pandas as pd
import streamlit as st
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
import os

# ---------------------------------------------------------
# 1. STREAMLIT UI CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="AI BI Dashboard", layout="wide")
st.title("📊 AI Business Intelligence & Data Cleaner")

# Initialize Session State to cache AI results
if "ai_results" not in st.session_state:
    st.session_state.ai_results = None

# ---------------------------------------------------------
# 2. LANGGRAPH STATE DEFINITION
# ---------------------------------------------------------
class AnalysisState(TypedDict):
    raw_data: Any      # The uploaded dataframe
    clean_data: Any    # The processed dataframe
    suggestions: str   # AI-generated insights
    tables: dict       # Stats summary
    uploaded_file: Any # The file object

# ---------------------------------------------------------
# 3. THE NODES (THE WORKERS)
# ---------------------------------------------------------
def load_data_node(state: AnalysisState):
    file_obj = state["uploaded_file"]
    file_obj.seek(0) # FIX: Move pointer to start to avoid EmptyDataError
    df = pd.read_csv(file_obj)
    return {"raw_data": df}

def cleaning_node(state: AnalysisState):
    """Automatically cleans the dataset for better analysis."""
    df = state["raw_data"].copy()
    
    # Standardize column names (lowercase, no spaces)
    df.columns = [col.strip().replace(" ", "_").lower() for col in df.columns]
    
    # Fill numeric missing values with median
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    
    # Fill categorical missing values
    cat_cols = df.select_dtypes(include="object").columns
    df[cat_cols] = df[cat_cols].fillna("Unknown")
    
    return {"clean_data": df}

def suggestion_node(state: AnalysisState):
    """Asks the AI to analyze the cleaned data."""
    df = state["clean_data"]
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.1,
        api_key=st.secrets["GROQ_API_KEY"]
    )
    
    prompt = f"""
    Analyze this cleaned CSV data. 
    Columns: {list(df.columns)}
    Data Preview: {df.head(5).to_string()}
    
    Provide:
    1. A high-level summary of the data.
    2. Three actionable business insights.
    3. Potential anomalies or risks found.
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"suggestions": response.content}

def table_node(state: AnalysisState):
    """Generates statistical summaries."""
    df = state["clean_data"]
    summary = df.describe(include='all').to_dict()
    return {"tables": summary}

# ---------------------------------------------------------
# 4. BUILDING THE GRAPH
# ---------------------------------------------------------
builder = StateGraph(AnalysisState)

builder.add_node("load_data", load_data_node)
builder.add_node("clean_data", cleaning_node)
builder.add_node("suggestions", suggestion_node)
builder.add_node("tables", table_node)

builder.set_entry_point("load_data")
builder.add_edge("load_data", "clean_data")
builder.add_edge("clean_data", "suggestions")
builder.add_edge("clean_data", "tables")
builder.add_edge("suggestions", END)
builder.add_edge("tables", END)

app = builder.compile()

# ---------------------------------------------------------
# 5. STREAMLIT SIDEBAR & FILTERS
# ---------------------------------------------------------
with st.sidebar:
    st.header("📂 Data Source")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    st.header("⚙️ Controls")
    refresh_ai = st.button("🔄 Re-run AI Analysis")

# ---------------------------------------------------------
# 6. DASHBOARD EXECUTION FLOW
# ---------------------------------------------------------
if uploaded_file:
    # Initial load for UI filtering
    df_preview = pd.read_csv(uploaded_file)
    uploaded_file.seek(0) # Reset again for the graph

    # Sidebar Filter
    st.sidebar.subheader("🎯 Filters")
    cat_cols = df_preview.select_dtypes(include="object").columns.tolist()
    filtered_df = df_preview.copy()
    
    if cat_cols:
        target_col = cat_cols[0]
        choice = st.sidebar.selectbox(f"Filter by {target_col}", ["All"] + list(df_preview[target_col].unique()))
        if choice != "All":
            filtered_df = filtered_df[filtered_df[target_col] == choice]

    # Run AI Workflow
    if st.session_state.ai_results is None or refresh_ai:
        with st.spinner("🧠 AI Agents are cleaning and analyzing..."):
            st.session_state.ai_results = app.invoke({"uploaded_file": uploaded_file})

    # UI LAYOUT
    res = st.session_state.ai_results
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows Found", len(res["clean_data"]))
    col2.metric("Duplicates Removed", len(res["raw_data"]) - len(res["clean_data"]))
    col3.metric("Data Quality", "High (Cleaned)")

    st.divider()

    left, right = st.columns([2, 1])

    with left:
        st.subheader("📊 Visual Exploration")
        if cat_cols:
            st.bar_chart(filtered_df[cat_cols[0]].value_counts())
        
        st.subheader("🧹 Cleaned Data Preview")
        st.dataframe(res["clean_data"].head(20), use_container_width=True)

    with right:
        st.subheader("🧠 Agent Insights")
        st.markdown(res["suggestions"])
        
        with st.expander("🛠️ View Statistical Summary"):
            st.write(res["tables"])
            
        # Download Cleaned Data
        csv_data = res["clean_data"].to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Cleaned CSV", data=csv_data, file_name="cleaned_data.csv", mime="text/csv")

else:
    st.info("👋 Welcome! Please upload a CSV file in the sidebar to start the AI analysis.")
