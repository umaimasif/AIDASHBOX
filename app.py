from typing import TypedDict, List, Any
import pandas as pd
import streamlit as st
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
import os

# ----------------------------
# STREAMLIT UI CONFIG
# ----------------------------
st.set_page_config(page_title="AI Business Intelligence", layout="wide")
st.title("🚀 AI-Powered BI Dashboard")

# Initialize Session State to cache AI results
if "ai_results" not in st.session_state:
    st.session_state.ai_results = None

# ----------------------------
# SIDEBAR: UPLOAD & FILTERING
# ----------------------------
with st.sidebar:
    st.header("📂 Data Source")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    st.header("⚙️ Dashboard Controls")
    refresh_ai = st.button("🔄 Re-run AI Analysis")

# ----------------------------
# LANGGRAPH LOGIC
# ----------------------------
class AnalysisState(TypedDict):
    raw_data: Any
    suggestions: str
    tables: dict
    uploaded_file: Any

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.1,
    api_key=st.secrets["GROQ_API_KEY"]
)

def load_data_node(state: AnalysisState):
    df = pd.read_csv(state["uploaded_file"])
    return {"raw_data": df}

def suggestion_node(state: AnalysisState):
    df = state["raw_data"]
    prompt = f"Analyze this dataset. Columns: {list(df.columns)}. Sample: {df.head(5).to_string()}. Give 3 actionable business insights."
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"suggestions": response.content}

def table_node(state: AnalysisState):
    df = state["raw_data"]
    summary = df.describe(include='all').to_dict()
    return {"tables": summary}

# Build Graph
builder = StateGraph(AnalysisState)
builder.add_node("load_data", load_data_node)
builder.add_node("suggestions", suggestion_node)
builder.add_node("tables", table_node)
builder.set_entry_point("load_data")
builder.add_edge("load_data", "suggestions")
builder.add_edge("load_data", "tables")
builder.add_edge("suggestions", END)
builder.add_edge("tables", END)
app = builder.compile()

# ----------------------------
# DASHBOARD EXECUTION
# ----------------------------
if uploaded_file:
    # 1. Load Data for Filtering (Immediate)
    df_raw = pd.read_csv(uploaded_file)
    
    # 2. Interactive Filters in Sidebar
    st.sidebar.subheader("🎯 Interactive Filters")
    cat_cols = df_raw.select_dtypes(include="object").columns.tolist()
    filters = {}
    
    filtered_df = df_raw.copy()
    for col in cat_cols[:3]: # Limit to first 3 categorical columns for UI cleanliness
        unique_vals = ["All"] + sorted(df_raw[col].unique().tolist())
        selection = st.sidebar.selectbox(f"Filter by {col}", unique_vals)
        if selection != "All":
            filtered_df = filtered_df[filtered_df[col] == selection]

    # 3. Run AI Analysis (Only if not cached or button clicked)
    if st.session_state.ai_results is None or refresh_ai:
        with st.spinner("Agents are thinking..."):
            # We pass the full raw data to the AI for global context
            st.session_state.ai_results = app.invoke({"uploaded_file": uploaded_file})

    # 4. DASHBOARD LAYOUT
    # Row 1: Key Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(filtered_df))
    col2.metric("Filtered Percentage", f"{(len(filtered_df)/len(df_raw)*100):.1f}%")
    col3.metric("Numeric Columns", len(df_raw.select_dtypes(include='number').columns))

    st.divider()

    # Row 2: Charts & AI
    left_side, right_side = st.columns([2, 1])

    with left_side:
        st.subheader("📊 Dynamic Visualizations")
        chart_col = st.selectbox("Select Column to Visualize", cat_cols)
        st.bar_chart(filtered_df[chart_col].value_counts())
        
        st.subheader("📄 Filtered Data Preview")
        st.dataframe(filtered_df.head(10), use_container_width=True)

    with right_side:
        st.subheader("🧠 AI Insights")
        st.info(st.session_state.ai_results["suggestions"])
        
        with st.expander("See Raw Statistical Summary"):
            st.write(st.session_state.ai_results["tables"])

else:
    st.info("👆 Please upload a CSV file in the sidebar to begin.")