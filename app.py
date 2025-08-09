import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import plotly.express as px
import certifi
from typing import Optional, Dict, Any, List

# --- App Configuration ---
st.set_page_config(
    page_title="Dissertation Model Predictor",
    page_icon="🤖",
    layout="wide"
)

# --- MongoDB Connection Class ---
class MongoDBConnection:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.client = None
        self.db = None
        
    def connect(self, database_name: str) -> bool:
        try:
            self.client = MongoClient(self.connection_string, server_api=ServerApi('1'), tlsCAFile=certifi.where())
            self.client.admin.command('ping')
            self.db = self.client[database_name]
            return True
        except Exception as e:
            st.error(f"Failed to connect to MongoDB: {str(e)}")
            return False
    
    def get_collections(self) -> List[str]:
        if self.db is None:
            return []
        return self.db.list_collection_names()
    
    def query_collection(self, collection_name: str, query: Dict[str, Any] = None, limit: int = 100) -> pd.DataFrame:
        if self.db is None:
            st.error("Not connected to database")
            return pd.DataFrame()
        
        try:
            collection = self.db[collection_name]
            cursor = collection.find(query or {}).limit(limit)
            data = list(cursor)
            return pd.DataFrame(data) if data else pd.DataFrame()
        except Exception as e:
            st.error(f"Error querying collection: {str(e)}")
            return pd.DataFrame()
    
    def insert_document(self, collection_name: str, document: Dict[str, Any]) -> bool:
        if self.db is None:
            st.error("Not connected to database")
            return False
        
        try:
            collection = self.db[collection_name]
            result = collection.insert_one(document)
            return result.inserted_id is not None
        except Exception as e:
            st.error(f"Error inserting document: {str(e)}")
            return False

# --- MongoDB Connection Sidebar ---
st.sidebar.header("🍃 MongoDB Connection")

# Initialize session state
if 'mongo_conn' not in st.session_state:
    st.session_state.mongo_conn = None
if 'connected' not in st.session_state:
    st.session_state.connected = False

# Try to use secrets first, but allow manual input as fallback
try:
    default_connection_string = st.secrets["mongo"]["connection_string"]
    default_db_name = st.secrets["mongo"]["db_name"]
    st.sidebar.info("✅ Using connection from secrets")
except:
    default_connection_string = ""
    default_db_name = ""
    st.sidebar.info("ℹ️ Enter connection details manually")

# Connection inputs
connection_string = st.sidebar.text_input(
    "MongoDB Connection String",
    value=default_connection_string,
    placeholder="mongodb+srv://username:password@cluster.mongodb.net/",
    type="password",
    help="Your full MongoDB connection string with password"
)

database_name = st.sidebar.text_input(
    "Database Name",
    value=default_db_name,
    placeholder="your_database_name",
    help="Name of your MongoDB database"
)

# Connect button
if st.sidebar.button("Connect to MongoDB", type="primary"):
    if connection_string and database_name:
        mongo_conn = MongoDBConnection(connection_string)
        if mongo_conn.connect(database_name):
            st.session_state.mongo_conn = mongo_conn
            st.session_state.connected = True
            st.sidebar.success("✅ Connected successfully!")
        else:
            st.session_state.connected = False
    else:
        st.sidebar.error("Please provide both connection string and database name")

# Show connection status
if st.session_state.connected:
    st.sidebar.success("🟢 MongoDB Connected")
    if st.sidebar.button("Disconnect"):
        st.session_state.connected = False
        st.session_state.mongo_conn = None
        st.rerun()
else:
    st.sidebar.error("🔴 Not Connected")

# --- Main App Title ---
st.title("🤖 Dissertation Model Predictor")

# --- MongoDB Data Section ---
if st.session_state.connected and st.session_state.mongo_conn:
    st.success("Connected to MongoDB!")
    
    # Get collections
    collections = st.session_state.mongo_conn.get_collections()
    
    if collections:
        with st.expander("📊 MongoDB Data Explorer", expanded=False):
            selected_collection = st.selectbox(
                "Select a collection:",
                collections
            )
            
            # Create tabs for different operations
            tab1, tab2, tab3 = st.tabs(["📋 View Data", "🔍 Query Data", "➕ Insert Data"])
            
            with tab1:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    limit = st.number_input("Number of documents to fetch:", 
                                          min_value=1, max_value=1000, value=100)
                
                with col2:
                    if st.button("Load Data", type="primary"):
                        df = st.session_state.mongo_conn.query_collection(selected_collection, limit=limit)
                        if not df.empty:
                            st.session_state.current_data = df
                            st.success(f"Loaded {len(df)} documents from {selected_collection}")
                
                # Display data if available
                if 'current_data' in st.session_state and not st.session_state.current_data.empty:
                    st.subheader(f"Data from {selected_collection}")
                    st.dataframe(st.session_state.current_data, use_container_width=True)
            
            with tab2:
                st.subheader("🔍 Custom Query")
                query_input = st.text_area(
                    "Enter MongoDB query (JSON format):",
                    placeholder='{"field": "value"}',
                    help="Leave empty for no filter"
                )
                
                if st.button("Execute Query"):
                    try:
                        import json
                        query = json.loads(query_input) if query_input.strip() else {}
                        df = st.session_state.mongo_conn.query_collection(selected_collection, query, limit=100)
                        if not df.empty:
                            st.success(f"Query returned {len(df)} documents")
                            st.dataframe(df, use_container_width=True)
                        else:
                            st.info("No documents found matching the query")
                    except json.JSONDecodeError:
                        st.error("Invalid JSON format in query")
            
            with tab3:
                st.subheader("➕ Insert New Document")
                document_input = st.text_area(
                    "Enter document (JSON format):",
                    placeholder='{"name": "John", "age": 30, "city": "New York"}',
                    help="Enter a valid JSON document to insert"
                )
                
                if st.button("Insert Document"):
                    try:
                        import json
                        document = json.loads(document_input)
                        if st.session_state.mongo_conn.insert_document(selected_collection, document):
                            st.success("Document inserted successfully!")
                        else:
                            st.error("Failed to insert document")
                    except json.JSONDecodeError:
                        st.error("Invalid JSON format in document")

# --- Your Original App Content Goes Here ---
# Add your model prediction functionality below this line

st.header("🔮 Model Predictions")
st.info("Add your model prediction functionality here!")

# Example placeholder for your model functionality
with st.expander("📈 Model Interface", expanded=True):
    st.write("This is where your dissertation model predictor will go.")
    st.write("You can add:")
    st.write("- Input fields for model features")
    st.write("- Model loading and prediction logic") 
    st.write("- Results visualization")
    st.write("- Data export functionality")

# --- Connection Help ---
if not st.session_state.connected:
    with st.expander("ℹ️ MongoDB Connection Help"):
        st.markdown("""
        ### Your Connection String Format:
        ```
        mongodb+srv://streamlit2025:YOUR_PASSWORD@nafld-app.cvmvo5c.mongodb.net/?retryWrites=true&w=majority&appName=NAFLD-APP
        ```
        
        **Replace YOUR_PASSWORD with your actual MongoDB password**
        
        ### Steps:
        1. Get your password from MongoDB Atlas
        2. Replace `<db_password>` in your connection string with your actual password
        3. Enter the full connection string in the sidebar
        4. Enter your database name
        5. Click "Connect to MongoDB"
        
        ### Need Help?
        - Make sure your MongoDB user has read/write permissions
        - Check that your IP address is whitelisted in MongoDB Atlas
        - Verify your connection string format is correct
        """)
