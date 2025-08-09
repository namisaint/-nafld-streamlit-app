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

# --- MongoDB Connection and Sidebar ---

import streamlit as st
from pymongo import MongoClient
import pandas as pd
from datetime import datetime

# --- MongoDB Connection Class ---
class MongoDBConnection:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.client = None
        self.db = None

    def connect(self, database_name: str) -> bool:
        try:
            # Simplest, most compatible form (PyMongo handles SRV + TLS automatically)
            self.client = MongoClient(self.connection_string)

            # Ping to confirm the connection
            self.client.admin.command("ping")

            # Set database handle
            self.db = self.client[database_name]
            return True
        except Exception as e:
            st.error("Failed to connect to MongoDB: " + str(e))
            return False

    def get_collections(self):
        if self.db is None:
            return []
        return self.db.list_collection_names()

    def query_collection(self, collection_name: str, query: dict = None, limit: int = 100) -> pd.DataFrame:
        if self.db is None:
            st.error("Not connected to database")
            return pd.DataFrame()
        try:
            collection = self.db[collection_name]
            cursor = collection.find(query or {}).limit(limit)
            data = list(cursor)
            return pd.DataFrame(data) if data else pd.DataFrame()
        except Exception as e:
            st.error("Error querying collection: " + str(e))
            return pd.DataFrame()

    def insert_document(self, collection_name: str, document: dict) -> bool:
        if self.db is None:
            st.error("Not connected to database")
            return False
        try:
            collection = self.db[collection_name]
            result = collection.insert_one(document)
            return result.inserted_id is not None
        except Exception as e:
            st.error("Error inserting document: " + str(e))
            return False


# --- Sidebar UI ---
st.sidebar.header("🍃 MongoDB Connection")

# Initialize session state
if "mongo_conn" not in st.session_state:
    st.session_state.mongo_conn = None
if "connected" not in st.session_state:
    st.session_state.connected = False

# Load defaults from secrets (your code expects these keys)
secrets_mongo = st.secrets.get("mongo", {})
default_connection_string = secrets_mongo.get("connection_string", "")
default_db_name = secrets_mongo.get("db_name", "nafld_db")

# Inputs
connection_string_input = st.sidebar.text_input(
    "Connection String",
    value=default_connection_string,
    type="password"
)
db_name_input = st.sidebar.text_input(
    "Database Name",
    value=default_db_name
)

# Connect button
if st.sidebar.button("Connect"):
    st.session_state.mongo_conn = MongoDBConnection(connection_string_input)
    ok = st.session_state.mongo_conn.connect(db_name_input)
    st.session_state.connected = ok
    if ok:
        st.sidebar.success("Connected to database: " + db_name_input)
    else:
        st.sidebar.error("Connection failed. Check Atlas Network Access and your secrets.")

# When connected, show collections and a test insert
if st.session_state.connected and st.session_state.mongo_conn is not None:
    try:
        cols = st.session_state.mongo_conn.get_collections()
        st.sidebar.write("Collections:")
        if cols:
            for c in cols:
                st.sidebar.write("- " + c)
        else:
            st.sidebar.info("No collections yet. Insert a document to create one.")
    except Exception as e:
        st.sidebar.error("Error listing collections: " + str(e))

    # Optional: Insert a test document to initialize a collection
    st.sidebar.subheader("Quick Test")
    test_collection = st.sidebar.text_input("Test collection name", value="predictions")
    if st.sidebar.button("Insert test doc"):
        doc = {
            "test": True,
            "inserted_at": datetime.utcnow().isoformat()
        }
        ok = st.session_state.mongo_conn.insert_document(test_collection, doc)
        if ok:
            st.sidebar.success("Inserted test doc into " + test_collection)
        else:
            st.sidebar.error("Insert failed")
