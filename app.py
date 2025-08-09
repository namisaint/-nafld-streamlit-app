import streamlit as st
import pandas as pd
from pymongo import MongoClient
from datetime import datetime

st.set_page_config(page_title="Dissertation Model Predictor", page_icon="🤖", layout="wide")

# --- MongoDB Connection Class ---
class MongoDBConnection:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.client = None
        self.db = None

    def connect(self, database_name: str) -> bool:
        try:
            # Simplest form: PyMongo handles SRV and TLS automatically
            self.client = MongoClient(self.connection_string)

            # If you still hit TLS issues after IP allowlist + requirements, uncomment below:
            # import certifi
            # self.client = MongoClient(self.connection_string, tls=True, tlsCAFile=certifi.where())

            # Ping to confirm
            self.client.admin.command("ping")
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

# --- Sidebar: MongoDB Connection ---
st.sidebar.header("🍃 MongoDB Connection")

# Session state
if "mongo_conn" not in st.session_state:
    st.session_state.mongo_conn = None
if "connected" not in st.session_state:
    st.session_state.connected = False

# Load defaults from Streamlit secrets
secrets_mongo = st.secrets.get("mongo", {})
default_connection_string = secrets_mongo.get("connection_string", "")
default_db_name = secrets_mongo.get("db_name", "nafld_db")

# Inputs
connection_string_input = st.sidebar.text_input("Connection String", value=default_connection_string, type="password")
db_name_input = st.sidebar.text_input("Database Name", value=default_db_name)

# Connect
if st.sidebar.button("Connect"):
    st.session_state.mongo_conn = MongoDBConnection(connection_string_input)
    ok = st.session_state.mongo_conn.connect(db_name_input)
    st.session_state.connected = ok
    if ok:
        st.sidebar.success("Connected to database: " + db_name_input)
    else:
        st.sidebar.error("Connection failed. Check Atlas IP Access List and your secrets.")

# When connected: collections and test tools
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

# --- Main Page ---
st.title("Dissertation Model Predictor")
st.write("Use the sidebar to connect to MongoDB, then interact with your data below.")

if st.session_state.connected and st.session_state.mongo_conn is not None:
    st.subheader("Browse a Collection")
    collection_to_view = st.text_input("Collection name to view", value="predictions")
    limit_rows = st.slider("Rows to fetch", min_value=10, max_value=500, value=100, step=10)
    if st.button("Load Collection"):
        df = st.session_state.mongo_conn.query_collection(collection_to_view, limit=limit_rows)
        if df.empty:
            st.info("No data found or collection does not exist yet.")
        else:
            st.dataframe(df.head(50))
