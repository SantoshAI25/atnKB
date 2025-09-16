import os
import psycopg2
import json
from flask import Flask, render_template, request, session, jsonify
from flask_session import Session
from dotenv import load_dotenv

# === LangSmith ===
from langsmith import Client
from langsmith.run_helpers import traceable

from postgres_utils import run_postgres_query
from pinecone_utils import search_with_filters
from llm_utils import llm, safe_tool_output

# ========================
# 1. ENV + APP SETUP
# ========================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# LangSmith setup
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGSMITH_TRACING")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT")

client = Client()

# PostgreSQL connection details from .env
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT")
}

# Test connection at startup
try:
    conn = psycopg2.connect(**DB_CONFIG)
    conn.close()
except Exception as e:
    raise RuntimeError(f"Database connection failed: {e}")

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    raise RuntimeError("Please set OPENAI_API_KEY and PINECONE_API_KEY in .env")

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = os.getenv("FLASK_SECRET_KEY", "supersecret")

# ---- Server-side sessions ----
app.config["SESSION_TYPE"] = "filesystem"  # or "redis", "sqlalchemy"
app.config["SESSION_FILE_DIR"] = "./.flask_session"
app.config["SESSION_PERMANENT"] = False
Session(app)


def ensure_session():
    """Make sure session has default state and prevent bloat."""
    if "messages" not in session:
        session["messages"] = []
    if "last_customer" not in session:
        session["last_customer"] = None

    # keep session light (last 20 messages)
    if len(session["messages"]) > 20:
        session["messages"] = session["messages"][-20:]


# ========================
# 2. TOOL DEFINITIONS
# ========================
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "postgres_tool",
            "description": """
Run SQL queries on the atn_table.

Schema:
Table: atn_table
Important Columns:

"Customer ID" TEXT,
"Customer Name" TEXT,
"Email" TEXT,
"Phone Number" TEXT,
"country" TEXT,
"Sales_Rep Name" TEXT,
"Qualifying Lead" TEXT,
"Package Purchased" TEXT,
"Postal Code" TEXT,
"Ad Lead" TEXT,
"Package of Customer Interest" TEXT,
"Investment Level" TEXT,
"Investable Assets" TEXT,
"monthly passive income goal of customer" TEXT,
"Investment Capacity" TEXT,
"Engagement Level" TEXT,
"Amount" TEXT,
"Amount Received" TEXT,
"Total Amount Funded" TEXT,
"Account status" TEXT,
"Date of Lead Creation" TEXT,
"Meeting Date" TEXT,
"Customer Goals filled in TF by Customer" TEXT,
"Credit Score" TEXT,
"Date Of Funding" TEXT,
"Form Submission Date" TEXT,
"TF Ending" TEXT

Rules for SQL:
- Always wrap column names in double quotes (" ") because they contain spaces.
- Table name is always atn_table.
- If the user query about a specific column and there are missing entries, ignore  the missing entries and use the rows with data available in that column.
""",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A valid SQL query using atn_table."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "pinecone_tool",
            "description": "Search semantic knowledge base for customer insights, preferences, or unstructured data like pain points, differences.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search string or natural language query."
                    }
                },
                "required": ["query"]
            }
        }
    }
]


# ========================
# 3. ROUTES
# ========================
@app.route("/", methods=["GET"])
def index():
    ensure_session()
    return render_template("chat.html", messages=session["messages"])


@traceable(name="chat_pipeline")
def run_chat_pipeline(conversation, tools):
    """Traceable wrapper for LLM call"""
    return llm.invoke(conversation, tools=tools)


@app.route("/chat", methods=["POST"])
def chat():
    ensure_session()
    data = request.get_json(silent=True) or request.form
    user_query = (data.get("message") or "").strip()
    if not user_query:
        return jsonify({"ok": False, "error": "Empty message"}), 400

    msgs = session["messages"]
    msgs.append({"role": "user", "content": user_query})

    # ========== STEP 1: Call LLM with tools ==========
    conversation = [
        {"role": "system", "content": """
        You are a friendly yet sharp customer insights assistant for the ATN Unlimited team.  
        Your purpose is to help the team understand customers better and surface actionable insights.  

        You have access to two tools:  
        - postgres_tool → structured data (IDs, counts, filtering by attributes).  
        - pinecone_tool → semantic queries (pain points, goals, unstructured notes).  

        Rules:
        1. Use postgres_tool for structured queries.  
        2. Use pinecone_tool for semantic/unstructured queries.  
        3. Provide clear, natural answers based on tool results.  
        4. If a query fails, retry with a simpler approach.  
        5. Keep responses concise, human-friendly, and insight-driven.  
        6. End every response with 1–2 smart follow-up questions.  
        """}
    ] + msgs[-10:]

    response = run_chat_pipeline(conversation, TOOLS)

    answer = ""
    tool_results = {}

    # ========== STEP 2: Handle tool calls ==========
    if getattr(response, "tool_calls", None):
        followup_conversation = conversation + [response]

        for call in response.tool_calls:
            fn_name = call.function.name
            args = json.loads(call.function.arguments)

            if fn_name == "postgres_tool":
                result = run_postgres_query(args["query"])
                tool_results[call.id] = result

            elif fn_name == "pinecone_tool":
                results, _, _, _ = search_with_filters(args["query"])
                result_text = "\n".join(
                    [doc.page_content for doc in results]
                ) if results else "No results found."
                tool_results[call.id] = safe_tool_output(result_text)

        # Add tool outputs back into conversation
        for tool_id, result in tool_results.items():
            followup_conversation.append({
                "role": "tool",
                "tool_call_id": tool_id,
                "content": result
            })

        # Final LLM response
        final_response = run_chat_pipeline(followup_conversation, tools=None)
        answer = final_response.content

    else:
        answer = response.content

    # Save assistant reply
    msgs.append({"role": "assistant", "content": answer})
    session["messages"] = msgs

    return jsonify({"ok": True, "reply": answer})


# ------------- run -------------
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5000")),
        debug=True,
        use_reloader=False  # important for Windows to avoid socket error
    )
