import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

# Load DB config from .env
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT")
}


def run_postgres_query(query: str):
    """Run a SQL query and return results as clean text."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute(query)

        if cur.description:  # SELECT query
            rows = cur.fetchall()
            headers = [desc[0] for desc in cur.description]

            if not rows:
                result = "No results found."
            else:
                formatted = []
                for row in rows:
                    formatted.append(
                        ", ".join(f"{col}: {val}" for col, val in zip(headers, row))
                    )
                result = "\n".join(formatted)
        else:  # INSERT/UPDATE/DELETE
            conn.commit()
            result = "Query executed successfully."

        cur.close()
        conn.close()
        return result

    except Exception as e:
        return f"Postgres error: {str(e)}"