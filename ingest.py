import json
import psycopg2
import os
from dotenv import load_dotenv
load_dotenv()

DB_CONFIG = {
    "dbname": os.getenv("PGDATABASE"),
    "user": os.getenv("PGUSER"),
    "password": os.getenv("PGPASSWORD"),
    "host": os.getenv("PGHOST"),
    "port": os.getenv("PGPORT"),
}

conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

with open("data (4).json", "r", encoding="utf-8") as f:
    records = json.load(f)  # This should now be a list

insert_query = """
INSERT INTO kb_table (
    "Customer ID", "Customer Name", "Email", "Phone Number", "country", "Sales_Rep Name", "Qualifying Lead", "Lead qualification Reason", "Ad Lead Qualification Reason",  "Ad Lead", 
    "Package of Customer Interest","Package Purchased","Postal Code", "Pain points Objections Outcomes", "Tags from GHL","Investment Level", "Investable Assets","Engagement Level", "Risk Profile", "Persona Type", "Customer Goals filled in forms by Customer",
    "Investment Capacity", "monthly passive income goal of customer","Date Of Funding","Account status","TF Form Submission Date","Credit Score","Date of Lead Creation","Total Amount Funded","Amount Received","Amount"
) VALUES (
    %(Customer ID)s, %(Customer Name)s, %(Email)s, %(Phone Number)s, %(country)s, %(Sales_Rep Name)s, %(Qualifying Lead)s, %(Lead qualification Reason)s, %(Ad Lead Qualification Reason)s,
    %(Ad Lead)s, %(Package of Customer Interest)s,%(Package Purchased)s, %(Postal Code)s, %(Pain points Objections Outcomes)s, %(Tags from GHL)s, %(Investment Level)s, %(Investable Assets)s, %(Engagement Level)s, %(Risk Profile)s, %(Persona Type)s, %(Customer Goals filled in forms by Customer)s,
    %(Investment Capacity)s, %(monthly passive income goal of customer)s,%(Date Of Funding)s,%(Account status)s,%(TF Form Submission Date)s,%(Credit Score)s,%(Date of Lead Creation)s,%(Total Amount Funded)s,%(Amount Received)s,%(Amount)s
)
"""

for rec in records:
    data = {
        "Customer ID": rec.get("Customer ID"),
        "Customer Name": rec.get("Customer Name"),
        "Email": rec.get("Email"),
        "Phone Number": rec.get("Phone Number"),
        "country": rec.get("country"),
        "Sales_Rep Name": rec.get("Sales_Rep Name"),
        "Qualifying Lead": rec.get("Qualifying Lead") == "True",
        "Lead qualification Reason": rec.get("Lead qualification Reason"),
        "Ad Lead Qualification Reason": rec.get("Ad Lead Qualification Reason"),
        "Package Purchased": rec.get("Package Purchased"),
        "Postal Code": rec.get("Postal Code"),
        "Ad Lead": rec.get("Ad Lead") == "True",
        "Package of Customer Interest": rec.get("Package of Customer Interest"),
        "Pain points Objections Outcomes": rec.get("Pain points Objections Outcomes"),
        "Tags from GHL": rec.get("Tags from GHL"),
        "Investment Level": rec.get("Investment Level"),
        "Investable Assets": rec.get("Investable Assets"),
        "Engagement Level": rec.get("Engagement Level"),
        "Risk Profile": rec.get("Risk Profile"),
        "Persona Type": rec.get("Persona Type"),
        "Customer Goals filled in forms by Customer": rec.get("Customer Goals filled in forms by Customer"),
        "Investment Capacity": rec.get("Investment Capacity"),
        "monthly passive income goal of customer": rec.get("monthly passive income goal of customer"),
        "Date Of Funding": rec.get("Date Of Funding"),
        "Account status":rec.get("Account status"),
        "TF Form Submission Date": rec.get("TF Form Submission Date"),
        "Credit Score": rec.get("Credit Score"),
        "Date of Lead Creation": rec.get("Date of Lead Creation"),
        "Total Amount Funded": rec.get("Total Amount Funded"),
        "Amount Received": rec.get("Amount Received"),
        "Amount": rec.get("Amount")
    }
    cur.execute(insert_query, data)

conn.commit()
cur.close()
conn.close()

print("✅ All JSON records inserted into PostgreSQL successfully!")