#!/usr/bin/env python3
"""
SQL Database Setup Script
Creates SQLite database with sample data using SQLAlchemy
"""

import os
from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    insert,
    Date
)
from datetime import datetime
import random


def insert_rows_into_table(rows, table, engine):
    """Insert rows into a table"""
    stmt = insert(table).values(rows)
    with engine.begin() as connection:
        connection.execute(stmt)


def create_sql_database():
    """Create SQLite database with sample data"""
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Remove existing database file
    db_file = "data/doc.db"
    if os.path.exists(db_file):
        os.remove(db_file)
    
    # Create engine
    engine = create_engine(f"sqlite:///{db_file}")
    metadata_obj = MetaData()
    
    # Define claims table
    claims = Table(
        "claims",
        metadata_obj,
        Column("claim_num", String(50), primary_key=True),
        Column("doc_id", String(50)),
        Column("received_date", Date),
        Column("doc_type", String(50)),
    )
    
    # Define docs table  
    docs = Table(
        "docs",
        metadata_obj,
        Column("claim_num", String(50)),
        Column("doc_id", String(50), primary_key=True),
        Column("received_date", Date),
        Column("doc_type", String(50)),
    )
    
    # Create all tables
    metadata_obj.create_all(engine)
    
    # Generate sample data
    doc_types = ["police_report", "medical_bill", "medical_report", "attorney_demand", 
                 "invoice", "estimate", "receipt"]
    
    rows_claims = []
    rows_docs = []
    
    # Generate data for 2024 and 2025
    for year in [2024, 2025]:
        num_months = 12 if year == 2024 else 4  # 2025 partial year
        
        for month in range(1, num_months + 1):
            # Random number of claims per month (10-30)
            num_claims = random.randint(10, 30)
            
            for _ in range(num_claims):
                # Generate random claim number and doc ID
                claim_num = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=10))
                doc_id = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=15))
                
                # Random day in month
                day = random.randint(1, 28)
                received_date = datetime(year, month, day).date()
                
                # Random doc type
                doc_type = random.choice(doc_types)
                
                # Add to both tables
                rows_claims.append({
                    "claim_num": claim_num,
                    "doc_id": doc_id,
                    "received_date": received_date,
                    "doc_type": doc_type
                })
                
                rows_docs.append({
                    "claim_num": claim_num,
                    "doc_id": doc_id,
                    "received_date": received_date,
                    "doc_type": doc_type
                })
    
    # Insert data
    insert_rows_into_table(rows_claims, claims, engine)
    insert_rows_into_table(rows_docs, docs, engine)
    
    print(f"Database created successfully at {db_file}")
    print(f"Created {len(rows_claims)} claims and {len(rows_docs)} docs")


if __name__ == "__main__":
    create_sql_database()
