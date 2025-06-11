import sqlite3
from sqlite3 import Error

# This script to create a SQLite database, define a table schema, insert sample data, and query the data.

# Function to create a database connection to a SQLite database
def create_connection(db_file):    
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print(f"Connected to SQLite version {sqlite3.version}")
        return conn
    except Error as e:
        print(e)
    return conn

# Function to create a table using the provided SQL statement
def create_table(conn, create_table_sql):
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
        print("Table created successfully")
    except Error as e:
        print(e)

# Function to insert sample data into the employees table
def insert_sample_data(conn):
    employees = [
        (1, 'Ruchit', 'Sales', 50000.0, '2020-01-15'),
        (2, 'Ramya', 'Marketing', 60000.0, '2019-05-22'),
        (3, 'Satish', 'IT', 75000.0, '2018-11-03'),
        (4, 'Bhavishya', 'HR', 55000.0, '2021-02-28'),
        (5, 'Teja', 'Finance', 80000.0, '2017-07-10')
    ]
    
    try:
        c = conn.cursor()
        c.executemany("INSERT INTO employees VALUES (?, ?, ?, ?, ?)", employees)
        conn.commit()
        print(f"Inserted {len(employees)} records")
    except Error as e:
        print(e)

def main():
    database = "company.db"
    
    # SQL statement for creating the employees table
    sql_create_employees_table = """CREATE TABLE IF NOT EXISTS employees (
                                    id integer PRIMARY KEY,
                                    name text NOT NULL,
                                    department text,
                                    salary real,
                                    hire_date text
                                );"""
    
    # Create a database connection
    conn = create_connection(database)
    
    if conn is not None:
        # Create employees table
        create_table(conn, sql_create_employees_table)
        
        # Insert sample data
        insert_sample_data(conn)
        
        # Verify data was inserted
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM employees")
        rows = cursor.fetchall()
        
        print("\nCurrent data in employees table:")
        for row in rows:
            print(row)
        
        # Close the connection
        conn.close()
    else:
        print("Error! Cannot create the database connection.")

if __name__ == '__main__':
    main()