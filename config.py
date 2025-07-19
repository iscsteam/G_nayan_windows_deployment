import os
from dotenv import load_dotenv
import mysql.connector
from mysql.connector import pooling, Error

# Load environment variables from a .env file
load_dotenv()

try:
    # --- Create a Connection Pool ---
    # Instead of a single connection, we create a pool. Your application
    # will borrow connections from this pool. This is more efficient and
    # handles dropped connections automatically.
    connection_pool = pooling.MySQLConnectionPool(
        pool_name="fastapi_pool",
        pool_size=5,  # Start with 5 connections, adjust as needed
        pool_reset_session=True,  # Ensures you get a clean session
        host=os.getenv("MYSQL_HOST", "mysql"),
        port=int(os.getenv("MYSQL_PORT", 3306)),
        user=os.getenv("MYSQL_USER", "iscs"),
        password=os.getenv("MYSQL_PASSWORD"),
        database=os.getenv("MYSQL_DB")
    )
    print("✅ MySQL Connection Pool created successfully")

except Error as e:
    print(f"❌ Error while creating MySQL connection pool: {e}")
    # If the pool fails to create, set it to None.
    # The get_connection function will handle this.
    connection_pool = None

def get_connection():
    """Gets a connection from the created pool."""
    if connection_pool is None:
        print("❌ Cannot get connection, the pool is not available.")
        return None
    
    try:
        # get_connection() from a pool is the correct way to borrow a connection.
        # It will wait if all connections are in use.
        connection = connection_pool.get_connection()
        if connection.is_connected():
            return connection
        return None
    except Error as e:
        print(f"❌ Error getting connection from pool: {e}")
        return None

def create_tables():
    """
    Creates all necessary tables using a connection from the pool.
    Ensures the connection is returned to the pool afterwards.
    """
    # Borrow a connection from the pool
    conn = get_connection()
    
    if conn is None:
        print("❌ Cannot create tables: Failed to get a valid DB connection from the pool.")
        return

    # Initialize cursor to None to prevent UnboundLocalError
    cursor = None
    try:
        cursor = conn.cursor()
        print("Executing CREATE TABLE statements...")
        # diabetic_retinopathy table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS diabetic_retinopathy (
            id INT AUTO_INCREMENT PRIMARY KEY, Patient_ID VARCHAR(100) NOT NULL,
            Predicted_Class VARCHAR(50), Stage VARCHAR(50), Confidence FLOAT,
            Explanation TEXT, Note TEXT, Risk_Factor TEXT, Review TEXT, Feedback TEXT,
            Doctors_Diagnosis TEXT, email_id VARCHAR(255), timestamp DATETIME
        )
        """)

        # api_logs table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS api_logs (
            id INT AUTO_INCREMENT PRIMARY KEY, timestamp DATETIME,
            level VARCHAR(10), message TEXT
        )
        """)

        # patient_form table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS patient_form (
            visit_id INT NOT NULL AUTO_INCREMENT PRIMARY KEY, name VARCHAR(255),
            patient_id INT, Duration_of_Diabetes INT, HbA1c_Level FLOAT,
            Blood_Pressure FLOAT, Fasting_Blood_Glucose FLOAT, BMI FLOAT,
            Cholesterol FLOAT, Age INT, Albuminuria FLOAT, Visual_Acuity VARCHAR(255),
            Date_of_registration DATE, Hospital_name VARCHAR(255), num_visits INT,
            mobile_number VARCHAR(15), gender VARCHAR(10)
        )
        """)

        conn.commit()
        print("✅ Tables checked/created successfully.")
    except Error as e:
        print(f"❌ Error creating tables: {e}")
    finally:
        # This 'finally' block ensures that resources are always released.
        if cursor:
            cursor.close()
        if conn:
            # For a pooled connection, .close() does NOT terminate it.
            # Instead, it returns the connection to the pool, ready for reuse.
            conn.close()
            print("✅ Connection returned to the pool after table creation.")