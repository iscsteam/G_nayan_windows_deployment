import os 
from dotenv import load_dotenv
load_dotenv()
import mysql.connector
from mysql.connector import Error
import os

def connection():
    try:
        connection = mysql.connector.connect(
            host=os.getenv("MYSQL_HOST", "mysql"),
            port=int(os.getenv("MYSQL_PORT", 3306)),
            user=os.getenv("MYSQL_USER", "iscs"),
            password=os.getenv("MYSQL_PASSWORD"),
            database=os.getenv("MYSQL_DB")
        )
        print("✅ Connected to MySQL")
        return connection
    except Error as e:
        print("❌ MySQL connection error:", e)
        return None
def create_tables(connection):
    if connection is None or not connection.is_connected():
        print("❌ Cannot create tables: No valid DB connection")
        return

    try:
        cursor = connection.cursor()
        # diabetic_retinopathy table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS diabetic_retinopathy (
            id INT AUTO_INCREMENT PRIMARY KEY,
            Patient_ID VARCHAR(100) NOT NULL,
            Predicted_Class VARCHAR(50),
            Stage VARCHAR(50),
            Confidence FLOAT,
            Explanation TEXT,
            Note TEXT,
            Risk_Factor TEXT,
            Review TEXT,
            Feedback TEXT,
            Doctors_Diagnosis TEXT,
            email_id VARCHAR(255),
            timestamp DATETIME
        )
        """)

        # api_logs table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS api_logs (
            id INT AUTO_INCREMENT PRIMARY KEY,
            timestamp DATETIME,
            level VARCHAR(10),
            message TEXT
        )
        """)

        # patient_form table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS patient_form (
            visit_id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(255),
            patient_id INT,
            Duration_of_Diabetes INT,
            HbA1c_Level FLOAT,
            Blood_Pressure FLOAT,
            Fasting_Blood_Glucose FLOAT,
            BMI FLOAT,
            Cholesterol FLOAT,
            Age INT,
            Albuminuria FLOAT,
            Visual_Acuity VARCHAR(255),
            Date_of_registration DATE,
            Hospital_name VARCHAR(255),
            num_visits INT,
            mobile_number VARCHAR(15),
            gender VARCHAR(10)
        )
        """)

        connection.commit()
        print("✅ Tables created or already exist")
    except Error as e:
        print(f"❌ Error creating tables: {e}")
    finally:
        cursor.close()

conn = connection()
if conn is not None:
    # Create tables if connection is successful
    create_tables(conn)
    print("✅ Tables created successfully")
else:
    print("❌ Failed to create tables due to connection error")



