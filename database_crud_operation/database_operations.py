# Import the functions from your newly configured db_config.py
from config import get_connection, create_tables

def data_from_db(query: str):
    """
    Executes a query to fetch data using a connection from the pool.
    This function is now robust against connection drops.
    """
    conn = None
    cursor = None
    try:
        # 1. Borrow a connection from the pool for this specific task
        conn = get_connection()
        if conn:
            # 2. Create a cursor to execute the query
            # Using dictionary=True is helpful for APIs as it returns key-value pairs
            cursor = conn.cursor(dictionary=True)
            cursor.execute(query)
            result = cursor.fetchall()
            return result
        else:
            print("❌ data_from_db: Failed to get a connection from the pool.")
            return None
    except Exception as e:
        print(f"❌ Error executing query '{query}': {e}")
        return None
    finally:
        # 3. Always return the connection to the pool when done
        if cursor:
            cursor.close()
        if conn:
            conn.close() # Returns the connection to the pool

# --- Application Startup Script ---
# This part of your script should run once when your application starts.

if __name__ == "__main__":
    print("🚀 Initializing Application...")
    
    # 1. On startup, ensure the tables exist.
    create_tables()

    # 2. The application is now ready to handle requests.
    print("\n✅ Application is ready and waiting for requests.")
    
    # --- Example of how to use data_from_db in your code ---
    print("\n--- Running an example query ---")
    example_patients = data_from_db("SELECT name, age FROM patient_form LIMIT 2")
    if example_patients is not None:
        print("✅ Example query successful. Data:", example_patients)
    else:
        print("❌ Example query failed.")