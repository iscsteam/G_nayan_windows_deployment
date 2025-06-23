# This is back end for g-nayan RUNNING IN FAST API 
docker-compose up --build
# To build image in a best way and quick way we can use COMPOSE_BAKE=true 
COMPOSE_BAKE=true docker-compose up --build 

# To connect mysql image
docker exec -it mysql-container mysql -u root -p
with password 


# for patient record create table 
CREATE TABLE diabetic_retinopathy (
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
);



# create a database wit this cofig for logging
CREATE TABLE api_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp DATETIME,
    level VARCHAR(10),
    message TEXT
);
# create patient_form
CREATE TABLE patient_form (
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
);

# THE pipeline is running in only in cpu but nor GPU
    rate(container_cpu_usage_seconds_total{container="fastapi"}[1m]) * 100 number of api hits

    sum(http_requests_total{handler="/get_data_from_api_logs"})

