#from .database import Base
from sqlalchemy import Column, Integer, String,Float,Date

from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class PatientForm(Base):
    __tablename__ = "patient"
    name = Column(String(255))
    visit_id = Column(Integer, primary_key=True, index=True, autoincrement=True)  # Unique row ID
    patient_id = Column(Integer)  # Your custom ID, allows duplicates
   
    
    
    Duration_of_Diabetes = Column(Integer)
    HbA1c_Level = Column(Float)
    Blood_Pressure = Column(Float)
    Fasting_Blood_Glucose = Column(Float)
    BMI = Column(Float)
    Cholesterol = Column(Float)
    Age = Column(Integer)
    #Smoking_Status = Column(String(255))
    Albuminuria = Column(Float)
    Visual_Acuity = Column(String(255))
    Date_of_registration = Column(Date)
    Hospital_name = Column(String(255))
    num_visits = Column(Integer)
    mobile_number = Column(String(15))
    gender = Column(String(10))
