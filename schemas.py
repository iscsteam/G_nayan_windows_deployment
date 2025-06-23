from pydantic import BaseModel,Field
from datetime import date
from typing import Optional
from typing import Literal
class Form2(BaseModel):
    #visit_id: int
    #patient_id: int
    name:str
    Duration_of_Diabetes :int
    HbA1c_Level:float
    Blood_Pressure: float
    Fasting_Blood_Glucose: float
    BMI: float
    Cholesterol: float
    Age: int
    #Smoking_Status: str
    Albuminuria: float
    Visual_Acuity: str
    Date_of_registration: date
    mobile_number: str
    gender: str
    Hospital_name: str
    num_visits: int



class UpdatePatientForm(BaseModel):
    name: str
    patient_id: int
    visit_id: int
    Age: int
    gender: str
    mobile_number: str
    Duration_of_Diabetes: float
    HbA1c_Level: float
    Blood_Pressure: float
    Fasting_Blood_Glucose: float
    BMI: float
    Cholesterol: float
    Albuminuria: float
    Visual_Acuity: str
    Date_of_registration: date
    Hospital_name: str
    num_visits: int

class PatientPartialUpdate(BaseModel):
    name: Optional[str] = None
    Age: Optional[int] = None
    gender: Optional[str] = None
    mobile_number: Optional[str] = None
    Duration_of_Diabetes: Optional[float] = None
    HbA1c_Level: Optional[float] = None
    Blood_Pressure: Optional[float] = None
    Fasting_Blood_Glucose: Optional[float] = None
    BMI: Optional[float] = None
    Cholesterol: Optional[float] = None
    Albuminuria: Optional[float] = None
    Visual_Acuity: Optional[str] = None
    Date_of_registration: Optional[date] = None
    Hospital_name: Optional[str] = None
    num_visits: Optional[int] = None