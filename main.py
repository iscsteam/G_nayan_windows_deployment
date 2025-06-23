import os
import json
import io
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile, HTTPException,Request,status,Depends
from fastapi.responses import JSONResponse
from torchvision import transforms, models
from PIL import Image
import cv2
import time
import uvicorn
import requests
import logger
from datetime import datetime, timedelta
from fastapi.middleware.cors import CORSMiddleware
import requests
from io import BytesIO
from PIL import Image
import yaml  
import configparser
from typing import Dict
from starlette.middleware.sessions import SessionMiddleware
from starlette.requests import Request
from dotenv import load_dotenv
import warnings
import logger
from efficientnet_pytorch import EfficientNet
warnings.filterwarnings("ignore", category=DeprecationWarning)
from datetime import datetime
import pytz
from utlities import explanation_labels,CATEGORY_MAPPING
from minio import Minio
from minio.error import S3Error
from datetime import timedelta
from config import connection
import logger
from logger import get_logger
from database_crud_operation.database_operations import data_from_db
from prometheus_fastapi_instrumentator import Instrumentator
from schemas import Form2,PatientPartialUpdate
from Models import PatientForm


#mysql connection
connection=connection()

app = FastAPI(title="FastAPI for DR", description="Diabetic Retinopathy Detection API", version="1.0.0")



# Adding session middleware
app.add_middleware(SessionMiddleware, secret_key="444555")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# logger.basicConfig(
#     filename='api_hits.log',
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     level=logger.INFO
# )
logger = get_logger()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Initialize MinIO client
minio_client = Minio(
    "minio:9000",  # Replace with your MinIO host
    access_key="minioadmin",
    secret_key="minioadmin123",
    secure=False  # Set to True if you're using HTTPS
)

BUCKET_NAME = "fundus-images"
def upload_image_to_minio(patient_id,image_file, image_name): #Name,age,gender,
    # Ensure the bucket exists
    found = minio_client.bucket_exists(BUCKET_NAME)
    if not found:
        minio_client.make_bucket(BUCKET_NAME)

    # Object path inside the bucket
    object_name = f"{patient_id}/{image_name}"

    # Upload the file (image_file should be a file-like object or bytes)
    minio_client.put_object(
        bucket_name=BUCKET_NAME,
        object_name=object_name,
        data=BytesIO(image_file),
        length=len(image_file)        
    )

    # Generate a presigned URL (valid for 1 hour)
    image_url = minio_client.presigned_get_object(
        bucket_name=BUCKET_NAME,
        object_name=object_name,
        expires=timedelta(hours=1)
    )

    return image_url


# model-1 for fundus or non funding images classifications 
class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
# Load the saved model
model_1 = BinaryClassifier()
model_1.load_state_dict(torch.load("model_paths/ResNet18.pth", map_location=torch.device('cpu'), weights_only=True))
#
# Define the transformation
transform_model_1 = transforms.Compose([
    transforms.ToTensor(),
   
])
IMG_SIZE_single=224
def predict_single_image(model, image_path, transform):
    try:
        # Fetch the image from the URL
        response = requests.get(image_path)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch the image. Status code: {response.status_code}")

        image = Image.open(BytesIO(response.content)).convert('RGB')
        # Load and preprocess the image
        image = image.resize((IMG_SIZE_single, IMG_SIZE_single))  # Ensure correct input size
        
        if transform:
            image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        else:
            image_tensor = transforms.ToTensor()(image).unsqueeze(0)
        # Put the model in evaluation mode
        model.eval()
        
        # Make the prediction
        with torch.no_grad():
            output = model(image_tensor).squeeze()
            prediction = (output >= 0.5).item()  # Convert to binary prediction
        
        # Interpret and return the prediction
        if prediction == 0:
            return "fundus"
        else:
            return "non-fundus"
    except Exception as e:
        return f"Error processing the image: {e}"

def save_results_to_minio(patient_id, results, filename='results.json'):
    # Ensure the bucket exists
    if not minio_client.bucket_exists(BUCKET_NAME):
        minio_client.make_bucket(BUCKET_NAME)

    # Create the object name (file path inside bucket)
    object_name = f"{patient_id}/{filename}"

    # Convert results to JSON string and then to bytes
    json_data = json.dumps(results, indent=4)
    json_bytes = json_data.encode('utf-8')
    json_stream = BytesIO(json_bytes)

    # Upload JSON to MinIO
    minio_client.put_object(
        bucket_name=BUCKET_NAME,
        object_name=object_name,
        data=json_stream,
        length=len(json_bytes),
        content_type="application/json"
    )

IMG_SIZE = 600
NUM_CLASSES = 4
BEST_MODEL_PATH = "model_paths/effinet_model_V2.pth"
checkpoint = torch.load(BEST_MODEL_PATH, map_location=DEVICE,weights_only=False)
def process_image_from_url(image_url):
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        return image
    except Exception as e:
        raise Exception(f"Image processing error: {e}")

class EfficientNetB3Model(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetB3Model, self).__init__()
        self.efficientnet = models.efficientnet_b3(weights=None)
        num_ftrs = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.efficientnet(x)
    

#left and right prediction model 
IMG_SIZE_left_right = 300  # Same as used during training
MODEL_SAVE_PATH_left_right = 'model_paths/left_right_Model.pth'
class BinaryClassifier_left_right(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        try:
            self.model = EfficientNet.from_pretrained('efficientnet-b3')
            num_features = self.model._fc.in_features
            self.model._fc = nn.Sequential(
                nn.Linear(num_features, 1),
                nn.Sigmoid()
            )
        except Exception as e:
            logger.error(f"Error initializing EfficientNet-B3: {e}")
            raise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
def load_model_left_right(model_path: str) -> torch.nn.Module:
    model = BinaryClassifier_left_right()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model
def preprocess_image_left_right(image_path: str) -> torch.Tensor:
    
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE_left_right + 20),
        transforms.CenterCrop(IMG_SIZE_left_right),
        transforms.ToTensor(),
    ])
    try:
        response = requests.get(image_path)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        image = transform(image)
        return image.unsqueeze(0)  # Add batch dimension
    except FileNotFoundError:
        logger.error(f"Image file not found: {image_path}")
        raise
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {e}")
        raise

def predict_left_rigth_image(image_path: str, model: torch.nn.Module) -> str:
    try:
        image_tensor = preprocess_image_left_right(image_path).to(DEVICE)
        with torch.no_grad():
            output = model(image_tensor).squeeze()
            prediction = "Right" if output >= 0.5 else "Left"
            #confidence = output.item() if output >= 0.5 else 1 - output.item()
        return prediction #f"Prediction: {prediction}, Confidence: {confidence:.2f}"
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise


def trim(im):
    percentage = 0.02
    img = np.array(im)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    im_bin = img_gray > 0.1 * np.mean(img_gray[img_gray != 0])
    row_sums = np.sum(im_bin, axis=1)
    col_sums = np.sum(im_bin, axis=0)
    rows = np.where(row_sums > img.shape[1] * percentage)[0]
    cols = np.where(col_sums > img.shape[0] * percentage)[0]
    min_row, min_col = np.min(rows), np.min(cols)
    max_row, max_col = np.max(rows), np.max(cols)
    im_crop = img[min_row:max_row + 1, min_col:max_col + 1]
    return Image.fromarray(im_crop)

def resize_maintain_aspect(image, desired_size):
    old_size = image.size
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    im = image.resize(new_size, Image.LANCZOS)
    new_im = Image.new("RGB", (desired_size, desired_size))
    new_im.paste(im, ((desired_size - new_size[0]) // 2, (desired_size - new_size[1]) // 2))
    return new_im

def apply_clahe_color(image):
    lab = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final_image = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return Image.fromarray(final_image)

def process_image(image):
    trimmed_image = trim(image)
    resized_image = resize_maintain_aspect(trimmed_image, IMG_SIZE)
    final_image = apply_clahe_color(resized_image)
    return final_image

def infer(model, image_url, transform):
    img = process_image_from_url(image_url)
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    return predicted.item(), confidence.item()

def generate_result(predicted_class, confidence):
    explanation = explanation_labels.get(predicted_class,'Unknown')
    confidence_percentage = round(confidence * 100, 2)
    stage = CATEGORY_MAPPING[predicted_class]

    if predicted_class == 3:  # Proliferative Diabetic Retinopathy
        Risk_Factor = round(confidence * 100, 2)
    else:
        Risk_Factor = round((1 - confidence) * 100, 2)

    result = {
        "predicted_class": int(predicted_class),
        "Stage": stage,
        "confidence": confidence_percentage,
        "explanation": explanation,
        "Note": None , 
        "Risk_Factor": Risk_Factor   
    }

    # Add warnings based on the confidence and predicted class
    if confidence < 0.55 and predicted_class == 0:
        result["Note"] = f"You have a higher chance of progressing to the next stage with a risk factor of {Risk_Factor}%. Please consult your doctor for further advice."
    elif confidence >= 0.55 and confidence <= 0.74 and predicted_class == 0:
        result["Note"] = f"You have a minimum chance of progressing to the next stage with a risk factor of {Risk_Factor}%."
    elif confidence >= 0.75 and predicted_class == 0:
        result["Note"] = "Your eye is in the safe zone."
    elif predicted_class == 1:
        result["Note"] = f"Mild diabetic retinopathy detected. Risk factor is {Risk_Factor}%. Please consult your doctor for further advice."
    elif predicted_class == 2:
        result["Note"] = f"Moderate to severe diabetic retinopathy detected. Risk factor is {Risk_Factor}%. Please consult your doctor for further advice."
    elif predicted_class == 3:
        result["Note"] = "Proliferative diabetic retinopathy detected. Urgent medical intervention is necessary to prevent severe vision loss. Please consult your healthcare provider immediately."

    return result

# Load the model once at the start of the application
model = EfficientNetB3Model(NUM_CLASSES).to(DEVICE)
checkpoint = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

@app.post("/infer_for_diabetic_retinopathy/upload images")
async def run_inference(patient_id: str, left_image: UploadFile = File(...), right_image: UploadFile = File(...),request: Request = None):
    # Define transform
    transform = transforms.Compose([transforms.ToTensor()])
    start_time = time.time()
    try :
        logger.info(f"[START] Inference API hit for patient: {patient_id} from {request.client.host}")
        # Upload the images to Azure Blob Storage
        left_image_url = upload_image_to_minio(patient_id, await left_image.read(), 'left_image.jpg')
        right_image_url = upload_image_to_minio(patient_id, await right_image.read(), 'right_image.jpg')

        #run inference for model_1 fundus and non fundus image 
        left_prediction= predict_single_image(model_1, left_image_url, transform_model_1)
        right_prediction= predict_single_image(model_1, right_image_url, transform_model_1)
        if left_prediction == "non-fundus" or right_prediction == "non-fundus":
                logger.warning(f"Invalid image detected. Left: {left_prediction}, Right: {right_prediction}")
                return JSONResponse(content={
                    "message": "One or more uploaded images are not valid fundus images. Please upload valid fundus images."
                })
        print(left_prediction,right_prediction)
        
        # model prediction for dectiting images of left and right images right 
        model_left_right = load_model_left_right(MODEL_SAVE_PATH_left_right)
        left_image_detection=predict_left_rigth_image(left_image_url,model_left_right)
        right_image_detection=predict_left_rigth_image(right_image_url,model_left_right)
        print(left_image_detection,right_image_detection)
        
        if left_image_detection == "Right" and right_image_detection == "Left":
                logger.warning(f"Image swap detected: Left image is 'Right', Right image is 'Left'. Patient uploaded images in wrong order.")
                return JSONResponse(content={
                    "message": f"The right image has been uploaded in place of the left image. Please reupload the correct images wit id{patient_id}"
                })
        if left_image_detection == "Left" and right_image_detection == "Left":
                logger.warning(f"Both images identified as Left eye. Missing right eye image.")
                return JSONResponse(content={
                    "message": "Both uploaded images are identified as left eye images. Please reupload the correct right eye image."
                })
        if left_image_detection == "Right" and right_image_detection == "Right":
                logger.warning(f"Both images identified as Right eye. Missing left eye image.")
                return JSONResponse(content={
                    "message": "Both uploaded images are identified as right eye images. Please reupload the correct left eye image."
                })
        # Run inference using image URLs
        left_class, left_confidence = infer(model, left_image_url, transform)
        right_class, right_confidence = infer(model, right_image_url, transform)
        print(left_class,right_class)
        # Generate detailed results
        left_result = generate_result(left_class, left_confidence)
        right_result = generate_result(right_class, right_confidence)
        
        # Prepare final results
        results = {
            "left_eye": left_result,
            "right_eye": right_result
        }
        print(left_result)
        print(right_result)
        save_results_to_minio(patient_id, results)
        process_time = time.time() - start_time
        logger.info(
            f"[SUCCESS] Processed inference for {patient_id} | Duration: {process_time:.2f}s"
        )
        
        return JSONResponse(content=results)
    except Exception as e:
        return JSONResponse(content={"error": str(e)})
@app.post("/submit_feedback_from_frontend/from_json_to_db")  
async def submit_feedback(data: dict):  # Receive JSON payload directly
    try:
        # Extract data for left_eye and right_eye
        left_eye = data.get("left_eye")
        right_eye = data.get("right_eye")

        # Validate that both left_eye and right_eye data are present
        if not left_eye or not right_eye:
            raise HTTPException(status_code=400, detail="Invalid data: Missing 'left_eye' or 'right_eye' fields.")
        # Extract patient ID if provided
        patient_id = data.get("patient_id","default_patient_id")  # Replace with a fallback if not provided
        email_id=data.get("email_id")
        # connection=connection()
        cursor = connection.cursor()

        # SQL query for inserting data
        query = """
        INSERT INTO diabetic_retinopathy (
            Patient_ID,
            Predicted_Class,
            Stage,
            Confidence,
            Explanation,
            Note,
            Risk_Factor,
            Review,
            Feedback,
            Doctors_Diagnosis,
            email_id,
            timestamp
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,%s,%s);
        """
        utc_now = datetime.now(pytz.utc)  # Get current UTC time
        kolkata_tz = pytz.timezone('Asia/Kolkata')
        kolkata_now = utc_now.astimezone(kolkata_tz)
        kolkata_now_str = kolkata_now.strftime('%Y-%m-%d %H:%M:%S')
        kolkata_now_str
        # Prepare values for left eye
        left_values = (
            patient_id + "_left",
            left_eye["predicted_class"],
            left_eye["Stage"],
            left_eye["confidence"],
            left_eye["explanation"],
            left_eye["Note"],
            left_eye["Risk_Factor"],
            left_eye["review"],
            left_eye["feedback"],
            left_eye["doctors_diagnosis"],
            email_id,
            kolkata_now_str
        )
        # Prepare values for right eye
        right_values = (
            patient_id + "_right",
            right_eye["predicted_class"],
            right_eye["Stage"],
            right_eye["confidence"],
            right_eye["explanation"],
            right_eye["Note"],
            right_eye["Risk_Factor"],
            right_eye["review"],
            right_eye["feedback"],
            right_eye["doctors_diagnosis"],
            email_id,
            kolkata_now_str
        )
        try:
            # Insert data for left eye
            cursor.execute(query, left_values)

            # Insert data for right eye
            cursor.execute(query, right_values)

            # Commit the transaction
            connection.commit()
            logger.info(f"Data inserted successfully for patient id {patient_id}")


        except Exception as e:
            print(f"An error occurred: {e}")
            connection.rollback()  # Rollback the transaction in case of an error
            raise HTTPException(status_code=500, detail=f"Database error: {e}")
        finally:
            cursor.close()
            #connection.close()  # Ensure the connection is closed

        return JSONResponse(content={"message": f"Feedback and results saved successfully for patient is:{patient_id}"})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
@app.get("/get_data_from_api_logs")
def get_api_logs():
    return data_from_db("SELECT * FROM api_logs")

@app.get("/get_data_from_db")
def get_dr_data():
    return data_from_db("SELECT * FROM diabetic_retinopathy")


@app.post("/", status_code=status.HTTP_201_CREATED)
def create_patient(patient_id: int, form: Form2):
    # create the DB connection
    conn = connection
    cursor = conn.cursor(dictionary=True)

    try:
        # 1. Check if patient exists
        cursor.execute("""
            SELECT * FROM patient_form 
            WHERE patient_id = %s 
            ORDER BY visit_id DESC 
            LIMIT 1
        """, (patient_id,))
        existing_patient = cursor.fetchone()

        # 2. If mobile number mismatch, reject
        if existing_patient and existing_patient["mobile_number"] != form.mobile_number:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Patient ID exists but with a different mobile number."
            )

        # 3. Validation checks
        if not (0 <= form.Duration_of_Diabetes <= 50):
            raise HTTPException(status_code=400, detail="Duration of Diabetes must be between 0 and 50 years.")
        if not (4.0 <= form.HbA1c_Level <= 14.0):
            raise HTTPException(status_code=400, detail="HbA1c Level must be between 4.0% and 14.0%.")
        if not (90 <= form.Blood_Pressure <= 200):
            raise HTTPException(status_code=400, detail="Blood Pressure must be between 90 and 200 mmHg.")
        if not (70 <= form.Fasting_Blood_Glucose <= 300):
            raise HTTPException(status_code=400, detail="Fasting Blood Glucose must be between 70 and 300 mg/dL.")
        if not (15 <= form.BMI <= 50):
            raise HTTPException(status_code=400, detail="BMI must be between 15 and 50 kg/m².")
        if not (100 <= form.Cholesterol <= 300):
            raise HTTPException(status_code=400, detail="Cholesterol must be between 100 and 300 mg/dL.")
        if not (18 <= form.Age <= 90):
            raise HTTPException(status_code=400, detail="Age must be between 18 and 90 years.")
        if not (0.5 <= form.Albuminuria <= 3.0):
            raise HTTPException(status_code=400, detail="Albuminuria must be between 0.5 and 3.0 mg/dL.")
        if form.Visual_Acuity not in ["Normal", "Mild", "Moderate", "Severe", "Proliferative"]:
            raise HTTPException(status_code=400, detail="Visual Acuity must be one of: Normal, Mild, Moderate, Severe, Proliferative")

        # 4. Insert into table
        insert_query = """
            INSERT INTO patient_form (
                patient_id, name, Duration_of_Diabetes, HbA1c_Level, Blood_Pressure,
                Fasting_Blood_Glucose, BMI, Cholesterol, Age, Albuminuria,
                Visual_Acuity, Date_of_registration, Hospital_name, num_visits,
                mobile_number, gender
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        cursor.execute(insert_query, (
            patient_id, form.name, form.Duration_of_Diabetes, form.HbA1c_Level,
            form.Blood_Pressure, form.Fasting_Blood_Glucose, form.BMI,
            form.Cholesterol, form.Age, form.Albuminuria, form.Visual_Acuity,
            form.Date_of_registration, form.Hospital_name, form.num_visits,
            form.mobile_number, form.gender
        ))

        conn.commit()
        
        logger.info(f"Data inserted successfully for patient id {patient_id}")

        return {"message": "Patient record created successfully", "patient_id": patient_id}
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        connection.rollback()  # Rollback the transaction in case of an error
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

    finally:
        cursor.close()
        #conn.close()
        


@app.get("/patient/{patient_id}", status_code=status.HTTP_200_OK)
def get_patient(patient_id: int):
    conn = connection
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT * FROM patient_form WHERE patient_id = %s", (patient_id,))
        patients = cursor.fetchall()
        if not patients:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found")
        logger.info(f"Data get successfully for patient id {patient_id}")
        return patients
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        connection.rollback()  # Rollback the transaction in case of an error
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    finally:
        cursor.close()
        #conn.close()


@app.delete("/patient/{patient_id}", status_code=status.HTTP_200_OK)
def delete_patient(patient_id: int):
    conn = connection  # Get a fresh DB connection
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT COUNT(*) FROM patient_form WHERE patient_id = %s", (patient_id,))
        (count,) = cursor.fetchone()

        if count == 0:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found")

        # Delete all records for that patient
        cursor.execute("DELETE FROM patient_form WHERE patient_id = %s", (patient_id,))
        conn.commit()
        logger.info(f" Deleted patient with patient_id {patient_id}")
        return {"detail": f"Deleted patient with patient_id {patient_id}"}
    except Exception as e:
        logger.error(f"❌ Error while deleting patient: {e}")
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")         
    finally:
        cursor.close()
        #conn.close()
@app.patch("/update/{patient_id}")
def update_patient_partial(patient_id: int, form: PatientPartialUpdate):
    conn = connection
    cursor = conn.cursor(dictionary=True)
    try:
        # Get latest visit record for the patient
        cursor.execute("""
            SELECT * FROM patient_form 
            WHERE patient_id = %s 
            ORDER BY visit_id DESC 
            LIMIT 1
        """, (patient_id,))
        patient = cursor.fetchone()
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")
        update_data = form.dict(exclude_unset=True)

        # Validations
        if "Duration_of_Diabetes" in update_data:
            if not (0 <= update_data["Duration_of_Diabetes"] <= 50):
                raise HTTPException(status_code=400, detail="Duration must be between 0 and 50")

        if "HbA1c_Level" in update_data:
            if not (4.0 <= update_data["HbA1c_Level"] <= 14.0):
                raise HTTPException(status_code=400, detail="HbA1c Level must be between 4.0% and 14.0%")

        if "Blood_Pressure" in update_data:
            if not (90 <= update_data["Blood_Pressure"] <= 200):
                raise HTTPException(status_code=400, detail="Blood Pressure must be between 90 and 200 mmHg")

        if "Fasting_Blood_Glucose" in update_data:
            if not (70 <= update_data["Fasting_Blood_Glucose"] <= 300):
                raise HTTPException(status_code=400, detail="Fasting Blood Glucose must be between 70 and 300 mg/dL")

        if "BMI" in update_data:
            if not (15 <= update_data["BMI"] <= 50):
                raise HTTPException(status_code=400, detail="BMI must be between 15 and 50")

        if "Cholesterol" in update_data:
            if not (100 <= update_data["Cholesterol"] <= 300):
                raise HTTPException(status_code=400, detail="Cholesterol must be between 100 and 300")

        if "Age" in update_data:
            if not (18 <= update_data["Age"] <= 90):
                raise HTTPException(status_code=400, detail="Age must be between 18 and 90")

        if "Albuminuria" in update_data:
            if not (0.5 <= update_data["Albuminuria"] <= 3.0):
                raise HTTPException(status_code=400, detail="Albuminuria must be between 0.5 and 3.0 mg/dL")

        if "Visual_Acuity" in update_data:
            if update_data["Visual_Acuity"] not in ["Normal", "Mild", "Moderate", "Severe", "Proliferative"]:
                raise HTTPException(status_code=400, detail=" Visual Acuity in [Normal, Mild, Moderate, Severe, Proliferative] ")

        # Build dynamic SQL update statement
        set_clause = ", ".join(f"{key} = %s" for key in update_data.keys())
        values = list(update_data.values())
        values.append(patient["patient_id"])  

        update_query = f"""
            UPDATE patient_form
            SET {set_clause}
            WHERE patient_id = %s
        """

        cursor.execute(update_query, values)
        conn.commit()
    
        logger.info(f"Data updated successfully for patient id {patient_id}")
        return {"message": "Updated successfully", "updated_fields": update_data}
    except Exception as e:
        logger.error(f"Error while updating patient: {e}")
        conn.rollback()  # Rollback the transaction in case of an error
        raise HTTPException(status_code=500,detail=f"Database error: {str(e)}")
    finally:
        cursor.close()
        #conn.close()

@app.get("/patients", status_code=status.HTTP_200_OK)
def get_all_patients():
    conn = connection  # No parentheses
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT * FROM patient_form")
        patients = cursor.fetchall()
        if not patients:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No patients found")
        logger.info("✅ Data retrieved successfully for all patients")
        return patients
    except Exception as e:
        logger.error(f"❌ Error retrieving patient data: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        cursor.close()


@app.get("/combined-report/{patient_id}", status_code=status.HTTP_200_OK)
def get_combined_patient_report(patient_id: int):
    try:
        conn = connection  # assuming 'connection' is your MySQL connection object
        cursor = conn.cursor(dictionary=True)

        query = """
        SELECT
            pf.visit_id,
            pf.name,
            pf.patient_id,
            pf.Hospital_name,
            pf.Date_of_registration,
            pf.mobile_number,
            pf.gender,
            pf.Age,
            pf.Blood_Pressure,
            pf.HbA1c_Level,
            pf.Fasting_Blood_Glucose,
            pf.Duration_of_Diabetes,
            pf.BMI,
            pf.Cholesterol,
            pf.Albuminuria,
            pf.Visual_Acuity,
            dr.Patient_ID AS eye_scan_id,
            dr.Stage,
            dr.Confidence,
            dr.Explanation,
            dr.Note,
            dr.Risk_Factor,
            dr.Review,
            dr.Feedback,
            dr.Doctors_Diagnosis,
            dr.timestamp
        FROM
            patient_form pf
        JOIN
            diabetic_retinopathy dr
            ON CAST(SUBSTRING_INDEX(dr.Patient_ID, '_', 1) AS UNSIGNED) = pf.patient_id
        WHERE
            pf.patient_id = %s
        """
        cursor.execute(query, (patient_id,))
        result = cursor.fetchall()
        cursor.close()

        if not result:
            raise HTTPException(status_code=404, detail="No records found for this patient ID.")

        return result

    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")

    finally:
        cursor.close()

    
ALLOWED_PATHS_FOR_INSTRUMENTATION = {
    "/get_data_from_api_logs",
    "/get_data_from_db",
    "/infer_for_diabetic_retinopathy/upload images"
}

# The path for the metrics endpoint itself
METRICS_PATH = "/metrics" # Default, but good to have as a variable


def should_instrument_callback(request: Request) -> bool:
    path = request.url.path
    # 1. Explicitly DO NOT instrument requests to the /metrics endpoint itself
    if path == METRICS_PATH:
        return False
    # 2. Instrument other allowed paths
    if path in ALLOWED_PATHS_FOR_INSTRUMENTATION:
        return True
    # 3. Do not instrument any other paths
    return False

# # Initialize Instrumentator
# instrumentator = Instrumentator(
   
# )


# instrumentator.instrument(
#     app,
#     should_instrument_hook=should_instrument_callback
# ).expose(app, endpoint=METRICS_PATH) # You can specify the endpoint path for expose too
#
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

if __name__ == "__main__":
    import uvicorn 
    uvicorn.run(app, host="0.0.0.0", port=8000)
