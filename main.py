from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from datetime import datetime
from typing import List
from schemas import RiskPredictionResponse, PatientQueueResponse, VitalSigns as SchemaVitalSigns
from utils import create_risk_assessment_with_priority, manage_patient_queue
from models import RiskAssessment, PatientQueue

app = FastAPI()

# Load models and encoders
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')
xgb = joblib.load('xgb_model.pkl')

# In-memory storage for patient queue (in production, use a database)
patient_assessments: List[RiskAssessment] = []

# Define input schema
class VitalSigns(BaseModel):
    Heart_Rate: float
    Respiratory_Rate: float
    Body_Temperature: float
    Oxygen_Saturation: float
    Systolic_Blood_Pressure: float
    Diastolic_Blood_Pressure: float
    Age: float
    Gender: int
    Weight_kg: float
    Height_m: float
    Derived_HRV: float
    Derived_Pulse_Pressure: float
    Derived_BMI: float
    Derived_MAP: float

@app.get("/")
def read_root():
    return {"message": "Welcome to the Health Risk Predictor FastAPI app with Priority Scheduling!"}

@app.post("/predict/", response_model=RiskPredictionResponse)
def predict_risk_with_priority(vitals: VitalSigns):
    """Predict risk with confidence score and priority calculation"""
    
    # Convert input to the format expected by utils
    vital_signs_data = {
        "heart_rate": int(vitals.Heart_Rate),
        "respiratory_rate": int(vitals.Respiratory_Rate),
        "body_temperature": vitals.Body_Temperature,
        "oxygen_saturation": int(vitals.Oxygen_Saturation),
        "systolic_blood_pressure": int(vitals.Systolic_Blood_Pressure),
        "diastolic_blood_pressure": int(vitals.Diastolic_Blood_Pressure),
        "age": int(vitals.Age),
        "gender": vitals.Gender,
        "weight": vitals.Weight_kg,
        "height": vitals.Height_m,
        "derived_hrv": vitals.Derived_HRV,
        "derived_pulse_pressure": vitals.Derived_Pulse_Pressure,
        "derived_bmi": vitals.Derived_BMI,
        "derived_map": vitals.Derived_MAP
    }
    
    # Create risk assessment with priority
    risk_assessment = create_risk_assessment_with_priority(
        vital_signs_data, scaler, xgb, label_encoder
    )
    
    # Add to patient assessments for queue management
    patient_assessments.append(risk_assessment)
    
    # Calculate estimated wait time based on current queue
    current_queue = manage_patient_queue(patient_assessments)
    
    # Find this patient's position in queue (last added)
    estimated_wait_time = 0
    if current_queue:
        # Find the patient with matching timestamp (more reliable than vital signs)
        for patient in current_queue:
            if abs((patient.risk_assessment.timestamp - risk_assessment.timestamp).total_seconds()) < 1:
                estimated_wait_time = patient.estimated_wait_time
                break
    
    return RiskPredictionResponse(
        risk_level=risk_assessment.risk_level,
        confidence_score=risk_assessment.confidence_score,
        priority_score=risk_assessment.priority_score,
        estimated_wait_time=estimated_wait_time,
        timestamp=risk_assessment.timestamp,
        details=SchemaVitalSigns(
            heart_rate=risk_assessment.vital_signs.heart_rate,
            respiratory_rate=risk_assessment.vital_signs.respiratory_rate,
            body_temperature=risk_assessment.vital_signs.body_temperature,
            oxygen_saturation=risk_assessment.vital_signs.oxygen_saturation,
            systolic_blood_pressure=risk_assessment.vital_signs.systolic_blood_pressure,
            diastolic_blood_pressure=risk_assessment.vital_signs.diastolic_blood_pressure,
            age=risk_assessment.vital_signs.age,
            gender=risk_assessment.vital_signs.gender,
            weight=risk_assessment.vital_signs.weight,
            height=risk_assessment.vital_signs.height,
            derived_hrv=risk_assessment.vital_signs.derived_hrv,
            derived_pulse_pressure=risk_assessment.vital_signs.derived_pulse_pressure,
            derived_bmi=risk_assessment.vital_signs.derived_bmi,
            derived_map=risk_assessment.vital_signs.derived_map
        )
    )

@app.get("/queue/", response_model=List[PatientQueueResponse])
def get_patient_queue():
    """Get current patient queue sorted by priority"""
    
    if not patient_assessments:
        return []
    
    current_queue = manage_patient_queue(patient_assessments)
    
    return [
        PatientQueueResponse(
            patient_id=patient.patient_id,
            risk_level=patient.risk_assessment.risk_level,
            confidence_score=patient.risk_assessment.confidence_score,
            priority_score=patient.get_final_priority(),
            queue_position=patient.queue_position,
            estimated_wait_time=patient.estimated_wait_time,
            timestamp=patient.risk_assessment.timestamp
        )
        for patient in current_queue
    ]

@app.post("/queue/update-priorities/")
def update_queue_priorities():
    """Update priorities for all patients in queue (accounts for time factor)"""
    
    if not patient_assessments:
        return {"message": "No patients in queue"}
    
    # Update time factors and recalculate priorities
    for assessment in patient_assessments:
        assessment.calculate_priority_score()
    
    updated_queue = manage_patient_queue(patient_assessments)
    
    return {
        "message": f"Updated priorities for {len(updated_queue)} patients",
        "high_priority_count": len([p for p in updated_queue if p.risk_assessment.risk_level.lower() == "high"]),
        "medium_priority_count": len([p for p in updated_queue if p.risk_assessment.risk_level.lower() == "medium"]),
        "low_priority_count": len([p for p in updated_queue if p.risk_assessment.risk_level.lower() == "low"])
    }

@app.delete("/queue/clear/")
def clear_queue():
    """Clear the patient queue"""
    global patient_assessments
    patient_count = len(patient_assessments)
    patient_assessments = []
    return {"message": f"Cleared {patient_count} patients from queue"}

@app.post("/feedback/")
def provide_feedback(patient_id: str, actual_wait_time: int, satisfaction_score: float, resource_utilization: float = 0.5):
    """
    Provide feedback to the RL system for learning
    Args:
        patient_id: Patient identifier
        actual_wait_time: Actual wait time in minutes
        satisfaction_score: Patient satisfaction (0.0 to 1.0)
        resource_utilization: Resource utilization efficiency (0.0 to 1.0)
    """
    from rl_scheduler import PriorityManager
    
    outcome = {
        'actual_wait_time': actual_wait_time,
        'satisfaction_score': satisfaction_score,
        'resource_utilization': resource_utilization,
        'timestamp': datetime.now()
    }
    
    priority_manager = PriorityManager()
    priority_manager.update_with_outcome(patient_id, outcome)
    
    return {
        "message": "Feedback received and RL system updated",
        "patient_id": patient_id,
        "outcome": outcome
    }

@app.get("/queue/next/", response_model=PatientQueueResponse)
def get_next_patient():
    """Get the next patient to be seen (highest priority)"""
    
    if not patient_assessments:
        # Return a proper error response that matches the response model
        raise HTTPException(status_code=404, detail="No patients in queue")
    
    from rl_scheduler import PriorityManager
    
    current_queue = manage_patient_queue(patient_assessments)
    priority_manager = PriorityManager()
    
    next_patient = priority_manager.get_next_patient(current_queue)
    
    if not next_patient:
        raise HTTPException(status_code=404, detail="No patients available")
    
    # Remove the patient from queue (they're being seen)
    patient_assessments[:] = [
        assessment for assessment in patient_assessments 
        if assessment.timestamp != next_patient.risk_assessment.timestamp
    ]
    
    return PatientQueueResponse(
        patient_id=next_patient.patient_id,
        risk_level=next_patient.risk_assessment.risk_level,
        confidence_score=next_patient.risk_assessment.confidence_score,
        priority_score=next_patient.get_final_priority(),
        queue_position=next_patient.queue_position,
        estimated_wait_time=next_patient.estimated_wait_time,
        timestamp=next_patient.risk_assessment.timestamp
    )