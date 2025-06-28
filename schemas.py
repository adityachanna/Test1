from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class VitalSigns(BaseModel):
    heart_rate: int
    respiratory_rate: int
    body_temperature: float
    oxygen_saturation: int
    systolic_blood_pressure: int
    diastolic_blood_pressure: int
    age: int
    gender: int  # 0 for female, 1 for male
    weight: float
    height: float
    derived_hrv: float
    derived_pulse_pressure: float
    derived_bmi: float
    derived_map: float

class RiskPredictionResponse(BaseModel):
    risk_level: str
    confidence_score: float
    priority_score: float
    estimated_wait_time: int
    timestamp: datetime
    details: VitalSigns

class PatientQueueResponse(BaseModel):
    patient_id: str
    risk_level: str
    confidence_score: float
    priority_score: float
    queue_position: int
    estimated_wait_time: int
    timestamp: datetime