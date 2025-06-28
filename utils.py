import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple, List
from models import VitalSigns, RiskAssessment, PatientQueue

def scale_data(data, scaler):
    """
    Scale the input data using the provided scaler.
    Args:
        data (list or np.ndarray or pd.DataFrame): Input features.
        scaler: Fitted scaler object.
    Returns:
        np.ndarray: Scaled data.
    """
    df = pd.DataFrame([data]) if not isinstance(data, (pd.DataFrame, np.ndarray)) else data
    return scaler.transform(df)

def encode_labels(labels, label_encoder):
    """
    Encode the labels using the provided label encoder.
    Args:
        labels (list or np.ndarray): Labels to encode.
        label_encoder: Fitted label encoder object.
    Returns:
        np.ndarray: Encoded labels.
    """
    return label_encoder.transform(labels)

def predict_risk_with_confidence(features, scaler, model, label_encoder):
    """
    Predict the risk label and confidence score for the given features.
    Args:
        features (list): Input features for prediction.
        scaler: Fitted scaler object.
        model: Trained model object.
        label_encoder: Fitted label encoder object.
    Returns:
        tuple: (predicted_risk_label, confidence_score)
    """
    scaled = scale_data(features, scaler)
    
    # Get prediction probabilities for confidence
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(scaled)
        confidence_score = float(np.max(probabilities[0]))
    else:
        # For models without predict_proba, use distance from decision boundary
        prediction = model.predict(scaled)
        confidence_score = 0.8  # Default confidence
    
    pred_encoded = model.predict(scaled)
    pred_label = label_encoder.inverse_transform(pred_encoded)[0]
    
    return pred_label, confidence_score

def predict_risk(features, scaler, model, label_encoder):
    """
    Predict the risk label for the given features.
    Args:
        features (list): Input features for prediction.
        scaler: Fitted scaler object.
        model: Trained model object.
        label_encoder: Fitted label encoder object.
    Returns:
        str: Predicted risk label.
    """
    pred_label, _ = predict_risk_with_confidence(features, scaler, model, label_encoder)
    return pred_label

def create_risk_assessment_with_priority(vital_signs_data: dict, scaler, model, label_encoder) -> RiskAssessment:
    """
    Create a RiskAssessment with priority scoring
    Args:
        vital_signs_data: Dictionary containing vital signs data
        scaler: Fitted scaler object
        model: Trained model object
        label_encoder: Fitted label encoder object
    Returns:
        RiskAssessment: Complete risk assessment with priority score
    """
    # Create VitalSigns object
    vital_signs = VitalSigns(**vital_signs_data)
    
    # Prepare features for prediction
    features = [
        vital_signs.heart_rate,
        vital_signs.respiratory_rate,
        vital_signs.body_temperature,
        vital_signs.oxygen_saturation,
        vital_signs.systolic_blood_pressure,
        vital_signs.diastolic_blood_pressure,
        vital_signs.age,
        vital_signs.gender,
        vital_signs.weight,
        vital_signs.height,
        vital_signs.derived_hrv,
        vital_signs.derived_pulse_pressure,
        vital_signs.derived_bmi,
        vital_signs.derived_map
    ]
    
    # Get prediction with confidence
    risk_level, confidence_score = predict_risk_with_confidence(features, scaler, model, label_encoder)
    
    # Create risk assessment
    risk_assessment = RiskAssessment(
        risk_level=risk_level,
        vital_signs=vital_signs,
        confidence_score=confidence_score,
        timestamp=datetime.now()
    )
    
    # Calculate priority score
    risk_assessment.calculate_priority_score()
    
    return risk_assessment

def manage_patient_queue(patient_assessments: List[RiskAssessment]) -> List[PatientQueue]:
    """
    Create and sort patient queue based on priority scores with RL integration
    Args:
        patient_assessments: List of RiskAssessment objects
    Returns:
        List[PatientQueue]: Sorted patient queue by priority
    """
    from rl_scheduler import PriorityManager
    
    patient_queue = []
    priority_manager = PriorityManager()
    
    for i, assessment in enumerate(patient_assessments):
        patient_id = f"patient_{i+1}_{int(assessment.timestamp.timestamp())}"
        
        queue_item = PatientQueue(
            patient_id=patient_id,
            risk_assessment=assessment
        )
        
        patient_queue.append(queue_item)
    
    # Use RL-enhanced priority management
    sorted_queue = priority_manager.calculate_dynamic_priority(patient_queue)
    
    # Update queue positions and estimated wait times
    for i, patient in enumerate(sorted_queue):
        patient.queue_position = i + 1
        patient.estimated_wait_time = calculate_estimated_wait_time(patient, i)
    
    return sorted_queue

def calculate_estimated_wait_time(patient: PatientQueue, position: int) -> int:
    """
    Calculate estimated wait time based on position and risk level
    Args:
        patient: PatientQueue object
        position: Position in queue (0-based)
    Returns:
        int: Estimated wait time in minutes
    """
    base_time_per_patient = {
        "high": 15,    # 15 minutes per high-risk patient
        "medium": 20,  # 20 minutes per medium-risk patient
        "low": 25      # 25 minutes per low-risk patient
    }
    
    risk_level = patient.risk_assessment.risk_level.lower()
    base_time = base_time_per_patient.get(risk_level, 20)
    
    # Calculate wait time based on position
    estimated_wait = position * base_time
    
    # Apply urgency factor for high-risk patients
    if risk_level == "high":
        estimated_wait = max(0, estimated_wait - 10)  # Reduce wait time for high-risk
    
    return estimated_wait