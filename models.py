from pydantic import BaseModel
from datetime import datetime
from typing import Optional
import math

class VitalSigns(BaseModel):
    heart_rate: int
    respiratory_rate: int
    body_temperature: float
    oxygen_saturation: int
    systolic_blood_pressure: int
    diastolic_blood_pressure: int
    age: int
    gender: int
    weight: float
    height: float
    derived_hrv: float
    derived_pulse_pressure: float
    derived_bmi: float
    derived_map: float

class RiskAssessment(BaseModel):
    risk_level: str
    vital_signs: VitalSigns
    confidence_score: float = 0.0
    priority_score: float = 0.0
    timestamp: Optional[datetime] = None
    time_factor: float = 1.0
    
    def calculate_priority_score(self) -> float:
        """Calculate priority score based on confidence, risk level, and time factor"""
        # Base risk scores
        risk_weights = {
            "high": 100.0,
            "medium": 50.0,
            "low": 10.0
        }
        
        base_score = risk_weights.get(self.risk_level.lower(), 10.0)
        
        # Apply confidence factor (higher confidence = higher priority)
        confidence_factor = self.confidence_score
        
        # Calculate time factor (urgency increases over time)
        time_urgency = self._calculate_time_urgency()
        
        # Age factor (elderly patients get higher priority)
        age_factor = min(self.vital_signs.age / 80.0, 1.5)
        
        # Critical vital signs factor
        critical_factor = self._calculate_critical_vitals_factor()
        
        # Final priority score calculation
        priority = (base_score * confidence_factor * time_urgency * 
                   (1 + age_factor) * (1 + critical_factor))
        
        self.priority_score = priority
        return priority
    
    def _calculate_time_urgency(self) -> float:
        """Calculate urgency based on time elapsed since assessment"""
        if not self.timestamp:
            self.timestamp = datetime.now()
            return 1.0
        
        time_elapsed = (datetime.now() - self.timestamp).total_seconds() / 60.0  # minutes
        
        if self.risk_level.lower() == "high":
            # High risk patients: exponential urgency increase
            return 1.0 + (time_elapsed / 10.0)  # Increases every 10 minutes
        elif self.risk_level.lower() == "medium":
            # Medium risk: moderate increase
            return 1.0 + (time_elapsed / 30.0)  # Increases every 30 minutes
        else:
            # Low risk: slow increase but with RL scheduling
            return 1.0 + (time_elapsed / 120.0)  # Increases every 2 hours
    
    def _calculate_critical_vitals_factor(self) -> float:
        """Calculate additional priority based on critical vital signs"""
        vs = self.vital_signs
        critical_factor = 0.0
        
        # Critical heart rate (bradycardia < 50, tachycardia > 120)
        if vs.heart_rate < 50 or vs.heart_rate > 120:
            critical_factor += 0.3
        
        # Critical blood pressure (hypotension < 90, hypertension > 180)
        if vs.systolic_blood_pressure < 90 or vs.systolic_blood_pressure > 180:
            critical_factor += 0.4
        
        # Critical oxygen saturation (< 90%)
        if vs.oxygen_saturation < 90:
            critical_factor += 0.5
        
        # Critical temperature (hypothermia < 35°C, hyperthermia > 39°C)
        if vs.body_temperature < 35.0 or vs.body_temperature > 39.0:
            critical_factor += 0.3
        
        # Critical respiratory rate (< 12 or > 25)
        if vs.respiratory_rate < 12 or vs.respiratory_rate > 25:
            critical_factor += 0.2
        
        return min(critical_factor, 1.0)  # Cap at 1.0

class PatientQueue(BaseModel):
    """Model for managing patient queue with RL-based scheduling"""
    patient_id: str
    risk_assessment: RiskAssessment
    queue_position: int = 0
    estimated_wait_time: int = 0  # minutes
    rl_adjustment: float = 0.0  # RL-based priority adjustment for low-risk patients
    rl_state: Optional[str] = None  # RL state for learning
    rl_action: Optional[str] = None  # RL action taken
    
    def get_final_priority(self) -> float:
        """Get final priority score including RL adjustments"""
        base_priority = self.risk_assessment.calculate_priority_score()
        
        # Apply RL adjustment for low-risk patients
        if self.risk_assessment.risk_level.lower() == "low":
            # RL can boost or reduce priority based on learned policy
            rl_boost = self.rl_adjustment
            return base_priority + rl_boost
        
        return base_priority
    
    def update_time_factor(self):
        """Update the time factor for dynamic priority adjustment"""
        self.risk_assessment.time_factor = self.risk_assessment._calculate_time_urgency()