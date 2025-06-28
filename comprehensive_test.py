#!/usr/bin/env python3
"""
Comprehensive test script to verify the patient queue and priority logic
"""

import requests
import json
import time
from datetime import datetime

# Base URL for the API
BASE_URL = "http://127.0.0.1:8002"

def clear_queue():
    """Clear the patient queue"""
    response = requests.delete(f"{BASE_URL}/queue/clear/")
    if response.status_code == 200:
        print("✅ Queue cleared")
        return True
    else:
        print(f"❌ Failed to clear queue: {response.text}")
        return False

def add_patient(patient_data, description):
    """Add a patient and return the response"""
    print(f"Adding {description}...")
    response = requests.post(f"{BASE_URL}/predict/", json=patient_data)
    if response.status_code == 200:
        result = response.json()
        print(f"   ✅ {description} added - Risk: {result['risk_level']}, Priority: {result['priority_score']:.1f}")
        return result
    else:
        print(f"   ❌ Failed to add {description}: {response.text}")
        return None

def test_comprehensive_queue():
    """Test the queue with multiple patients of different risk levels"""
    print("🏥 Testing Comprehensive Patient Queue Logic\n")
    
    # Clear existing queue
    clear_queue()
    
    # Test patient data with different risk profiles
    patients = [
        # Low risk patient
        {
            "data": {
                "Heart_Rate": 75.0,
                "Respiratory_Rate": 16.0,
                "Body_Temperature": 98.6,
                "Oxygen_Saturation": 99,
                "Systolic_Blood_Pressure": 115,
                "Diastolic_Blood_Pressure": 75,
                "Age": 25.0,
                "Gender": 0,
                "Weight_kg": 60.0,
                "Height_m": 1.65,
                "Derived_HRV": 55.0,
                "Derived_Pulse_Pressure": 40.0,
                "Derived_BMI": 22.0,
                "Derived_MAP": 88.3
            },
            "description": "Young Low-Risk Patient"
        },
        # Medium risk patient (elderly)
        {
            "data": {
                "Heart_Rate": 95.0,
                "Respiratory_Rate": 20.0,
                "Body_Temperature": 99.2,
                "Oxygen_Saturation": 95,
                "Systolic_Blood_Pressure": 145,
                "Diastolic_Blood_Pressure": 85,
                "Age": 75.0,
                "Gender": 1,
                "Weight_kg": 80.0,
                "Height_m": 1.70,
                "Derived_HRV": 25.0,
                "Derived_Pulse_Pressure": 60.0,
                "Derived_BMI": 27.7,
                "Derived_MAP": 105.0
            },
            "description": "Elderly Medium-Risk Patient"
        },
        # High risk patient (critical vitals)
        {
            "data": {
                "Heart_Rate": 125.0,
                "Respiratory_Rate": 28.0,
                "Body_Temperature": 102.0,
                "Oxygen_Saturation": 88,
                "Systolic_Blood_Pressure": 85,
                "Diastolic_Blood_Pressure": 45,
                "Age": 65.0,
                "Gender": 0,
                "Weight_kg": 55.0,
                "Height_m": 1.60,
                "Derived_HRV": 15.0,
                "Derived_Pulse_Pressure": 40.0,
                "Derived_BMI": 21.5,
                "Derived_MAP": 58.3
            },
            "description": "Critical High-Risk Patient"
        }
    ]
    
    # Add all patients
    results = []
    for patient in patients:
        result = add_patient(patient["data"], patient["description"])
        if result:
            results.append(result)
    
    print(f"\n📋 Added {len(results)} patients to queue")
    
    # Check queue order
    print("\n🔍 Checking queue order...")
    response = requests.get(f"{BASE_URL}/queue/")
    if response.status_code == 200:
        queue = response.json()
        print(f"Queue has {len(queue)} patients:")
        for i, patient in enumerate(queue, 1):
            print(f"   {i}. {patient['patient_id']} - {patient['risk_level']} (Priority: {patient['priority_score']:.1f})")
        
        # Verify high risk is first
        if queue and queue[0]['risk_level'] in ['High Risk', 'high']:
            print("✅ High-risk patient is correctly prioritized first")
        else:
            print("⚠️  Queue ordering may not be optimal")
    
    # Test getting next patient
    print("\n👨‍⚕️ Testing next patient selection...")
    response = requests.get(f"{BASE_URL}/queue/next/")
    if response.status_code == 200:
        next_patient = response.json()
        print(f"✅ Next patient: {next_patient['patient_id']} - {next_patient['risk_level']}")
        print(f"   Priority score: {next_patient['priority_score']:.1f}")
    else:
        print(f"❌ Failed to get next patient: {response.text}")
    
    # Check queue after removing patient
    print("\n📋 Checking queue after removing patient...")
    response = requests.get(f"{BASE_URL}/queue/")
    if response.status_code == 200:
        queue = response.json()
        print(f"Queue now has {len(queue)} patients")
    
    # Test feedback endpoint
    print("\n📝 Testing feedback system...")
    feedback_data = {
        "patient_id": "test_patient_123",
        "actual_wait_time": 30,
        "satisfaction_score": 0.85,
        "resource_utilization": 0.7
    }
    
    response = requests.post(f"{BASE_URL}/feedback/", params=feedback_data)
    if response.status_code == 200:
        result = response.json()
        print("✅ Feedback system working")
        print(f"   Message: {result['message']}")
    else:
        print(f"❌ Feedback system failed: {response.text}")

def main():
    """Run comprehensive tests"""
    print("🧪 Running Comprehensive API Tests\n")
    test_comprehensive_queue()
    print("\n🎯 Tests completed!")

if __name__ == "__main__":
    main()
