"""
Reinforcement Learning Scheduler for Patient Priority Management
This module implements a simple Q-learning algorithm to optimize scheduling
for low-risk patients while maintaining priority for high-risk patients.
"""

import numpy as np
import json
import os
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
from models import RiskAssessment, PatientQueue

class PatientSchedulerRL:
    """
    RL-based scheduler that learns optimal scheduling policies for different risk levels
    """
    
    def __init__(self, learning_rate=0.1, discount_factor=0.95, epsilon=0.1, epsilon_decay=0.995):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = 0.01
        
        # Q-table: state -> action -> value
        self.q_table: Dict[str, Dict[str, float]] = {}
        
        # Action space for low-risk patients
        self.actions = {
            'immediate': 0,      # Schedule immediately
            'delay_15': 1,       # Delay 15 minutes
            'delay_30': 2,       # Delay 30 minutes
            'delay_60': 3,       # Delay 60 minutes
            'delay_120': 4       # Delay 2 hours
        }
        
        # Load existing Q-table if available
        self.load_q_table()
    
    def get_state(self, patient: PatientQueue, queue_info: Dict) -> str:
        """
        Generate state representation for RL
        Args:
            patient: PatientQueue object
            queue_info: Dictionary with current queue information
        Returns:
            str: State representation
        """
        risk_level = patient.risk_assessment.risk_level.lower()
        confidence_bucket = int(patient.risk_assessment.confidence_score * 10)  # 0-10
        queue_length = min(queue_info.get('total_patients', 0), 10)  # Cap at 10
        high_risk_count = min(queue_info.get('high_risk_count', 0), 5)  # Cap at 5
        hour = datetime.now().hour
        time_bucket = 'morning' if 6 <= hour < 12 else 'afternoon' if 12 <= hour < 18 else 'evening'
        
        state = f"{risk_level}_{confidence_bucket}_{queue_length}_{high_risk_count}_{time_bucket}"
        return state
    
    def choose_action(self, state: str) -> str:
        """
        Choose action using epsilon-greedy policy
        Args:
            state: State representation
        Returns:
            str: Action to take
        """
        if state not in self.q_table:
            self.q_table[state] = {action: 0.0 for action in self.actions.keys()}
        
        # Epsilon-greedy action selection
        if np.random.random() < self.epsilon:
            # Exploration: random action
            return np.random.choice(list(self.actions.keys()))
        else:
            # Exploitation: best action
            return max(self.q_table[state], key=self.q_table[state].get)
    
    def get_reward(self, action: str, patient: PatientQueue, outcome: Dict) -> float:
        """
        Calculate reward based on action taken and outcome
        Args:
            action: Action that was taken
            patient: PatientQueue object
            outcome: Dictionary with outcome metrics
        Returns:
            float: Reward value
        """
        base_reward = 0.0
        
        # Reward components
        wait_time_actual = outcome.get('actual_wait_time', 0)
        patient_satisfaction = outcome.get('satisfaction_score', 0.5)  # 0-1
        resource_utilization = outcome.get('resource_utilization', 0.5)  # 0-1
        
        # Penalty for excessive wait times
        if wait_time_actual > 120:  # More than 2 hours
            base_reward -= 10.0
        elif wait_time_actual > 60:  # More than 1 hour
            base_reward -= 5.0
        elif wait_time_actual > 30:  # More than 30 minutes
            base_reward -= 2.0
        
        # Reward for patient satisfaction
        base_reward += patient_satisfaction * 10.0
        
        # Reward for efficient resource utilization
        base_reward += resource_utilization * 5.0
        
        # Bonus for appropriate scheduling of low-risk patients
        if patient.risk_assessment.risk_level.lower() == "low":
            if action in ['delay_30', 'delay_60'] and patient_satisfaction > 0.7:
                base_reward += 3.0  # Good balance
            elif action == 'immediate' and resource_utilization < 0.3:
                base_reward -= 2.0  # Wasted resources
        
        return base_reward
    
    def update_q_value(self, state: str, action: str, reward: float, next_state: str):
        """
        Update Q-value using Q-learning update rule
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
        """
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in self.actions.keys()}
        
        if next_state not in self.q_table:
            self.q_table[next_state] = {a: 0.0 for a in self.actions.keys()}
        
        # Q-learning update
        current_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state].values())
        
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[state][action] = new_q
        
        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
    
    def apply_action_to_patient(self, action: str, patient: PatientQueue) -> PatientQueue:
        """
        Apply the chosen action to modify patient scheduling
        Args:
            action: Action to apply
            patient: PatientQueue object to modify
        Returns:
            PatientQueue: Modified patient object
        """
        action_effects = {
            'immediate': {'delay': 0, 'priority_boost': 10.0},
            'delay_15': {'delay': 15, 'priority_boost': 2.0},
            'delay_30': {'delay': 30, 'priority_boost': 0.0},
            'delay_60': {'delay': 60, 'priority_boost': -3.0},
            'delay_120': {'delay': 120, 'priority_boost': -8.0}
        }
        
        effect = action_effects.get(action, action_effects['delay_30'])
        
        # Apply delay to estimated wait time
        patient.estimated_wait_time += effect['delay']
        
        # Apply RL adjustment to priority
        patient.rl_adjustment = effect['priority_boost']
        
        return patient
    
    def schedule_patients_with_rl(self, patients: List[PatientQueue], queue_info: Dict) -> List[PatientQueue]:
        """
        Schedule patients using RL for low-risk patients
        Args:
            patients: List of PatientQueue objects
            queue_info: Current queue information
        Returns:
            List[PatientQueue]: Scheduled patients
        """
        scheduled_patients = []
        
        for patient in patients:
            if patient.risk_assessment.risk_level.lower() == "low":
                # Use RL for low-risk patients
                state = self.get_state(patient, queue_info)
                action = self.choose_action(state)
                patient = self.apply_action_to_patient(action, patient)
                
                # Store state-action for potential learning update
                patient.rl_state = state
                patient.rl_action = action
            
            scheduled_patients.append(patient)
        
        # Sort by final priority (including RL adjustments)
        scheduled_patients.sort(key=lambda p: p.get_final_priority(), reverse=True)
        
        # Update queue positions
        for i, patient in enumerate(scheduled_patients):
            patient.queue_position = i + 1
        
        return scheduled_patients
    
    def provide_feedback(self, patient_id: str, outcome: Dict):
        """
        Provide feedback to the RL system for learning
        Args:
            patient_id: Patient identifier
            outcome: Dictionary with outcome metrics
        """
        # In a real implementation, you would track state-action pairs
        # and update Q-values based on outcomes
        # This is a simplified version
        pass
    
    def save_q_table(self, filename: str = "q_table.json"):
        """Save Q-table to file"""
        filepath = os.path.join(os.path.dirname(__file__), filename)
        with open(filepath, 'w') as f:
            json.dump(self.q_table, f, indent=2)
    
    def load_q_table(self, filename: str = "q_table.json"):
        """Load Q-table from file"""
        filepath = os.path.join(os.path.dirname(__file__), filename)
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    self.q_table = json.load(f)
            except Exception as e:
                print(f"Error loading Q-table: {e}")
                self.q_table = {}

class PriorityManager:
    """
    Main priority management system that combines rule-based and RL-based scheduling
    """
    
    def __init__(self):
        self.rl_scheduler = PatientSchedulerRL()
        self.patient_history = []
    
    def calculate_dynamic_priority(self, patients: List[PatientQueue]) -> List[PatientQueue]:
        """
        Calculate dynamic priorities for all patients
        Args:
            patients: List of PatientQueue objects
        Returns:
            List[PatientQueue]: Patients with updated priorities
        """
        # Prepare queue information for RL
        queue_info = {
            'total_patients': len(patients),
            'high_risk_count': len([p for p in patients if p.risk_assessment.risk_level.lower() == "high"]),
            'medium_risk_count': len([p for p in patients if p.risk_assessment.risk_level.lower() == "medium"]),
            'low_risk_count': len([p for p in patients if p.risk_assessment.risk_level.lower() == "low"]),
            'average_confidence': np.mean([p.risk_assessment.confidence_score for p in patients]) if patients else 0.0
        }
        
        # Update time factors for all patients
        for patient in patients:
            patient.risk_assessment.calculate_priority_score()
        
        # Apply RL scheduling
        scheduled_patients = self.rl_scheduler.schedule_patients_with_rl(patients, queue_info)
        
        return scheduled_patients
    
    def get_next_patient(self, patients: List[PatientQueue]) -> PatientQueue:
        """
        Get the next patient to be seen
        Args:
            patients: List of PatientQueue objects
        Returns:
            PatientQueue: Next patient to be seen
        """
        if not patients:
            return None
        
        # Update priorities
        updated_patients = self.calculate_dynamic_priority(patients)
        
        # Return highest priority patient
        return updated_patients[0] if updated_patients else None
    
    def update_with_outcome(self, patient_id: str, outcome: Dict):
        """
        Update the system with outcome feedback
        Args:
            patient_id: Patient identifier
            outcome: Outcome metrics
        """
        self.rl_scheduler.provide_feedback(patient_id, outcome)
        self.patient_history.append({
            'patient_id': patient_id,
            'outcome': outcome,
            'timestamp': datetime.now()
        })
        
        # Periodically save the Q-table
        if len(self.patient_history) % 10 == 0:
            self.rl_scheduler.save_q_table()
