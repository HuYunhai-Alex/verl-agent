
from verl import DataProto
import torch
import numpy as np
from typing import Dict, List, Optional, Any
from enum import Enum
import re
from dataclasses import dataclass
from collections import Counter

class DoctorRole(Enum):
    """Doctor role enumeration for medical consultation"""
    SPECIALIST = "specialist"
    INTERNIST = "internist"
    RADIOLOGIST = "radiologist"
    ATTENDING = "attending"

@dataclass
class ConsultationMetrics:
    """Consultation evaluation metrics"""
    medical_accuracy: float = 0.0
    professional_depth: float = 0.0
    collaboration_quality: float = 0.0
    diagnostic_confidence: float = 0.0
    treatment_appropriateness: float = 0.0
    communication_clarity: float = 0.0

class MedicalConsultationRewardManager:
    """
    Multi-dimensional medical consultation reward manager.
    Evaluates consultation quality from multiple perspectives including medical accuracy,
    professional depth, collaboration quality, etc.
    """
    
    def __init__(
        self,
        tokenizer,
        num_examine: int = 5,
        normalize_by_length: bool = True,
        medical_accuracy: float = 0.3,
        professional_depth: float = 0.2,
        collaboration_quality: float = 0.2,
        diagnostic_confidence: float = 0.1,
        treatment_appropriateness: float = 0.1,
        communication_clarity: float = 0.1,
    ):
        """
        Initialize medical consultation reward manager
        
        Args:
            tokenizer: Tokenizer for text processing
            num_examine: Number of samples to examine and print
            normalize_by_length: Whether to normalize rewards by response length
            medical_accuracy_weight: Weight for medical accuracy evaluation
            professional_depth_weight: Weight for professional depth evaluation
            collaboration_quality_weight: Weight for collaboration quality evaluation
            diagnostic_confidence_weight: Weight for diagnostic confidence evaluation
            treatment_appropriateness_weight: Weight for treatment appropriateness evaluation
            communication_clarity_weight: Weight for communication clarity evaluation
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.normalize_by_length = normalize_by_length
        
        # Evaluation metric weights
        self.metric_weights = {
            'medical_accuracy': medical_accuracy,
            'professional_depth': professional_depth,
            'collaboration_quality': collaboration_quality,
            'diagnostic_confidence': diagnostic_confidence,
            'treatment_appropriateness': treatment_appropriateness,
            'communication_clarity': communication_clarity,
        }
        
        # Doctor role weights
        self.role_weights = {
            DoctorRole.SPECIALIST: 1.2,    # Specialist opinions carry more weight
            DoctorRole.INTERNIST: 1.0,     # Standard weight
            DoctorRole.RADIOLOGIST: 1.1,   # Imaging expertise is important
            DoctorRole.ATTENDING: 1.3,     # Attending has highest authority
        }
        
        # Medical terminology dictionary (can be expanded)
        self.medical_terms = {
            'diagnosis', 'treatment', 'symptom', 'syndrome', 'pathology', 'etiology',
            'prognosis', 'therapy', 'medication', 'dosage', 'contraindication',
            'differential', 'clinical', 'laboratory', 'imaging', 'radiology',
            'histology', 'biopsy', 'surgery', 'intervention', 'monitoring',
            'follow-up', 'complications', 'adverse', 'side effects', 'efficacy'
        }
        
        # Collaboration keywords
        self.collaboration_keywords = {
            'agree', 'disagree', 'suggest', 'recommend', 'consider', 'opinion',
            'perspective', 'experience', 'discussion', 'consultation', 'consensus',
            'alternative', 'additional', 'furthermore', 'however', 'although'
        }
    
    def __call__(self, data: Dict[str, Any]) -> torch.Tensor:
        """
        Calculate comprehensive rewards for medical consultation
        
        Args:
            data: Consultation data containing prompts, responses, and metadata
            
        Returns:
            torch.Tensor: Calculated rewards for each consultation turn
        """
        # Extract consultation data
        prompts = data.get('prompts', [])
        responses = data.get('responses', [])
        doctor_roles = data.get('doctor_role', [])
        
        if not prompts or not responses:
            return torch.zeros(1)
        
        # Decode responses if they are token IDs
        if isinstance(responses, torch.Tensor):
            decoded_responses = self.tokenizer.batch_decode(responses, skip_special_tokens=True)
        else:
            decoded_responses = responses
        
        # Calculate rewards for each consultation turn
        rewards = []
        
        for i, (prompt, response, doctor_role) in enumerate(zip(prompts, decoded_responses, doctor_roles)):
            # Extract doctor contributions
            doctor_contributions = self._extract_doctor_contributions(prompt, response)
            
            # Evaluate consultation metrics
            metrics = self._evaluate_consultation_metrics(
                prompt=prompt,
                response=response,
                doctor_role=doctor_role,
                doctor_contributions=doctor_contributions
            )
            
            # Calculate comprehensive reward
            reward = self._compute_comprehensive_reward(metrics, doctor_role)
            rewards.append(reward)
            
            # Print examination samples
            if i < self.num_examine:
                self._print_examination_sample(i, prompt, response, doctor_role, metrics, reward)
        
        # Convert to tensor and normalize if required
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        
        if self.normalize_by_length:
            response_lengths = torch.tensor([len(resp.split()) for resp in decoded_responses])
            rewards_tensor = rewards_tensor / torch.clamp(response_lengths, min=1.0)
        
        return rewards_tensor
    
    def _extract_doctor_contributions(self, prompt: str, response: str) -> Dict[str, List[str]]:
        """
        Extract contributions from different doctors in the consultation
        
        Args:
            prompt: Input prompt containing consultation history
            response: Current response
            
        Returns:
            Dict mapping doctor roles to their contributions
        """
        contributions = {role.value: [] for role in DoctorRole}
        
        # Extract from prompt (consultation history)
        for role in DoctorRole:
            role_pattern = rf"{role.value.title()}:\s*([^:]+?)(?=\n[A-Z]|\n\n|$)"
            matches = re.findall(role_pattern, prompt, re.IGNORECASE | re.DOTALL)
            contributions[role.value].extend([match.strip() for match in matches])
        
        # Current response is from the current doctor
        # Extract doctor role from response context if available
        current_role = self._identify_current_doctor_role(prompt, response)
        if current_role:
            contributions[current_role].append(response.strip())
        
        return contributions
    
    def _identify_current_doctor_role(self, prompt: str, response: str) -> Optional[str]:
        """
        Identify the current doctor role from context
        
        Args:
            prompt: Input prompt
            response: Current response
            
        Returns:
            Doctor role string or None
        """
        # Look for role indicators in the prompt
        for role in DoctorRole:
            role_indicators = [
                f"As a {role.value}",
                f"As an {role.value}" if role.value.startswith(('a', 'e', 'i', 'o', 'u')) else f"As a {role.value}",
                f"{role.value.title()}:",
            ]
            
            for indicator in role_indicators:
                if indicator.lower() in prompt.lower():
                    return role.value
        
        return None
    
    def _evaluate_consultation_metrics(
        self,
        prompt: str,
        response: str,
        doctor_role: str,
        doctor_contributions: Dict[str, List[str]]
    ) -> ConsultationMetrics:
        """
        Evaluate consultation metrics for a single turn
        
        Args:
            prompt: Input prompt
            response: Current response
            doctor_role: Current doctor role
            doctor_contributions: All doctor contributions
            
        Returns:
            ConsultationMetrics with evaluated scores
        """
        metrics = ConsultationMetrics()
        
        # Medical accuracy evaluation
        metrics.medical_accuracy = self._evaluate_medical_accuracy(response, doctor_role)
        
        # Professional depth evaluation
        metrics.professional_depth = self._evaluate_professional_depth(response, doctor_role)
        
        # Collaboration quality evaluation
        metrics.collaboration_quality = self._evaluate_collaboration_quality(
            response, doctor_contributions
        )
        
        # Diagnostic confidence evaluation
        metrics.diagnostic_confidence = self._evaluate_diagnostic_confidence(response)
        
        # Treatment appropriateness evaluation
        metrics.treatment_appropriateness = self._evaluate_treatment_appropriateness(response)
        
        # Communication clarity evaluation
        metrics.communication_clarity = self._evaluate_communication_clarity(response)
        
        return metrics
    
    def _evaluate_medical_accuracy(self, response: str, doctor_role: str) -> float:
        """
        Evaluate medical accuracy of the response
        
        Args:
            response: Doctor's response
            doctor_role: Doctor's role
            
        Returns:
            Medical accuracy score (0-1)
        """
        score = 0.0
        
        # Count medical terminology usage
        medical_term_count = self._count_medical_terms(response)
        score += min(medical_term_count * 0.1, 0.4)
        
        # Evaluate logical structure
        logical_score = self._evaluate_logical_structure(response)
        score += logical_score * 0.3
        
        # Check for evidence-based references
        evidence_score = self._evaluate_evidence_references(response)
        score += evidence_score * 0.3
        
        return min(score, 1.0)
    
    def _evaluate_professional_depth(self, response: str, doctor_role: str) -> float:
        """
        Evaluate professional depth based on doctor role
        
        Args:
            response: Doctor's response
            doctor_role: Doctor's role
            
        Returns:
            Professional depth score (0-1)
        """
        score = 0.0
        
        # Role-specific depth evaluation
        if doctor_role == DoctorRole.SPECIALIST.value:
            score += self._evaluate_specialist_depth(response)
        elif doctor_role == DoctorRole.INTERNIST.value:
            score += self._evaluate_internist_depth(response)
        elif doctor_role == DoctorRole.RADIOLOGIST.value:
            score += self._evaluate_radiologist_depth(response)
        elif doctor_role == DoctorRole.ATTENDING.value:
            score += self._evaluate_attending_depth(response)
        
        # General depth indicators
        score += self._evaluate_general_depth(response)
        
        return min(score, 1.0)
    
    def _evaluate_collaboration_quality(
        self,
        response: str,
        doctor_contributions: Dict[str, List[str]]
    ) -> float:
        """
        Evaluate collaboration quality
        
        Args:
            response: Current response
            doctor_contributions: All doctor contributions
            
        Returns:
            Collaboration quality score (0-1)
        """
        score = 0.0
        
        # Check for collaboration keywords
        collaboration_count = sum(
            1 for keyword in self.collaboration_keywords
            if keyword.lower() in response.lower()
        )
        score += min(collaboration_count * 0.1, 0.4)
        
        # Evaluate references to other doctors' opinions
        reference_score = self._evaluate_doctor_references(response, doctor_contributions)
        score += reference_score * 0.3
        
        # Evaluate constructive discussion
        constructive_score = self._evaluate_constructive_discussion(response)
        score += constructive_score * 0.3
        
        return min(score, 1.0)
    
    def _evaluate_diagnostic_confidence(self, response: str) -> float:
        """
        Evaluate diagnostic confidence
        
        Args:
            response: Doctor's response
            
        Returns:
            Diagnostic confidence score (0-1)
        """
        score = 0.0
        
        # Look for confidence indicators
        confidence_indicators = [
            'likely', 'probable', 'suggest', 'indicate', 'consistent with',
            'differential diagnosis', 'most likely', 'rule out', 'consider'
        ]
        
        confidence_count = sum(
            1 for indicator in confidence_indicators
            if indicator.lower() in response.lower()
        )
        score += min(confidence_count * 0.15, 0.6)
        
        # Penalize excessive uncertainty
        uncertainty_indicators = ['unsure', 'unclear', 'difficult to say', 'hard to determine']
        uncertainty_count = sum(
            1 for indicator in uncertainty_indicators
            if indicator.lower() in response.lower()
        )
        score -= uncertainty_count * 0.1
        
        return max(min(score, 1.0), 0.0)
    
    def _evaluate_treatment_appropriateness(self, response: str) -> float:
        """
        Evaluate treatment appropriateness
        
        Args:
            response: Doctor's response
            
        Returns:
            Treatment appropriateness score (0-1)
        """
        score = 0.0
        
        # Check for treatment mentions
        treatment_keywords = [
            'treatment', 'therapy', 'medication', 'drug', 'surgery', 'intervention',
            'management', 'care', 'follow-up', 'monitoring', 'dosage'
        ]
        
        treatment_count = sum(
            1 for keyword in treatment_keywords
            if keyword.lower() in response.lower()
        )
        score += min(treatment_count * 0.1, 0.5)
        
        # Evaluate personalization
        personalization_score = self._evaluate_personalization(response)
        score += personalization_score * 0.5
        
        return min(score, 1.0)
    
    def _evaluate_communication_clarity(self, response: str) -> float:
        """
        Evaluate communication clarity
        
        Args:
            response: Doctor's response
            
        Returns:
            Communication clarity score (0-1)
        """
        score = 0.0
        
        # Evaluate text structure
        structure_score = self._evaluate_text_structure(response)
        score += structure_score * 0.4
        
        # Evaluate readability
        readability_score = self._evaluate_readability(response)
        score += readability_score * 0.3
        
        # Penalize excessive complexity
        complexity_penalty = self._evaluate_complexity_penalty(response)
        score -= complexity_penalty * 0.2
        
        # Evaluate coherence
        coherence_score = self._evaluate_coherence(response)
        score += coherence_score * 0.3
        
        return max(min(score, 1.0), 0.0)
    
    def _compute_comprehensive_reward(self, metrics: ConsultationMetrics, doctor_role: str) -> float:
        """
        Compute comprehensive reward based on all metrics
        
        Args:
            metrics: Evaluated consultation metrics
            doctor_role: Doctor role
            
        Returns:
            Comprehensive reward score
        """
        # Calculate weighted score
        weighted_score = (
            metrics.medical_accuracy * self.metric_weights['medical_accuracy'] +
            metrics.professional_depth * self.metric_weights['professional_depth'] +
            metrics.collaboration_quality * self.metric_weights['collaboration_quality'] +
            metrics.diagnostic_confidence * self.metric_weights['diagnostic_confidence'] +
            metrics.treatment_appropriateness * self.metric_weights['treatment_appropriateness'] +
            metrics.communication_clarity * self.metric_weights['communication_clarity']
        )
        
        # Apply role weight
        role_enum = DoctorRole(doctor_role) if doctor_role in [r.value for r in DoctorRole] else DoctorRole.INTERNIST
        role_weight = self.role_weights.get(role_enum, 1.0)
        
        final_reward = weighted_score * role_weight
        
        return final_reward
    
    def _print_examination_sample(
        self,
        index: int,
        prompt: str,
        response: str,
        doctor_role: str,
        metrics: ConsultationMetrics,
        reward: float
    ):
        """Print examination sample for debugging"""
        print(f"\n=== Medical Consultation Examination Sample {index + 1} ===")
        print(f"Doctor Role: {doctor_role}")
        print(f"Prompt: {prompt[:200]}...")
        print(f"Response: {response[:200]}...")
        print(f"Metrics:")
        print(f"  Medical Accuracy: {metrics.medical_accuracy:.3f}")
        print(f"  Professional Depth: {metrics.professional_depth:.3f}")
        print(f"  Collaboration Quality: {metrics.collaboration_quality:.3f}")
        print(f"  Diagnostic Confidence: {metrics.diagnostic_confidence:.3f}")
        print(f"  Treatment Appropriateness: {metrics.treatment_appropriateness:.3f}")
        print(f"  Communication Clarity: {metrics.communication_clarity:.3f}")
        print(f"Final Reward: {reward:.3f}")
        print("=" * 60)
    
    # Helper methods for specific evaluations
    def _count_medical_terms(self, text: str) -> int:
        """Count medical terms in text"""
        words = set(text.lower().split())
        return len(words.intersection(self.medical_terms))
    
    def _evaluate_logical_structure(self, text: str) -> float:
        """Evaluate logical structure of text"""
        # Simple heuristic: check for logical connectors
        logical_connectors = ['therefore', 'because', 'since', 'thus', 'consequently', 'as a result']
        connector_count = sum(1 for connector in logical_connectors if connector in text.lower())
        return min(connector_count * 0.2, 1.0)
    
    def _evaluate_evidence_references(self, text: str) -> float:
        """Evaluate evidence-based references"""
        evidence_indicators = ['study', 'research', 'evidence', 'literature', 'guidelines', 'protocol']
        evidence_count = sum(1 for indicator in evidence_indicators if indicator in text.lower())
        return min(evidence_count * 0.2, 1.0)
    
    def _evaluate_specialist_depth(self, text: str) -> float:
        """Evaluate specialist-specific depth"""
        specialist_terms = ['pathophysiology', 'etiology', 'mechanism', 'specialized', 'advanced']
        term_count = sum(1 for term in specialist_terms if term in text.lower())
        return min(term_count * 0.2, 1.0)
    
    def _evaluate_internist_depth(self, text: str) -> float:
        """Evaluate internist-specific depth"""
        internist_terms = ['systemic', 'comprehensive', 'holistic', 'comorbidity', 'multisystem']
        term_count = sum(1 for term in internist_terms if term in text.lower())
        return min(term_count * 0.2, 1.0)
    
    def _evaluate_radiologist_depth(self, text: str) -> float:
        """Evaluate radiologist-specific depth"""
        radiology_terms = ['imaging', 'scan', 'contrast', 'anatomy', 'morphology', 'findings']
        term_count = sum(1 for term in radiology_terms if term in text.lower())
        return min(term_count * 0.2, 1.0)
    
    def _evaluate_attending_depth(self, text: str) -> float:
        """Evaluate attending-specific depth"""
        attending_terms = ['management', 'coordination', 'supervision', 'decision', 'leadership']
        term_count = sum(1 for term in attending_terms if term in text.lower())
        return min(term_count * 0.2, 1.0)
    
    def _evaluate_general_depth(self, text: str) -> float:
        """Evaluate general professional depth"""
        depth_indicators = ['detailed', 'thorough', 'comprehensive', 'analysis', 'evaluation']
        indicator_count = sum(1 for indicator in depth_indicators if indicator in text.lower())
        return min(indicator_count * 0.1, 0.5)
    
    def _evaluate_doctor_references(self, text: str, contributions: Dict[str, List[str]]) -> float:
        """Evaluate references to other doctors' opinions"""
        reference_score = 0.0
        for role in DoctorRole:
            if role.value in text.lower() or role.value.title() in text:
                reference_score += 0.25
        return min(reference_score, 1.0)
    
    def _evaluate_constructive_discussion(self, text: str) -> float:
        """Evaluate constructive discussion elements"""
        constructive_terms = ['build on', 'add to', 'complement', 'expand', 'clarify', 'support']
        term_count = sum(1 for term in constructive_terms if term in text.lower())
        return min(term_count * 0.2, 1.0)
    
    def _evaluate_personalization(self, text: str) -> float:
        """Evaluate treatment personalization"""
        personalization_terms = ['patient', 'individual', 'specific', 'tailored', 'customized']
        term_count = sum(1 for term in personalization_terms if term in text.lower())
        return min(term_count * 0.2, 1.0)
    
    def _evaluate_text_structure(self, text: str) -> float:
        """Evaluate text structure and organization"""
        sentences = text.split('.')
        if len(sentences) < 2:
            return 0.3
        elif len(sentences) < 5:
            return 0.7
        else:
            return 1.0
    
    def _evaluate_readability(self, text: str) -> float:
        """Evaluate text readability"""
        words = text.split()
        if not words:
            return 0.0
        
        avg_word_length = sum(len(word) for word in words) / len(words)
        # Prefer moderate word length (not too simple, not too complex)
        if 4 <= avg_word_length <= 7:
            return 1.0
        elif 3 <= avg_word_length <= 8:
            return 0.7
        else:
            return 0.4
    
    def _evaluate_complexity_penalty(self, text: str) -> float:
        """Evaluate complexity penalty for overly complex text"""
        complex_terms = ['notwithstanding', 'nevertheless', 'furthermore', 'consequently']
        complex_count = sum(1 for term in complex_terms if term in text.lower())
        return min(complex_count * 0.1, 0.5)
    
    def _evaluate_coherence(self, text: str) -> float:
        """Evaluate text coherence"""
        # Simple heuristic: check for transition words
        transition_words = ['first', 'second', 'next', 'then', 'finally', 'in conclusion']
        transition_count = sum(1 for word in transition_words if word in text.lower())
        return min(transition_count * 0.2, 1.0)
    
    def _evaluate_uniqueness_penalty(self, text: str) -> float:
        """Evaluate penalty for repetitive content"""
        words = text.lower().split()
        if not words:
            return 0.0
        
        word_counts = Counter(words)
        repeated_words = sum(1 for count in word_counts.values() if count > 2)
        repetition_ratio = repeated_words / len(set(words))
        
        return min(repetition_ratio, 0.3) 