import torch
import numpy as np
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from transformers import PreTrainedTokenizer
import uuid
from typing import List, Dict, Any, Optional
from agent_system.multi_turn_rollout.utils import process_image, to_list_of_dict, torch_to_numpy, filter_group_data
from agent_system.environments import EnvironmentManagerBase
from enum import Enum
from dataclasses import dataclass
import random

class DoctorRole(Enum):
    """Doctor role enumeration for medical consultation"""
    SPECIALIST = "specialist"  
    INTERNIST = "internist"   
    RADIOLOGIST = "radiologist"  
    ATTENDING = "attending"    # Attending Physician

@dataclass
class ConsultationTurn:
    """Single turn in medical consultation"""
    doctor_role: DoctorRole
    input_text: str
    response_text: str
    turn_index: int
    consultation_round: int

class MedicalConsultationCollector:
    """
    Medical consultation collector for managing multi-doctor consultation process.
    Supports multi-turn dialogue between different doctor roles.
    """
    
    def __init__(
        self,
        tokenizer,
        max_consultation_rounds: int = 3,
        max_turn_length: int = 256,
        enable_discussion: bool = True,
        discussion_turns: int = 2,
    ):
        """
        Initialize medical consultation collector
        
        Args:
            tokenizer: Tokenizer for text processing
            max_consultation_rounds: Maximum number of consultation rounds
            max_turn_length: Maximum length for each turn
            enable_discussion: Whether to enable discussion between doctors
            discussion_turns: Number of discussion turns
        """
        self.tokenizer = tokenizer
        self.max_consultation_rounds = max_consultation_rounds
        self.max_turn_length = max_turn_length
        self.enable_discussion = enable_discussion
        self.discussion_turns = discussion_turns
        
        # Define doctor consultation order
        self.doctor_order = [
            DoctorRole.SPECIALIST,
            DoctorRole.INTERNIST,
            DoctorRole.RADIOLOGIST,
            DoctorRole.ATTENDING,
        ]
        
        # Doctor role prompts
        self.role_prompts = {
            DoctorRole.SPECIALIST: "As a specialist, please provide your professional opinion on this case:",
            DoctorRole.INTERNIST: "As an internist, please analyze this case from internal medicine perspective:",
            DoctorRole.RADIOLOGIST: "As a radiologist, please interpret the imaging findings and provide insights:",
            DoctorRole.ATTENDING: "As the attending physician, please synthesize all opinions and provide final recommendations:",
        }
    
    def preprocess_sample_for_doctor(
        self, 
        sample: Dict[str, Any], 
        doctor_role: DoctorRole,
        consultation_history: List[ConsultationTurn] = None
    ) -> Dict[str, Any]:
        """
        Preprocess sample for specific doctor role
        
        Args:
            sample: Input sample data
            doctor_role: Target doctor role
            consultation_history: Previous consultation history
            
        Returns:
            Preprocessed sample for the doctor
        """
        # Build consultation context
        context_parts = []
        
        # Add original case information
        if 'prompt' in sample:
            context_parts.append(f"Medical Case:\n{sample['prompt']}")
        
        # Add previous consultation history
        if consultation_history:
            context_parts.append("\nPrevious Consultation:")
            for turn in consultation_history:
                context_parts.append(
                    f"\n{turn.doctor_role.value.title()}: {turn.response_text}"
                )
        
        # Add role-specific prompt
        role_prompt = self.role_prompts.get(doctor_role, "Please provide your medical opinion:")
        context_parts.append(f"\n{role_prompt}")
        
        # Combine all context
        full_context = "\n".join(context_parts)
        
        # Create preprocessed sample
        preprocessed_sample = sample.copy()
        preprocessed_sample.update({
            'doctor_role': doctor_role.value,
            'consultation_context': full_context,
            'role_prompt': role_prompt,
        })
        
        return preprocessed_sample
    
    def multi_turn_loop(
        self,
        gen_batch: DataProto,
        doctor_worker_groups: Dict[DoctorRole, Any],
        envs: Optional[Any] = None,
        is_train: bool = True,
    ) -> DataProto:
        """
        Execute multi-turn medical consultation loop
        
        Args:
            gen_batch: Generation batch data
            doctor_worker_groups: Dictionary of doctor worker groups
            envs: Environment instances (optional)
            is_train: Whether in training mode
            
        Returns:
            Consultation results as DataProto
        """
        print(f"Starting medical consultation with {len(gen_batch)} cases...")
        
        # Initialize consultation data
        consultation_data = []
        
        # Process each sample in the batch
        for sample_idx in range(len(gen_batch)):
            sample = gen_batch[sample_idx].to_single_dict()
            print(f"Processing case {sample_idx + 1}/{len(gen_batch)}")
            
            # Execute consultation for this sample
            sample_consultation = self.multi_doctor_consultation_loop(
                sample=sample,
                doctor_worker_groups=doctor_worker_groups,
                is_train=is_train
            )
            
            consultation_data.extend(sample_consultation)
        
        # Gather and return consultation data
        return self.gather_consultation_data(consultation_data)
    
    def multi_doctor_consultation_loop(
        self,
        sample: Dict[str, Any],
        doctor_worker_groups: Dict[DoctorRole, Any],
        is_train: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Execute consultation loop for a single medical case
        
        Args:
            sample: Single medical case sample
            doctor_worker_groups: Dictionary of doctor worker groups
            is_train: Whether in training mode
            
        Returns:
            List of consultation turn results
        """
        consultation_results = []
        consultation_history = []
        
        # Execute consultation rounds
        for round_idx in range(self.max_consultation_rounds):
            print(f"  Consultation round {round_idx + 1}/{self.max_consultation_rounds}")
            
            # Each doctor provides their opinion
            for doctor_role in self.doctor_order:
                if doctor_role not in doctor_worker_groups:
                    print(f"  Warning: {doctor_role.value} worker group not available")
                    continue
                
                print(f"    {doctor_role.value.title()} consultation...")
                
                # Preprocess sample for current doctor
                doctor_sample = self.preprocess_sample_for_doctor(
                    sample=sample,
                    doctor_role=doctor_role,
                    consultation_history=consultation_history
                )
                
                # Convert to DataProto for worker group
                doctor_batch = DataProto.from_single_dict(data=doctor_sample)
                
                # Generate response from doctor
                worker_group = doctor_worker_groups[doctor_role]
                
                if is_train:
                    # Use generate_sequences for training
                    response_output = worker_group.generate_sequences(doctor_batch)
                else:
                    # Use generate_sequences for evaluation
                    response_output = worker_group.generate_sequences(doctor_batch)
                
                # Extract response
                response_data = response_output.batch
                response_text = self.tokenizer.decode(
                    response_data['responses'][0], 
                    skip_special_tokens=True
                )
                
                # Create consultation turn
                consultation_turn = ConsultationTurn(
                    doctor_role=doctor_role,
                    input_text=doctor_sample['consultation_context'],
                    response_text=response_text,
                    turn_index=len(consultation_history),
                    consultation_round=round_idx
                )
                
                consultation_history.append(consultation_turn)
                
                # Prepare result data
                result_data = {
                    'case_id': sample.get('case_id', f'case_{hash(str(sample))}'),
                    'doctor_role': doctor_role.value,
                    'consultation_round': round_idx,
                    'turn_index': consultation_turn.turn_index,
                    'input_text': consultation_turn.input_text,
                    'response_text': response_text,
                    'prompts': doctor_sample['consultation_context'],
                    'responses': response_data['responses'][0],
                    'response_length': len(response_data['responses'][0]),
                }
                
                # Add additional data from response_output
                if hasattr(response_output, 'meta_info') and response_output.meta_info:
                    result_data.update(response_output.meta_info)
                
                consultation_results.append(result_data)
                
                print(f"    {doctor_role.value.title()} completed consultation")
            
            # Optional discussion phase
            if self.enable_discussion and round_idx < self.max_consultation_rounds - 1:
                print(f"  Discussion phase for round {round_idx + 1}")
                self._conduct_discussion(
                    sample=sample,
                    consultation_history=consultation_history,
                    doctor_worker_groups=doctor_worker_groups,
                    consultation_results=consultation_results,
                    round_idx=round_idx,
                    is_train=is_train
                )
        
        print(f"  Consultation completed with {len(consultation_results)} turns")
        return consultation_results
    
    def _conduct_discussion(
        self,
        sample: Dict[str, Any],
        consultation_history: List[ConsultationTurn],
        doctor_worker_groups: Dict[DoctorRole, Any],
        consultation_results: List[Dict[str, Any]],
        round_idx: int,
        is_train: bool = True,
    ):
        """
        Conduct discussion phase between doctors
        
        Args:
            sample: Medical case sample
            consultation_history: Current consultation history
            doctor_worker_groups: Dictionary of doctor worker groups
            consultation_results: Current consultation results
            round_idx: Current round index
            is_train: Whether in training mode
        """
        # Randomly select doctors for discussion
        available_doctors = [role for role in self.doctor_order if role in doctor_worker_groups]
        discussion_doctors = random.sample(
            available_doctors, 
            min(2, len(available_doctors))
        )
        
        for discussion_turn in range(self.discussion_turns):
            for doctor_role in discussion_doctors:
                # Create discussion prompt
                discussion_prompt = self._create_discussion_prompt(
                    sample=sample,
                    consultation_history=consultation_history,
                    doctor_role=doctor_role,
                    discussion_turn=discussion_turn
                )
                
                # Generate discussion response
                doctor_sample = {
                    **sample,
                    'doctor_role': doctor_role.value,
                    'consultation_context': discussion_prompt,
                    'is_discussion': True,
                }
                
                doctor_batch = DataProto.from_single_dict(data=doctor_sample)
                worker_group = doctor_worker_groups[doctor_role]
                
                response_output = worker_group.generate_sequences(doctor_batch)
                response_data = response_output.batch
                response_text = self.tokenizer.decode(
                    response_data['responses'][0], 
                    skip_special_tokens=True
                )
                
                # Add discussion turn to history
                discussion_consultation_turn = ConsultationTurn(
                    doctor_role=doctor_role,
                    input_text=discussion_prompt,
                    response_text=response_text,
                    turn_index=len(consultation_history),
                    consultation_round=round_idx
                )
                
                consultation_history.append(discussion_consultation_turn)
                
                # Add to results
                result_data = {
                    'case_id': sample.get('case_id', f'case_{hash(str(sample))}'),
                    'doctor_role': doctor_role.value,
                    'consultation_round': round_idx,
                    'turn_index': discussion_consultation_turn.turn_index,
                    'input_text': discussion_prompt,
                    'response_text': response_text,
                    'prompts': discussion_prompt,
                    'responses': response_data['responses'][0],
                    'response_length': len(response_data['responses'][0]),
                    'is_discussion': True,
                }
                
                consultation_results.append(result_data)
    
    def _create_discussion_prompt(
        self,
        sample: Dict[str, Any],
        consultation_history: List[ConsultationTurn],
        doctor_role: DoctorRole,
        discussion_turn: int,
    ) -> str:
        """
        Create discussion prompt for doctor
        
        Args:
            sample: Medical case sample
            consultation_history: Current consultation history
            doctor_role: Current doctor role
            discussion_turn: Discussion turn number
            
        Returns:
            Discussion prompt string
        """
        context_parts = []
        
        # Add case information
        if 'prompt' in sample:
            context_parts.append(f"Medical Case:\n{sample['prompt']}")
        
        # Add recent consultation history
        if consultation_history:
            context_parts.append("\nConsultation Discussion:")
            for turn in consultation_history[-4:]:  # Last 4 turns
                context_parts.append(
                    f"{turn.doctor_role.value.title()}: {turn.response_text}"
                )
        
        # Add discussion prompt
        discussion_prompts = {
            DoctorRole.SPECIALIST: "Please respond to the previous discussions and provide additional specialist insights:",
            DoctorRole.INTERNIST: "Please comment on the previous opinions and add your internal medicine perspective:",
            DoctorRole.RADIOLOGIST: "Please discuss the imaging findings in light of the previous comments:",
            DoctorRole.ATTENDING: "Please facilitate the discussion and help reach consensus:",
        }
        
        role_prompt = discussion_prompts.get(
            doctor_role, 
            "Please contribute to the discussion with your medical expertise:"
        )
        context_parts.append(f"\n{role_prompt}")
        
        return "\n".join(context_parts)
    
    def gather_consultation_data(self, consultation_data: List[Dict[str, Any]]) -> DataProto:
        """
        Gather consultation data into DataProto format
        
        Args:
            consultation_data: List of consultation turn data
            
        Returns:
            Gathered consultation data as DataProto
        """
        if not consultation_data:
            # Return empty DataProto if no data
            return DataProto.from_single_dict(data={})
        
        print(f"Gathering consultation data from {len(consultation_data)} turns")
        
        # Organize data by keys
        gathered_data = {}
        tensor_keys = ['responses']
        list_keys = ['prompts', 'response_text', 'input_text', 'doctor_role', 'case_id']
        
        # Initialize data structures
        for key in tensor_keys:
            gathered_data[key] = []
        
        for key in list_keys:
            gathered_data[key] = []
        
        # Collect data from all consultation turns
        for turn_data in consultation_data:
            for key in tensor_keys:
                if key in turn_data:
                    gathered_data[key].append(turn_data[key])
            
            for key in list_keys:
                if key in turn_data:
                    gathered_data[key].append(turn_data[key])
        
        # Convert to appropriate formats
        for key in tensor_keys:
            if gathered_data[key]:
                if isinstance(gathered_data[key][0], torch.Tensor):
                    gathered_data[key] = torch.stack(gathered_data[key])
                else:
                    gathered_data[key] = torch.tensor(gathered_data[key])
        
        # Add consultation statistics
        consultation_stats = {
            'total_turns': len(consultation_data),
            'total_cases': len(set(turn['case_id'] for turn in consultation_data)),
            'doctor_participation': {
                role: sum(1 for turn in consultation_data if turn.get('doctor_role') == role)
                for role in ['specialist', 'internist', 'radiologist', 'attending']
            },
            'avg_response_length': sum(turn.get('response_length', 0) for turn in consultation_data) / len(consultation_data),
        }
        
        # Create DataProto with metadata
        result = DataProto.from_single_dict(data=gathered_data)
        result.meta_info = {
            'consultation_stats': consultation_stats,
            'consultation_type': 'medical_multi_doctor',
        }
        
        print(f"Consultation data gathered: {consultation_stats}")
        return result 