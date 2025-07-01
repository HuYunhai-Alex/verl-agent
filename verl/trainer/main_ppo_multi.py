# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import os

import hydra
import ray

from verl.trainer.ppo.ray_trainer_multi import RayPPOTrainer
from verl.trainer.ppo.reward import load_reward_manager

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask


class MedicalConsultationDataset:
    """Medical consultation dataset"""
    
    def __init__(self, data_path: str = None, tokenizer=None):
        self.tokenizer = tokenizer
        
        # Sample medical case data
        self.medical_cases = [
            {
                "case_id": "case_001",
                "patient_info": "65-year-old male presenting with chest pain for 3 hours",
                "symptoms": "Substernal crushing chest pain radiating to left arm, with sweating and nausea",
                "vital_signs": "BP: 160/95 mmHg, HR: 110 bpm, RR: 22/min, T: 37.2°C",
                "history": "Hypertension for 10 years, smoking history for 30 years",
                "physical_exam": "Rapid heart rate, irregular rhythm, clear lung sounds bilaterally",
                "lab_results": "Elevated troponin I, elevated CK-MB",
                "imaging": "ECG shows ST elevation in leads II, III, aVF",
                "prompt": "Please provide your medical opinion on this case and recommend next steps.",
            },
            {
                "case_id": "case_002",
                "patient_info": "45-year-old female with persistent cough for 2 weeks",
                "symptoms": "Dry cough, fever, fatigue, shortness of breath",
                "vital_signs": "BP: 120/80 mmHg, HR: 95 bpm, RR: 24/min, T: 38.5°C",
                "history": "No significant past medical history, non-smoker",
                "physical_exam": "Decreased breath sounds in right lower lobe, dullness to percussion",
                "lab_results": "Elevated white blood cell count, increased CRP",
                "imaging": "Chest X-ray shows right lower lobe consolidation",
                "prompt": "What is your differential diagnosis and treatment plan?",
            },
            {
                "case_id": "case_003",
                "patient_info": "72-year-old male with sudden onset severe headache",
                "symptoms": "Worst headache of life, photophobia, neck stiffness",
                "vital_signs": "BP: 180/110 mmHg, HR: 85 bpm, RR: 18/min, T: 37.0°C",
                "history": "Hypertension, diabetes mellitus type 2",
                "physical_exam": "Nuchal rigidity, Kernig's sign positive",
                "lab_results": "Normal CBC, glucose elevated",
                "imaging": "CT head shows subarachnoid hemorrhage",
                "prompt": "Please evaluate this neurological emergency and suggest management.",
            },
            {
                "case_id": "case_004",
                "patient_info": "28-year-old pregnant woman at 32 weeks gestation with abdominal pain",
                "symptoms": "Right upper quadrant pain, nausea, vomiting",
                "vital_signs": "BP: 150/95 mmHg, HR: 100 bpm, RR: 20/min, T: 37.5°C",
                "history": "First pregnancy, no complications until now",
                "physical_exam": "RUQ tenderness, mild peripheral edema",
                "lab_results": "Elevated liver enzymes, proteinuria",
                "imaging": "Ultrasound shows normal fetal development",
                "prompt": "What are your concerns and recommendations for this pregnant patient?",
            },
                        {
                "case_id": "case_005",
                "patient_info": "65-year-old male presenting with chest pain for 3 hours",
                "symptoms": "Substernal crushing chest pain radiating to left arm, with sweating and nausea",
                "vital_signs": "BP: 160/95 mmHg, HR: 110 bpm, RR: 22/min, T: 37.2°C",
                "history": "Hypertension for 10 years, smoking history for 30 years",
                "physical_exam": "Rapid heart rate, irregular rhythm, clear lung sounds bilaterally",
                "lab_results": "Elevated troponin I, elevated CK-MB",
                "imaging": "ECG shows ST elevation in leads II, III, aVF",
                "prompt": "Please provide your medical opinion on this case and recommend next steps.",
            },
            {
                "case_id": "case_006",
                "patient_info": "45-year-old female with persistent cough for 2 weeks",
                "symptoms": "Dry cough, fever, fatigue, shortness of breath",
                "vital_signs": "BP: 120/80 mmHg, HR: 95 bpm, RR: 24/min, T: 38.5°C",
                "history": "No significant past medical history, non-smoker",
                "physical_exam": "Decreased breath sounds in right lower lobe, dullness to percussion",
                "lab_results": "Elevated white blood cell count, increased CRP",
                "imaging": "Chest X-ray shows right lower lobe consolidation",
                "prompt": "What is your differential diagnosis and treatment plan?",
            },
            {
                "case_id": "case_007",
                "patient_info": "72-year-old male with sudden onset severe headache",
                "symptoms": "Worst headache of life, photophobia, neck stiffness",
                "vital_signs": "BP: 180/110 mmHg, HR: 85 bpm, RR: 18/min, T: 37.0°C",
                "history": "Hypertension, diabetes mellitus type 2",
                "physical_exam": "Nuchal rigidity, Kernig's sign positive",
                "lab_results": "Normal CBC, glucose elevated",
                "imaging": "CT head shows subarachnoid hemorrhage",
                "prompt": "Please evaluate this neurological emergency and suggest management.",
            },
            {
                "case_id": "case_008",
                "patient_info": "28-year-old pregnant woman at 32 weeks gestation with abdominal pain",
                "symptoms": "Right upper quadrant pain, nausea, vomiting",
                "vital_signs": "BP: 150/95 mmHg, HR: 100 bpm, RR: 20/min, T: 37.5°C",
                "history": "First pregnancy, no complications until now",
                "physical_exam": "RUQ tenderness, mild peripheral edema",
                "lab_results": "Elevated liver enzymes, proteinuria",
                "imaging": "Ultrasound shows normal fetal development",
                "prompt": "What are your concerns and recommendations for this pregnant patient?",
            },
                        {
                "case_id": "case_009",
                "patient_info": "65-year-old male presenting with chest pain for 3 hours",
                "symptoms": "Substernal crushing chest pain radiating to left arm, with sweating and nausea",
                "vital_signs": "BP: 160/95 mmHg, HR: 110 bpm, RR: 22/min, T: 37.2°C",
                "history": "Hypertension for 10 years, smoking history for 30 years",
                "physical_exam": "Rapid heart rate, irregular rhythm, clear lung sounds bilaterally",
                "lab_results": "Elevated troponin I, elevated CK-MB",
                "imaging": "ECG shows ST elevation in leads II, III, aVF",
                "prompt": "Please provide your medical opinion on this case and recommend next steps.",
            },
            {
                "case_id": "case_010",
                "patient_info": "45-year-old female with persistent cough for 2 weeks",
                "symptoms": "Dry cough, fever, fatigue, shortness of breath",
                "vital_signs": "BP: 120/80 mmHg, HR: 95 bpm, RR: 24/min, T: 38.5°C",
                "history": "No significant past medical history, non-smoker",
                "physical_exam": "Decreased breath sounds in right lower lobe, dullness to percussion",
                "lab_results": "Elevated white blood cell count, increased CRP",
                "imaging": "Chest X-ray shows right lower lobe consolidation",
                "prompt": "What is your differential diagnosis and treatment plan?",
            },
            {
                "case_id": "case_011",
                "patient_info": "72-year-old male with sudden onset severe headache",
                "symptoms": "Worst headache of life, photophobia, neck stiffness",
                "vital_signs": "BP: 180/110 mmHg, HR: 85 bpm, RR: 18/min, T: 37.0°C",
                "history": "Hypertension, diabetes mellitus type 2",
                "physical_exam": "Nuchal rigidity, Kernig's sign positive",
                "lab_results": "Normal CBC, glucose elevated",
                "imaging": "CT head shows subarachnoid hemorrhage",
                "prompt": "Please evaluate this neurological emergency and suggest management.",
            },
            {
                "case_id": "case_012",
                "patient_info": "28-year-old pregnant woman at 32 weeks gestation with abdominal pain",
                "symptoms": "Right upper quadrant pain, nausea, vomiting",
                "vital_signs": "BP: 150/95 mmHg, HR: 100 bpm, RR: 20/min, T: 37.5°C",
                "history": "First pregnancy, no complications until now",
                "physical_exam": "RUQ tenderness, mild peripheral edema",
                "lab_results": "Elevated liver enzymes, proteinuria",
                "imaging": "Ultrasound shows normal fetal development",
                "prompt": "What are your concerns and recommendations for this pregnant patient?",
            },
                        {
                "case_id": "case_013",
                "patient_info": "65-year-old male presenting with chest pain for 3 hours",
                "symptoms": "Substernal crushing chest pain radiating to left arm, with sweating and nausea",
                "vital_signs": "BP: 160/95 mmHg, HR: 110 bpm, RR: 22/min, T: 37.2°C",
                "history": "Hypertension for 10 years, smoking history for 30 years",
                "physical_exam": "Rapid heart rate, irregular rhythm, clear lung sounds bilaterally",
                "lab_results": "Elevated troponin I, elevated CK-MB",
                "imaging": "ECG shows ST elevation in leads II, III, aVF",
                "prompt": "Please provide your medical opinion on this case and recommend next steps.",
            },
            {
                "case_id": "case_014",
                "patient_info": "45-year-old female with persistent cough for 2 weeks",
                "symptoms": "Dry cough, fever, fatigue, shortness of breath",
                "vital_signs": "BP: 120/80 mmHg, HR: 95 bpm, RR: 24/min, T: 38.5°C",
                "history": "No significant past medical history, non-smoker",
                "physical_exam": "Decreased breath sounds in right lower lobe, dullness to percussion",
                "lab_results": "Elevated white blood cell count, increased CRP",
                "imaging": "Chest X-ray shows right lower lobe consolidation",
                "prompt": "What is your differential diagnosis and treatment plan?",
            },
            {
                "case_id": "case_015",
                "patient_info": "72-year-old male with sudden onset severe headache",
                "symptoms": "Worst headache of life, photophobia, neck stiffness",
                "vital_signs": "BP: 180/110 mmHg, HR: 85 bpm, RR: 18/min, T: 37.0°C",
                "history": "Hypertension, diabetes mellitus type 2",
                "physical_exam": "Nuchal rigidity, Kernig's sign positive",
                "lab_results": "Normal CBC, glucose elevated",
                "imaging": "CT head shows subarachnoid hemorrhage",
                "prompt": "Please evaluate this neurological emergency and suggest management.",
            },
            {
                "case_id": "case_016",
                "patient_info": "28-year-old pregnant woman at 32 weeks gestation with abdominal pain",
                "symptoms": "Right upper quadrant pain, nausea, vomiting",
                "vital_signs": "BP: 150/95 mmHg, HR: 100 bpm, RR: 20/min, T: 37.5°C",
                "history": "First pregnancy, no complications until now",
                "physical_exam": "RUQ tenderness, mild peripheral edema",
                "lab_results": "Elevated liver enzymes, proteinuria",
                "imaging": "Ultrasound shows normal fetal development",
                "prompt": "What are your concerns and recommendations for this pregnant patient?",
            },
        ]
        
        self.max_prompt_length = 1024
        self.return_raw_chat = False
        self.return_full_prompt = False
        self.truncation = "error"
        self.filter_overlong_prompts = True
    
    def __len__(self):
        return len(self.medical_cases)
    
    def __getitem__(self, idx):
        case = self.medical_cases[idx]
        row_dict: dict = {}
        
        # Format the medical case into a prompt
        prompt_parts = [
            f"Patient Information: {case['patient_info']}",
            f"Symptoms: {case['symptoms']}",
            f"Vital Signs: {case['vital_signs']}",
            f"Medical History: {case['history']}",
            f"Physical Examination: {case['physical_exam']}",
            f"Laboratory Results: {case['lab_results']}",
        ]
        
        if 'imaging' in case:
            prompt_parts.append(f"Imaging: {case['imaging']}")
        
        prompt_parts.append(f"Question: {case['prompt']}")
        
        full_prompt = "\n\n".join(prompt_parts)
        
        raw_prompt = self.tokenizer.apply_chat_template(full_prompt, add_generation_prompt=True, tokenize=False)
        model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_prompt"] = full_prompt

        # get prompts with chat template
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt  # array of strings

        row_dict["case_id"] = case["case_id"]
        row_dict["data_source"] = "medical_consultation"
        row_dict["prompt"] = raw_prompt
        row_dict["raw_prompt"] = full_prompt
        
        return row_dict

def collate_fn(batch):
    """Collate function for batch processing"""
    return {
        "case_id": [item["case_id"] for item in batch],
        "prompt": [item["prompt"] for item in batch],
        "raw_prompt": [item["raw_prompt"] for item in batch],
        "data_source": [item["data_source"] for item in batch],
    }

@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config) -> None:
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(
            runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_LOGGING_LEVEL": "WARN", "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true"}},
            num_cpus=config.ray_init.num_cpus,
        )

    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    def run(self, config):
        # print initial config
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
        OmegaConf.resolve(config)

        # download the checkpoint from hdfs
        local_path = copy_to_local(config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False))

        # from agent_system.environments import make_envs
        # envs, val_envs = make_envs(config)
        envs, val_envs = None, None 

        # instantiate tokenizer
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)  # used for multimodal LLM, could be none

        # vllm early verify
        if config.actor_rollout_ref.rollout.name in ["vllm"]:
            from verl.utils.vllm_utils import is_version_ge

            if config.actor_rollout_ref.model.get("lora_rank", 0) > 0:
                if not is_version_ge(pkg="vllm", minver="0.7.3"):
                    raise NotImplementedError("PPO LoRA is not supported before vllm 0.7.3")

        # define worker classes
        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            assert config.critic.strategy in ["fsdp", "fsdp2"]
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = ActorRolloutRefWorker
            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer_multi import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.DoctorSpecialist: ray.remote(actor_rollout_cls),
            #Role.DoctorInternist: ray.remote(actor_rollout_cls),
            Role.DoctorRadiologist: ray.remote(actor_rollout_cls),
            #Role.DoctorAttending: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }

        global_pool_id = "global_pool"
        
        resource_pool_spec = {
            global_pool_id: [1],
            "doctor_specialist": [1],
            #"doctor_internist": [1],
            #"doctor_radiologist": [1],
            #"doctor_attending": [1],
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
            Role.RewardModel: global_pool_id,
            Role.DoctorSpecialist: "doctor_specialist",
            Role.DoctorRadiologist: global_pool_id,
            #Role.DoctorInternist: "doctor_internist",
            #Role.DoctorRadiologist: "doctor_radiologist",
            #Role.DoctorAttending: "doctor_attending",
        }

        # we should adopt a multi-source reward function here
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # - finally, we combine all the rewards together
        # - The reward type depends on the tag of the data
        if config.reward_model.enable:
            if config.reward_model.strategy in ["fsdp", "fsdp2"]:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        # use reference model
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        reward_manager_name = config.reward_model.get("reward_manager", "episode")
        if reward_manager_name == 'episode':
            from agent_system.reward_manager.episode import EpisodeRewardManager
            reward_manager_cls = EpisodeRewardManager
        else:
            raise NotImplementedError

        reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=0, normalize_by_length=False)

        # Note that we always use function-based RM for validation
        val_reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=1, normalize_by_length=False)

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        assert config.actor_rollout_ref.rollout.n == 1, "In verl, actor_rollout_ref.rollout.n>1 is for GRPO. In verl+env, we keep n=1, and achieve GRPO by env.rollout.n"

        # from agent_system.multi_turn_rollout import TrajectoryCollector
        # traj_collector = TrajectoryCollector(config=config, tokenizer=tokenizer, processor=processor)

        from agent_system.multi_turn_rollout.medical_consultation_rollout import MedicalConsultationCollector
        traj_collector = MedicalConsultationCollector(tokenizer=tokenizer)

        from verl.utils.dataset.rl_dataset import collate_fn

        # train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
        # val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
        # train_sampler = create_rl_sampler(config.data, train_dataset)
        
        train_dataset = MedicalConsultationDataset(tokenizer=tokenizer)
        val_dataset = MedicalConsultationDataset(tokenizer=tokenizer)
        train_sampler = create_rl_sampler(config.data, train_dataset)
        
        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=config.trainer.device,
            traj_collector=traj_collector,
            envs=envs,
            val_envs=val_envs,
        )
        trainer.init_workers()
        trainer.fit()


def create_rl_dataset(data_paths, data_config, tokenizer, processor):
    """Create a dataset.

    Arguments:
        data_config: The data config.
        tokenizer (Tokenizer): The tokenizer.
        processor (Processor): The processor.

    Returns:
        dataset (Dataset): The dataset.
    """
    from torch.utils.data import Dataset

    from verl.utils.dataset.rl_dataset import RLHFDataset

    if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
        from verl.utils.import_utils import load_extern_type

        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
        if not issubclass(dataset_cls, Dataset):
            raise TypeError(f"The custom dataset class '{data_config.custom_cls.name}' from '{data_config.custom_cls.path}' must inherit from torch.utils.data.Dataset")
    else:
        dataset_cls = RLHFDataset
    print(f"Using dataset class: {dataset_cls.__name__}")

    dataset = dataset_cls(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
    )

    return dataset


def create_rl_sampler(data_config, dataset):
    """Create a sampler for the dataset.

    Arguments:
        data_config: The data config.
        dataset (Dataset): The dataset.

    Returns:
        sampler (Sampler): The sampler.
    """
    import torch
    from torch.utils.data import RandomSampler, SequentialSampler

    # use sampler for better ckpt resume
    if data_config.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(data_config.get("seed", 1))
        sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
    else:
        sampler = SequentialSampler(data_source=dataset)

    return sampler


if __name__ == "__main__":
    main()
