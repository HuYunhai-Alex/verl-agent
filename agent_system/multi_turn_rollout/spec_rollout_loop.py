from collections import defaultdict
import random
import torch
import numpy as np
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from transformers import PreTrainedTokenizer
import uuid
from verl.models.transformers.qwen2_vl import get_rope_index
from agent_system.multi_turn_rollout.utils import process_image, to_list_of_dict, torch_to_numpy, filter_group_data
from agent_system.environments import EnvironmentManagerBase
from typing import List, Dict

import logging

# Configure a root logger. In an application, prefer configuring this once at entry‑point.
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.DEBUG,
)
logger = logging.getLogger(__name__)

class TrajectoryCollector:
    def __init__(self, config, tokenizer: PreTrainedTokenizer, processor=None):
        """
        Initialize the TrajectoryProcessor class.
        
        Parameters:
            config: Configuration object containing data processing settings
            tokenizer (PreTrainedTokenizer): Tokenizer for text encoding and decoding
            processor: Image processor for multimodal inputs
        """
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor

    def preprocess_single_sample(
        self,
        item: int,
        gen_batch: DataProto,
        spec_history,
        template_enable: bool = True,
    ):
        """
        Process a single observation sample, organizing environment observations (text and/or images) 
        into a format processable by the model.
        
        Parameters:
            item (int): Sample index in the batch
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation, may contain 'text', 'image', 'anchor' keys
        
        Returns:
            dict: Contains processed input data such as input_ids, attention_mask, etc.
        """

        raw_prompt = gen_batch.non_tensor_batch['raw_prompt'][item]
        obs_content = raw_prompt[0]['content']
        data_source = gen_batch.non_tensor_batch['data_source'][item]
        next_ids = None

        if spec_history is not None and spec_history[-1].get('next_ids', None) is not None:
            # logger.warning(f"Processing item {item} latest spec_history: {spec_history[-1]}")
            next_ids = spec_history[-1]['next_ids'] # Get the next response ids if available

        # Build chat structure
        
        # Apply chat template
        if next_ids is None and template_enable:
            chat = np.array([{
                "content": obs_content,
                "role": "user",
            }])
            prompt_with_chat_template = self.tokenizer.apply_chat_template(
                chat,
                add_generation_prompt=True,
                tokenize=False
            )
        else:
            prompt_with_chat_template = self.tokenizer.decode(next_ids, skip_special_tokens=False)
        
        # Initialize return dict
        row_dict = {}
        
        # Process multimodal data
        raw_prompt = prompt_with_chat_template

        # logger.warning(f"Processing item {item} with raw prompt: {raw_prompt}")
        
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                            tokenizer=self.tokenizer,
                                                                            max_length=self.config.data.max_prompt_length,
                                                                            pad_token_id=self.tokenizer.pad_token_id,
                                                                            left_pad=True,
                                                                            truncation=self.config.data.truncation,)
        
        

        position_ids = compute_position_id_with_mask(attention_mask)
        
        # Build final output dict
        row_dict.update({
            'input_ids': input_ids[0],
            'attention_mask': attention_mask[0],
            'position_ids': position_ids[0],
            'raw_prompt_ids': self.tokenizer.encode(raw_prompt, add_special_tokens=True),
            'raw_prompt': raw_prompt,
            'index': item,
            'data_source': data_source
        })

        # if self.config.data.get('return_raw_chat', False):
        #     row_dict['raw_prompt'] = chat.tolist()
        
        return row_dict

    def preprocess_batch(
        self,
        gen_batch: DataProto, 
        spec_history_batch: List, 
        template_enable: bool = True,
    ) -> DataProto:
        """
        Process a batch of observation samples, converting environment observations into model-processable format.
        
        Parameters:
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation dictionary
                - 'text' (None or List[str]): Text observation data
                - 'image' (np.ndarray or torch.Tensor): Image observation data
                - 'anchor' (None or Any): Anchor observation without any histories or additional info. (for GiGPO only).
        
        Returns:
            DataProto: Contains processed batch data with preserved metadata
        """
        batch_size = len(gen_batch.batch['input_ids'])
        processed_samples = []
        # logger.warning(f"Preprocessing batch of size {batch_size} with gen_batch: {gen_batch}")

        # logger.warning(f"Spec history batch: {spec_history_batch}")         
        # Process each sample in parallel
        for item in range(batch_size):
            # Extract per-sample observations
            processed = self.preprocess_single_sample(
                item=item,
                gen_batch=gen_batch,
                spec_history=spec_history_batch[item],
                template_enable=template_enable,
            )
            processed_samples.append(processed)
        
        # Aggregate batch data
        batch = collate_fn(processed_samples)
        
        # Create DataProto with preserved metadata
        new_batch = DataProto.from_single_dict(
            data=batch,
            meta_info=gen_batch.meta_info
        )

        return new_batch


    def gather_rollout_data(
            self,
            total_batch_list: List[List[Dict]],
            episode_rewards: np.ndarray,
            episode_lengths: np.ndarray,
            traj_uid: np.ndarray,
            ) -> DataProto:
        """
        Collect and organize trajectory data, handling batch size adjustments to meet parallel training requirements.
        
        Parameters:
            total_batch_list (List[List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
        
        Returns:
            DataProto: Collected and organized trajectory data
        """
        batch_size = len(total_batch_list)

        episode_rewards_mean = np.mean(episode_rewards)
        episode_rewards_min = np.min(episode_rewards)
        episode_rewards_max = np.max(episode_rewards)

        episode_lengths_mean = np.mean(episode_lengths)
        episode_lengths_min = np.min(episode_lengths)
        episode_lengths_max = np.max(episode_lengths)

        # success_rate = {}
        # for key, value in success.items():
        #     success_rate[key] = np.mean(value)
        
        effective_batch = []
        for bs in range(batch_size):
            # sum the rewards for each data in total_batch_list[bs]
            for data in total_batch_list[bs]:
                assert traj_uid[bs] == data['traj_uid'], "data is not from the same trajectory"
                if data['active_masks']:
                    # episode_rewards
                    data['episode_rewards'] = episode_rewards[bs]
                    data['episode_rewards_mean'] = episode_rewards_mean
                    data['episode_rewards_min'] = episode_rewards_min
                    data['episode_rewards_max'] = episode_rewards_max
                    # episode_lengths
                    data['episode_lengths'] = episode_lengths[bs]
                    data['episode_lengths_mean'] = episode_lengths_mean
                    data['episode_lengths_min'] = episode_lengths_min
                    data['episode_lengths_max'] = episode_lengths_max
                    # success_rate
                    # for key, value in success_rate.items():
                    #     data[key] = value

                    effective_batch.append(data)
            
        # Convert trajectory data to DataProto format
        gen_batch_output = DataProto.from_single_dict(
            data=collate_fn(effective_batch)
        )
        return gen_batch_output

    def step(
        self,
        draft_rollout_wg,
        target_rollout_wg,
        text_actions,
        batch,
        batch_input: DataProto,
    ):
        """Compare draft actions against target model output.

        Reward(i) = prefix_match_len / len(target_ids)
        Done(i)   = prefix_match_len == len(target_ids)
        Next obs  = 当前 prompt + 通过的 prefix token 重新 decode 出文本。
        """
        # 1. target model output
        # logger.warning(f"before log_probs batch {batch}")
        draft_log_probs = draft_rollout_wg.compute_log_prob(batch)
        target_log_probs = target_rollout_wg.compute_ref_log_prob(batch)

        batch_size = len(draft_log_probs)
        rewards = np.zeros(batch_size, dtype=np.float32)
        dones = np.zeros(batch_size, dtype=bool)

        accept_batch = []
        passed_token_batch = []

        for i in range(batch_size):
            accept = 0
            is_done = False
            # logger.warning(f"Processing batch {i} with draft response decoded: {self.tokenizer.decode(batch.batch['responses'][i], skip_special_tokens=True)}")
            sequence_length = len(draft_log_probs.batch[i].get("old_log_probs", []))
            for j in range(sequence_length):
                r = random.random()
                # logger.warning(f"r is {r}, Draft log prob: {draft_log_probs.batch[i].get('old_log_probs', [])[j]}, Target ref log prob: {target_log_probs.batch[i].get('ref_log_prob', [])[j]}")
                if torch.log(torch.tensor(r)) <= (draft_log_probs.batch[i].get("old_log_probs", [])[j] - target_log_probs.batch[i].get("ref_log_prob", [])[j] + 1e-8):
                    accept += 1
                    if batch.batch["responses"][i][j] == self.tokenizer.eos_token_id:
                        logger.warning(f"Accepting token {j} for batch {i} with EOS token.")
                        accept = len(draft_log_probs.batch[i].get("old_log_probs", []))  # Accept all tokens if EOS is found
                        is_done = True
                        break
                else:
                    break
            # passed_token = batch.batch['responses'][i][:j]
            logger.warning(f"input_ids: {batch.batch['input_ids'][i][batch.batch['attention_mask'][i].bool()][-5:]}, responses: {batch.batch['responses'][i]}")
            # assert batch.batch['input_ids'][i][:-sequence_length] == batch.batch['responses'][i][:-sequence_length ], f"Input IDs and responses mismatch for batch {i}: {self.tokenizer.decode(batch.batch['input_ids'][i][:-sequence_length])} vs {self.tokenizer.decode(batch.batch['responses'][i][:-sequence_length])}"
            if accept == 0:
                logger.warning(f"Accept is 0 for batch {i}, using the first token as next input.")
                next_input_ids = batch.batch['input_ids'][i][batch.batch['attention_mask'][i].bool()][:j-sequence_length+1]

            next_input_ids = batch.batch['input_ids'][i][batch.batch['attention_mask'][i].bool()][:j-sequence_length]
            passed_token_batch.append(next_input_ids)
            
            accept_batch.append(accept)
            rewards[i] = accept / max(1, len(draft_log_probs.batch[i].get("old_log_probs", [])))
            dones[i] = is_done

        logger.warning(f"Accept batch: {accept_batch}")
                
        # 返回给主循环的 "obs" — 用文本字段即可
        return passed_token_batch, rewards, dones

    def vanilla_multi_turn_loop(
        self,
        gen_batch: DataProto,
        draft_rollout_wg,
        target_rollout_wg,
        envs: EnvironmentManagerBase,
    ) -> DataProto:
        """Collect trajectories with verbose logging.

        Args:
            gen_batch: Initial prompts batch.
            draft_rollout_wg: Worker group that produces draft sequences.
            target_rollout_wg: Worker group that executes verified actions.
            envs: Vectorised environments.

        Returns:
            Tuple with trajectory list, rewards, lengths, success flags, and traj_uid array.
        """
        logger.warning("Starting `vanilla_multi_turn_loop` – requested max_steps=%d", self.config.env.max_steps)

        if self.config.env.rollout.n > 0:
            gen_batch = gen_batch.repeat(repeat_times=self.config.env.rollout.n, interleave=True)

        batch_size = len(gen_batch.batch["input_ids"])

        if self.config.env.rollout.n > 0:
            uid_batch: List[str] = []
            for i in range(batch_size):
                if i % self.config.env.rollout.n == 0:
                    uid = str(uuid.uuid4())
                uid_batch.append(uid)
            uid_batch = np.array(uid_batch, dtype=object)
        else:
            uid = str(uuid.uuid4())
            uid_batch = np.array([uid] * batch_size, dtype=object)
        traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)
        logger.warning("Generated traj_uid for each env: %s", traj_uid)

        is_done = np.zeros(batch_size, dtype=bool)
        total_batch_list: List[list] = [[] for _ in range(batch_size)]
        episode_lengths = np.zeros(batch_size, dtype=np.int32)
        episode_rewards = np.zeros(batch_size, dtype=np.float32)
        template_enable = True

        spec_history_batch = [None] * batch_size  # Placeholder for spec history, if needed

        for step_idx in range(self.config.env.max_steps):
            logger.warning("\u27a1\ufe0f  Step %d / %d", step_idx + 1, self.config.env.max_steps)
            active_masks = np.logical_not(is_done)
            if not active_masks.any():
                logger.warning("All environments done at step %d", step_idx)
                break

            batch = self.preprocess_batch(gen_batch=gen_batch, spec_history_batch=spec_history_batch, template_enable=template_enable)
            template_enable = False

            batch_input = batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=[
                    k
                    for k in [
                        "raw_prompt_ids",
                        "multi_modal_data",
                        "raw_prompt",
                        "tools_kwargs",
                    ]
                    if k in batch.non_tensor_batch
                ],
            )
            batch_input.meta_info = gen_batch.meta_info

            # Draft generation.
            batch_output = draft_rollout_wg.generate_sequences(batch_input)
            # logger.warning("Draft sequences generated – shape: %s", batch_output.batch["responses"].shape)

            # Merge outputs back.
            batch.non_tensor_batch["uid"] = uid_batch
            batch.non_tensor_batch["traj_uid"] = traj_uid
            batch = batch.union(batch_output)

            # Decode text actions for execution.
            text_actions = self.tokenizer.batch_decode(
                batch.batch["responses"],
                skip_special_tokens=True,
            )
            # logger.warning("Decoded %d actions", len(text_actions))

            # Execute actions in environment.
            passed_token_batch, rewards, dones = self.step(draft_rollout_wg, target_rollout_wg, batch.batch["responses"], batch, batch_input)
            # logger.warning("Env step: rewards shape=%s, dones shape=%s", rewards.shape, dones.shape)

            for item in range(batch_size):
                spec_history = {
                        "input_ids": batch.batch["input_ids"][item].tolist(),
                        "attention_mask": batch.batch["attention_mask"][item].tolist(),
                        "position_ids": batch.batch["position_ids"][item].tolist(),
                        # "raw_prompt_ids": batch.non_tensor_batch["raw_prompt_ids"][item],
                        "raw_prompt": batch.non_tensor_batch["raw_prompt"][item],
                        "generated_tokens": text_actions[item],
                        "next_ids": passed_token_batch[item],  # Placeholder for passed text
                        "text_action": text_actions[item],
                        "image_action": None,  # Placeholder for image actions if needed
                        }
                if spec_history_batch[item] is None:
                    spec_history_batch[item] = [spec_history]
                else:
                    spec_history_batch[item].append(spec_history)
                
            # Squeeze singleton dims if any.
            if rewards.ndim == 2:
                rewards = rewards.squeeze(1)
            if dones.ndim == 2:
                dones = dones.squeeze(1)

            batch.non_tensor_batch["is_action_valid"] = np.ones(batch_size, dtype=bool)

            # Accumulate episode stats.
            episode_rewards += torch_to_numpy(rewards) * torch_to_numpy(active_masks)
            episode_lengths[active_masks] += 1

            batch.non_tensor_batch["rewards"] = torch_to_numpy(rewards, is_object=True)
            batch.non_tensor_batch["active_masks"] = torch_to_numpy(active_masks, is_object=True)

            # Store per‑env step data.
            for i, env_dict in enumerate(to_list_of_dict(batch)):
                total_batch_list[i].append(env_dict)

            # Update termination flags & observation.
            is_done = np.logical_or(is_done, dones)

            logger.warning(
                "STEP %-3d | actions=%s | r=%s | R=%s | done=%s",
                step_idx,
                [a.strip() for a in text_actions],
                rewards.tolist(),
                episode_rewards.tolist(),
                dones.tolist(),
            )

            if is_done.all():
                break

        logger.warning("Collection finished – mean_ep_reward=%.3f, mean_ep_length=%.2f", episode_rewards.mean(), episode_lengths.mean())

        return total_batch_list, episode_rewards, episode_lengths, traj_uid

    
    def dynamic_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            draft_rollout_wg, 
            target_rollout_wg,
            envs: EnvironmentManagerBase,
            ) -> DataProto:
        """
        Conduct dynamic rollouts until a target batch size is met. 
        Keeps sampling until the desired number of effective trajectories is collected.
        Adopted from DAPO (https://arxiv.org/abs/2503.14476)

        Args:
            gen_batch (DataProto): Initial batch for rollout.
            actor_rollout_wg: Actor model workers for generating responses.
            envs (EnvironmentManagerBase): Environment manager instance.

        Returns:
            total_batch_list (List[Dict]): Complete set of rollout steps.
            total_episode_rewards (np.ndarray): Accumulated rewards.
            total_episode_lengths (np.ndarray): Lengths per episode.
            total_success (Dict[str, np.ndarray]): Success metrics.
            total_traj_uid (np.ndarray): Trajectory IDs.
        """
        total_batch_list = []
        total_episode_rewards = []
        total_episode_lengths = []
        total_success = []
        total_traj_uid = []
        try_count: int = 0
        max_try_count = self.config.algorithm.filter_groups.max_num_gen_batches

        while len(total_batch_list) < self.config.data.train_batch_size * self.config.env.rollout.n and try_count < max_try_count:

            logger.warning(f"Attempt {try_count + 1}/{max_try_count} to collect enough trajectories. Current count: {len(total_batch_list)}")

            if len(total_batch_list) > 0:
                print(f"valid num={len(total_batch_list)} < target num={self.config.data.train_batch_size * self.config.env.rollout.n}. Keep generating... ({try_count}/{max_try_count})")
            try_count += 1

            batch_list, episode_rewards, episode_lengths, success, traj_uid = self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                draft_rollout_wg=draft_rollout_wg,
                target_rollout_wg=target_rollout_wg,
                envs=envs,
            )
            batch_list, episode_rewards, episode_lengths, traj_uid = filter_group_data(batch_list=batch_list,
                                                                                                episode_rewards=episode_rewards, 
                                                                                                episode_lengths=episode_lengths, 
                                                                                                success=success, 
                                                                                                traj_uid=traj_uid, 
                                                                                                config=self.config,
                                                                                                last_try=(try_count == max_try_count),
                                                                                                )
            
            total_batch_list += batch_list
            total_episode_rewards.append(episode_rewards)
            total_episode_lengths.append(episode_lengths)
            #total_success.append(success)
            total_traj_uid.append(traj_uid)

        total_episode_rewards = np.concatenate(total_episode_rewards, axis=0)
        total_episode_lengths = np.concatenate(total_episode_lengths, axis=0)
        total_success = None
        total_traj_uid = np.concatenate(total_traj_uid, axis=0)

        return total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid

    def multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            is_train: bool = True,
            ) -> DataProto:
        """
        Select and run the appropriate rollout loop (dynamic or vanilla).

        Args:
            gen_batch (DataProto): Initial prompt batch.
            actor_rollout_wg: Actor model workers.
            envs (EnvironmentManagerBase): Environment manager for interaction.
            is_train (bool): Whether in training mode (affects dynamic sampling).

        Returns:
            DataProto: Final collected trajectory data with metadata.
        """
        # Initial observations from the environment
        if self.config.algorithm.filter_groups.enable and is_train:
            # Dynamic Sampling (for DAPO and Dynamic GiGPO)
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid = \
                self.dynamic_multi_turn_loop(
                gen_batch=gen_batch,
                draft_rollout_wg=actor_rollout_wg["draft"],
                target_rollout_wg=actor_rollout_wg["target"],
                envs=envs,
            )
        else:
            # Vanilla Sampling   
            total_batch_list, total_episode_rewards, total_episode_lengths, total_traj_uid = \
                self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                draft_rollout_wg=actor_rollout_wg["draft"],
                target_rollout_wg=actor_rollout_wg["target"],
                envs=envs,
            )
        assert len(total_batch_list) == len(total_episode_rewards)
        assert len(total_batch_list) == len(total_episode_lengths)
        assert len(total_batch_list) == len(total_traj_uid)
        

        # Create trajectory data
        gen_batch_output: DataProto = self.gather_rollout_data(
            total_batch_list=total_batch_list,
            episode_rewards=total_episode_rewards,
            episode_lengths=total_episode_lengths,
            traj_uid=total_traj_uid,
        )
        
        # logger.warning(f"Collected {len(gen_batch_output.batch['input_ids'])} trajectories. Gen_batch_output details: {gen_batch_output.batch.items()}, {gen_batch_output.non_tensor_batch.items()}")

        return gen_batch_output