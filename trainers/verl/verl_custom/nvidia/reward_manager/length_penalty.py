from verl import DataProto
import torch
import json
import math
import multiprocessing as mp

class LengthPenalty:
    def __init__(self, config=None):
        self.config = config

    def _ngram_repetition_reward(self, args):
        sequence, ngram_size, penalty = args
        # Create ngram array using unfold
        ngram_array = sequence.unfold(0, ngram_size, 1)
        
        # Get unique ngrams and their counts
        unique_ngrams, counts = torch.unique(ngram_array, dim=0, return_counts=True)
        
        # Find positions of repeated ngrams
        repeated_mask = counts > 1
        if not repeated_mask.any():
            return torch.zeros(len(sequence), dtype=torch.float32)
            
        repeated_ngrams = unique_ngrams[repeated_mask]
        curr_reward = torch.zeros(len(sequence), dtype=torch.float32)
        
        if len(repeated_ngrams) > 0:
            # Find all occurrences of repeated ngrams
            for ng in repeated_ngrams:
                matches = (ngram_array == ng.unsqueeze(0)).all(dim=1)
                positions = torch.where(matches)[0]
                
                # Apply penalty to all occurrences except the first one
                for pos in positions[1:]:
                    curr_reward[pos] = penalty
        return curr_reward

    def get_repetition_reward(self, generations):
        bsz = generations.shape[0]
        ngram_size = self.config.repetition_ngram_size
        
        # Prepare arguments for multiprocessing
        args = [(generations[i], ngram_size, self.config.repetition_penalty) for i in range(bsz)]
        
        # Use multiprocessing to process sequences in parallel
        num_processes = min(256, bsz) 
        
        with mp.Pool(processes=num_processes) as pool:
            results = pool.map(self._ngram_repetition_reward, args)
        
        # Stack results
        curr_reward = torch.stack(results)
        return curr_reward

    def __call__(self, data:DataProto):
        if self.config is None or self.config.length_penalty_type == 'none':
            return data.batch['token_level_scores']

        with torch.no_grad():    
            prompt_ids = data.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            response_ids = data.batch['responses']
            response_length = response_ids.shape[-1]
            valid_response_length = data.batch['attention_mask'][:, prompt_length:].sum(dim=-1)
            response_mask = data.batch['attention_mask'][:, -response_length:]
            scores = data.batch['token_level_scores']
            data_sources = data.non_tensor_batch['data_source']

            modulated_scores = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
            
            if self.config.length_penalty_type == 'linear':
                modulated_scores =  scores * (1 - valid_response_length.unsqueeze(-1) / self.config.max_length)
            elif self.config.length_penalty_type == 'instance_linear':
                # Penalty is only applied to if prompt that have scores >= 1.0
                index = data.non_tensor_batch['uid']
                instance_scores = scores.sum(dim=-1)
                bsz = scores.shape[0]

                # get minimum length for each instance where score >= 1.0
                id2min = {}
                for i in range(bsz):
                    if instance_scores[i] >= 1.0:
                        idx = index[i]
                        if idx not in id2min:
                            id2min[idx] = valid_response_length[i]
                        else:
                            id2min[idx] = min(id2min[idx], valid_response_length[i])

                # get length coef 
                length_coef = []
                for i in range(bsz):
                    idx = index[i]
                    if idx not in id2min:
                        length_coef.append(1.0)
                    else:
                        coef = 1.0 - (valid_response_length[i] - id2min[idx]) / (self.config.max_length - id2min[idx])
                        length_coef.append(min(1.0, coef))

                modulated_scores[torch.arange(bsz), valid_response_length-1] = torch.tensor(length_coef, dtype=torch.float32)    
                modulated_scores = scores * modulated_scores  
            elif self.config.length_penalty_type == 'cosine':
                instance_scores = scores.sum(dim=-1)
                bsz = scores.shape[0]
                
                # Pre-compute masks
                is_binary_source = torch.tensor([ds in self.config.binary_score_data_sources for ds in data_sources], device=instance_scores.device)
                is_selected_source = torch.tensor([ds in self.config.selected_data_sources for ds in data_sources], device=instance_scores.device)
                
                # Convert scores to binary where needed
                cur_scores = torch.where(
                    is_binary_source & (instance_scores > 0.99),
                    torch.ones_like(instance_scores),
                    torch.where(
                        is_binary_source,
                        torch.zeros_like(instance_scores),
                        instance_scores
                    )
                )
                
                # Apply cosine length penalty only to selected sources
                if is_selected_source.any():
                    # Pre-compute progress values
                    progress = valid_response_length.float() / self.config.max_length
                    cosine = torch.cos(progress * math.pi)
                    
                    # Compute min and max values based on scores
                    min_values = torch.where(
                        cur_scores > 0.99,
                        torch.full_like(cur_scores, self.config.min_value_correct),
                        torch.full_like(cur_scores, self.config.max_value_wrong)
                    )
                    max_values = torch.where(
                        cur_scores > 0.99,
                        torch.full_like(cur_scores, self.config.max_value_correct),
                        torch.full_like(cur_scores, self.config.min_value_wrong)
                    )
                    
                    # Apply cosine formula
                    r = min_values + 0.5 * (max_values - min_values) * (1.0 + cosine)
                    
                    # Apply linear penalty for low scores and long generations
                    long_gen_mask = (cur_scores <= 0.99) & (valid_response_length + self.config.max_length_margin > self.config.max_length)
                    if long_gen_mask.any():
                        linear_penalty = (max_values - min_values) / self.config.max_length_margin * \
                                       (valid_response_length - self.config.max_length + self.config.max_length_margin) + min_values
                        r = torch.where(long_gen_mask, linear_penalty, r)
                    
                    # Only apply to selected sources
                    cur_scores = torch.where(is_selected_source, r, cur_scores)

                # Add repetition penalty all responses
                modulated_scores = self.get_repetition_reward(response_ids)
                modulated_scores = torch.where(response_mask.bool(), modulated_scores, torch.zeros_like(modulated_scores))

                # Add length penalty
                modulated_scores[torch.arange(bsz), valid_response_length-1] += cur_scores
            else:
                raise ValueError(f"Length penalty type {self.config.length_penalty_type} not supported")
        
        return modulated_scores