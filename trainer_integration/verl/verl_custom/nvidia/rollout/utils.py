import logging

import torch

logger = logging.getLogger(__name__)


def convert_right_padding_to_left(
    tokenizer, input_ids, attention_mask, device, max_len=None
):
    """
    Converts right-padded tensors to left-padded tensors with optional custom length.

    Args:
        tokenizer: The tokenizer object with pad_token_id attribute
        input_ids (torch.Tensor): Right-padded input IDs tensor of shape [batch_size, seq_length]
        attention_mask (torch.Tensor): Right-padded attention mask tensor of shape [batch_size, seq_length]
        device: The device to place the new tensors on
        max_len (int, optional): The desired maximum length of the returned tensors.
                                If None, uses the original sequence length.

    Returns:
        tuple: (left_padded_input_ids, left_padded_attention_mask)
    """
    batch_size, orig_seq_length = input_ids.size()

    # Use original length if max_len is not specified
    seq_length = max_len if max_len is not None else orig_seq_length

    # Create new tensors with the desired size
    left_padded_input_ids = torch.full(
        (batch_size, seq_length),
        tokenizer.pad_token_id,
        dtype=input_ids.dtype,
        device=device,
    )
    left_padded_attention_mask = torch.zeros(
        (batch_size, seq_length), dtype=attention_mask.dtype, device=device
    )

    for i in range(batch_size):
        # Get the non-padded length of this sequence
        seq_len = attention_mask[i].sum().item()

        # Contract: caller passes ``max_len = max_starting_message_length``
        # (the rollout-side seed-slot width). Seeds are system + dataset
        # instance prompt; the empirical SWE-Gym cap is 12000 tokens. A
        # seq_len > seq_length here means the dataset filter or the seed
        # construction regressed — surface loudly rather than silently
        # right-trimming the prompt and shifting the offset arithmetic.
        if seq_len > seq_length:
            raise RuntimeError(
                f'convert_right_padding_to_left: seq_len={seq_len} > '
                f'max_len={seq_length} for batch item {i}; rollout-side '
                f'seed-slot contract violated (max_starting_message_length).'
            )

        # Calculate the offset for left padding
        offset = seq_length - seq_len

        # Copy the non-padded tokens to the end
        left_padded_input_ids[i, offset:] = input_ids[i, :seq_len]
        left_padded_attention_mask[i, offset:] = (
            1  # Set attention mask for non-padding tokens
        )

    return left_padded_input_ids, left_padded_attention_mask


def pad_to_max_length_right(tokenizer, encodings, max_length, device):
    """
    Pads tokenizer outputs to a specific maximum length with configurable padding side.

    Args:
        tokenizer: The tokenizer object with pad_token_id attribute
        encodings (dict): Dictionary containing 'input_ids', 'attention_mask', and optionally 'assistant_masks'
        max_length (int): The desired maximum length to pad to
        device: The device to place the tensors on

    Returns:
        dict: Dictionary with padded tensors for 'input_ids', 'attention_mask', and 'assistant_masks' if present
    """
    batch_size = len(encodings['input_ids'])

    # Initialize output tensors
    padded_input_ids = torch.full(
        (batch_size, max_length),
        tokenizer.pad_token_id,
        dtype=torch.long,
        device=device,
    )
    padded_attention_mask = torch.zeros(
        (batch_size, max_length), dtype=torch.long, device=device
    )
    padded_assistant_mask = torch.zeros(
        (batch_size, max_length), dtype=torch.long, device=device
    )
    padded_log_probs = torch.zeros(
        (batch_size, max_length), dtype=torch.float, device=device
    )

    # Fill tensors with actual values
    for i in range(batch_size):
        seq_len = (
            encodings['attention_mask'][i].sum().item()
            if isinstance(encodings['attention_mask'][i], torch.Tensor)
            else sum(encodings['attention_mask'][i])
        )
        # Contract: caller passes ``max_length = self.total_len`` (rollout's
        # ``max_prompt_length + max_response_length``), which equals vLLM's
        # ``max_model_len``. vLLM enforces ``seed + body <= max_model_len``
        # per call, so the packed sequence (prompt + response) cannot exceed
        # ``max_length`` here. A seq_len > max_length means vLLM violated
        # its own bound — surface loudly rather than silently right-trimming
        # tokens the reward was already scored on.
        if seq_len > max_length:
            raise RuntimeError(
                f'pad_to_max_length_right: seq_len={seq_len} > '
                f'max_length={max_length} for batch item {i}; vLLM '
                f'max_model_len contract violated.'
            )
        actual_len = seq_len

        # Right padding - copy sequence data to the beginning
        padded_input_ids[i, :actual_len] = torch.tensor(
            encodings['input_ids'][i][:actual_len], device=device
        )
        padded_attention_mask[i, :actual_len] = torch.tensor(
            encodings['attention_mask'][i][:actual_len], device=device
        )
        padded_assistant_mask[i, :actual_len] = torch.tensor(
            encodings['assistant_masks'][i][:actual_len], device=device
        )
        padded_log_probs[i, :actual_len] = torch.tensor(
            encodings['log_probs'][i][:actual_len], device=device
        )

    return (
        padded_input_ids,
        padded_attention_mask,
        padded_assistant_mask,
        padded_log_probs,
    )


import hashlib
import uuid


def get_unique_id(instance, existing_ids=None):
    base = f'{instance["instance_id"]}_{instance["trajectory_id"]}'
    base_hash = hashlib.sha256(base.encode('utf-8')).hexdigest()[:16]
    while True:
        rand = uuid.uuid4().hex[:8]
        uid = f'{base_hash}_{rand}'
        if uid not in existing_ids:
            return uid
