import torch

def boost_high_score_advantages(advantages, scores, correct_sample_advantage_boost_value, correct_sample_advantage_boost_threshold):
    """Boost advantages for high-scoring samples
    
    Args:
        advantages: Original advantage values of shape [batch_size, response_length]
        scores: Corresponding scores for each sample of shape [batch_size]
        correct_sample_advantage_boost_value: Value to boost advantages for high-scoring samples
        
    Returns:
        Modified advantage values with boosted values for high-scoring samples
    """
    # Use torch.isclose for floating point comparison with a small tolerance
    high_score_mask = scores >= correct_sample_advantage_boost_threshold - 1e-5
    high_score_mask = high_score_mask.unsqueeze(-1).expand_as(advantages)
    
    # Boost advantages for high-scoring samples
    advantages = advantages + high_score_mask * correct_sample_advantage_boost_value
    return advantages


def apply_padding_mask(response_mask: torch.Tensor, is_padded: torch.Tensor, log_stats: bool = False) -> torch.Tensor:
    """
    Apply padding mask to response mask to exclude padded samples from loss computation.
    
    Args:
        response_mask (torch.Tensor): Original response mask of shape (batch_size, response_length)
        is_padded (torch.Tensor): Boolean tensor of shape (batch_size,) indicating padded samples
        log_stats (bool): Whether to log masking statistics
        
    Returns:
        torch.Tensor: Modified response mask with padded samples masked out
    """
    if is_padded is None:
        return response_mask
    
    # Ensure is_padded is on the same device as response_mask
    if isinstance(is_padded, (list, tuple)):
        is_padded = torch.tensor(is_padded, device=response_mask.device, dtype=torch.bool)
    elif not isinstance(is_padded, torch.Tensor):
        is_padded = torch.tensor(is_padded, device=response_mask.device, dtype=torch.bool)
    else:
        is_padded = is_padded.to(response_mask.device).bool()
    
    # Log statistics if requested
    if log_stats:
        num_padded = is_padded.sum().item()
        total_samples = is_padded.size(0)
        if num_padded > 0:
            print(f"[PADDING MASK] Masking {num_padded}/{total_samples} samples ({num_padded/total_samples*100:.1f}%) due to padding")
    
    # Apply mask: set response_mask to 0 for padded samples
    # (~is_padded) creates a mask where True means NOT padded
    sample_mask = (~is_padded).float().unsqueeze(1)  # Shape: (batch_size, 1)
    masked_response_mask = response_mask * sample_mask
    
    return masked_response_mask