# =============================================================================
# CREDENCE: Dynamic Head Configuration for Ablation Studies
# =============================================================================
#
# Provides flexible head configuration generation with geometric dropout spacing
# for principled ensemble diversity in ablation studies.
#
# =============================================================================

from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class HeadConfig:
    """Configuration for a single ensemble head."""
    head_id: int
    name: str
    hidden_dim: int = 256
    dropout_rate: float = 0.1
    pooling: str = "cls"  # "cls" | "mean" | "last"
    
    def __repr__(self):
        return f"Head({self.name}, d={self.dropout_rate:.3f}, pool={self.pooling})"


def generate_head_configs(
    n_heads: int,
    d_min: float = 0.05,
    d_max: float = 0.30,
    hidden_dim: int = 256,
    use_pooling_diversity: bool = True,
) -> List[HeadConfig]:
    """
    Generate diverse head configurations with geometric dropout spacing.
    
    The geometric spacing formula ensures even coverage of the dropout range
    in log-space, which is more principled than linear spacing for probabilities.
    
    Formula: d_h = 1 - exp(log(1-d_max) + ((h-1)/(H-1)) * [log(1-d_min) - log(1-d_max)])
    
    Args:
        n_heads: Number of ensemble heads (H)
        d_min: Minimum dropout rate
        d_max: Maximum dropout rate  
        hidden_dim: Hidden dimension for all heads
        use_pooling_diversity: If True, alternate between CLS and mean pooling
        
    Returns:
        List of HeadConfig objects
    """
    configs = []
    
    for h in range(n_heads):
        # Geometric dropout spacing
        if n_heads == 1:
            dropout = (d_min + d_max) / 2
        else:
            # Formula from paper: geometric spacing in log-space
            log_keep_max = np.log(1 - d_max)
            log_keep_min = np.log(1 - d_min)
            log_keep_h = log_keep_max + (h / (n_heads - 1)) * (log_keep_min - log_keep_max)
            dropout = 1 - np.exp(log_keep_h)
        
        # Pooling diversity (alternate between CLS and mean)
        if use_pooling_diversity:
            pooling = "cls" if h % 2 == 0 else "mean"
        else:
            pooling = "cls"
        
        configs.append(HeadConfig(
            head_id=h,
            name=f"head_{h}_d{dropout:.3f}_{pooling}",
            hidden_dim=hidden_dim,
            dropout_rate=dropout,
            pooling=pooling,
        ))
    
    return configs

