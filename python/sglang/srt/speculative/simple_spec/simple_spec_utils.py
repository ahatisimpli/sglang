from dataclasses import dataclass  
from typing import Optional  
import torch  
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode  
  
@dataclass  
class SimpleSpecDraftInput:  
    draft_tokens: Optional[torch.Tensor] = None  
    draft_probs: Optional[torch.Tensor] = None  
    num_draft_tokens: int = 0  
    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.NULL  
  
@dataclass  
class SimpleSpecVerifyInput:  
    draft_tokens: Optional[torch.Tensor] = None  
    target_probs: Optional[torch.Tensor] = None  
    accepted_tokens: Optional[torch.Tensor] = None  
    num_accepted: int = 0



    class SimpleSpecPerformanceMonitor:  
        """Performance monitoring for simple speculative decoding."""  
        
        def __init__(self):  
            self.total_draft_tokens = 0  
            self.total_accepted_tokens = 0  
            self.total_speculation_rounds = 0  
            self.total_time_drafting = 0.0  
            self.total_time_verification = 0.0  
            
        def record_speculation_round(self, num_draft_tokens: int, num_accepted_tokens: int,   
                                draft_time: float, verify_time: float):  
            """Record metrics for a speculation round."""  
            self.total_draft_tokens += num_draft_tokens  
            self.total_accepted_tokens += num_accepted_tokens  
            self.total_speculation_rounds += 1  
            self.total_time_drafting += draft_time  
            self.total_time_verification += verify_time  
        
        def get_acceptance_rate(self) -> float:  
            """Get current acceptance rate."""  
            if self.total_draft_tokens == 0:  
                return 0.0  
            return self.total_accepted_tokens / self.total_draft_tokens  
        
        def get_speedup_ratio(self) -> float:  
            """Get estimated speedup ratio."""  
            if self.total_speculation_rounds == 0:  
                return 1.0  
            
            avg_accepted = self.total_accepted_tokens / self.total_speculation_rounds  
            return max(1.0, avg_accepted)  # At least 1x speedup  
        
        def get_metrics_dict(self) -> dict:  
            """Get all metrics as dictionary."""  
            return {  
                "acceptance_rate": self.get_acceptance_rate(),  
                "speedup_ratio": self.get_speedup_ratio(),  
                "total_draft_tokens": self.total_draft_tokens,  
                "total_accepted_tokens": self.total_accepted_tokens,  
                "total_speculation_rounds": self.total_speculation_rounds,  
                "avg_draft_time": self.total_time_drafting / max(1, self.total_speculation_rounds),  
                "avg_verify_time": self.total_time_verification / max(1, self.total_speculation_rounds),  
            }