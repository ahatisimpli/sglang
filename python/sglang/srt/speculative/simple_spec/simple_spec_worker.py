import torch
from sglang.srt.speculative.simple_spec.simple_spec_utils import SimpleSpecDraftInput, SimpleSpecVerifyInput
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode

from sglang.srt.speculative.simple_spec.simple_spec_utils import SimpleSpecPerformanceMonitor

from .simple_spec_utils import SimpleSpecDraftInput, SimpleSpecVerifyInput, SimpleSpecPerformanceMonitor

from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

from .simple_spec_cuda_graph_runner import SimpleSpecCudaGraphRunner





class SimpleSpecWorker:

    def __init__(self, server_args, gpu_id, tp_rank, pp_rank, dp_rank, nccl_port, target_worker):  
        # ... existing initialization ...  
        self.server_args = server_args  
        self.num_draft_tokens = server_args.simple_spec_num_draft_tokens  
        self.acceptance_threshold = server_args.simple_spec_acceptance_threshold  
        self.target_worker = target_worker 
        self.performance_monitor = SimpleSpecPerformanceMonitor()  
        self.gpu_id = gpu_id  # Add this line  
        self.device = server_args.device  # Add this line  
        
        # Share memory pools with target worker (critical!)  
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (  
            target_worker.get_memory_pool()  
        )  

            # Initialize attention backend and cuda graphs  
        self.draft_model_runner.server_args.disable_cuda_graph = (  
              server_args.disable_cuda_graph  
        )  
        
        # Initialize as draft worker with shared memory pools  
        super().__init__(  
            server_args=server_args,  
            gpu_id=gpu_id,  
            tp_rank=tp_rank,  
            pp_rank=pp_rank,  
            dp_rank=dp_rank,  
            nccl_port=nccl_port,  
            is_draft_worker=True,  
            req_to_token_pool=self.req_to_token_pool,  
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,  
        )

      
       # Use empty context for simple spec (no complex TP context needed)  
        from sglang.srt.utils import empty_context  
        with empty_context():  
            self.init_attention_backend()  
            self.init_cuda_graphs()


    def draft(self, batch: ScheduleBatch) -> SimpleSpecDraftInput:  
        """Generate draft tokens using the draft model."""  
        draft_tokens = []  
        draft_probs = []  
        
        current_batch = batch  
        for step in range(self.num_draft_tokens):  
            # Get forward batch  
            model_worker_batch = current_batch.get_model_worker_batch()  
            forward_batch = ForwardBatch.init_new(model_worker_batch, self.draft_model_runner)  
            
            # Try CUDA graph first, fallback to regular forward  
            can_cuda_graph = self.cuda_graph_runner and self.cuda_graph_runner.can_run(forward_batch)  
            
            if can_cuda_graph:  
                logits_output = self.cuda_graph_runner.replay(forward_batch)  
            else:  
                # Initialize attention backend if not using CUDA graph  
                if hasattr(self.draft_attn_backend, 'init_forward_metadata'):  
                    self.draft_attn_backend.init_forward_metadata(forward_batch)  
                logits_output = self.model_runner.forward(current_batch)  
            
            next_token = torch.argmax(logits_output.next_token_logits, dim=-1)  
            next_prob = torch.softmax(logits_output.next_token_logits, dim=-1)  
            
            draft_tokens.append(next_token)  
            draft_probs.append(next_prob)  
            
            # Update batch for next iteration  
            current_batch = self._update_batch_with_token(current_batch, next_token)  
        
        return SimpleSpecDraftInput(  
            draft_tokens=torch.stack(draft_tokens, dim=1),  
            draft_probs=torch.stack(draft_probs, dim=1),  
            num_draft_tokens=len(draft_tokens)  
        )
    def verify(self, input_ids, draft_input: SimpleSpecDraftInput, **kwargs):
        # Run the target model to verify the draft tokens
        # Accept greedily left-to-right
        batch_size = input_ids.shape[0]
        accepted_tokens = []
        num_accepted = []
        if draft_input.draft_tokens is None:
            # No draft tokens generated
            accepted_tokens_padded = torch.zeros((batch_size, 0), dtype=torch.long)
            return SimpleSpecVerifyInput(
                draft_tokens=None,
                target_probs=None,
                accepted_tokens=accepted_tokens_padded,
                num_accepted=0,
            )
        for i in range(batch_size):
            prefix = input_ids[i].unsqueeze(0)
            tokens = draft_input.draft_tokens[i]
            accepted = []
            for j, token in enumerate(tokens):
                tgt_input = torch.cat([prefix, tokens[:j+1].unsqueeze(0)], dim=1)
                with torch.no_grad():
                    logits = self.target_model(tgt_input)[:, -1, :]
                    pred = logits.argmax(-1)
                if pred.item() == token.item():
                    accepted.append(token.item())
                else:
                    break
            accepted_tokens.append(torch.tensor(accepted, dtype=torch.long))
            num_accepted.append(len(accepted))
        # Pad accepted_tokens to max length
        max_len = max(num_accepted) if num_accepted else 0
        accepted_tokens_padded = torch.zeros((batch_size, max_len), dtype=torch.long)
        for i, tokens in enumerate(accepted_tokens):
            if len(tokens) > 0:
                accepted_tokens_padded[i, :len(tokens)] = tokens
        return SimpleSpecVerifyInput(
            draft_tokens=draft_input.draft_tokens,
            target_probs=None,  # Could be filled if needed
            accepted_tokens=accepted_tokens_padded,
            num_accepted=max(num_accepted) if num_accepted else 0,
        )



    def _update_batch_with_token(self, batch: ScheduleBatch, next_token: torch.Tensor) -> ScheduleBatch:  
        """Update batch with new token for autoregressive drafting."""  
        import torch  
        from sglang.srt.model_executor.forward_batch_info import ForwardMode  
        
        # Create new input_ids by appending the new token  
        if next_token.dim() == 1:  
            next_token = next_token.unsqueeze(-1)  
        
        new_input_ids = torch.cat([batch.input_ids, next_token], dim=-1)  
        
        # Create a new batch to avoid modifying the original  
        updated_batch = batch.copy()  
        updated_batch.input_ids = new_input_ids  
        updated_batch.seq_lens = batch.seq_lens + 1  
        
        # Update positions for next forward pass  
        batch_size = new_input_ids.size(0)  
        seq_len = new_input_ids.size(1)  
        updated_batch.positions = torch.arange(seq_len, device=new_input_ids.device).unsqueeze(0).expand(batch_size, -1)  
        
        # Update output cache locations if needed  
        if hasattr(updated_batch, 'out_cache_loc') and updated_batch.out_cache_loc is not None:  
            # Allocate new cache locations for the new tokens  
            updated_batch.out_cache_loc = torch.cat([  
                batch.out_cache_loc,   
                batch.out_cache_loc + 1  # Simple increment for new positions  
            ], dim=-1)  
        
        return updated_batch


    def _create_verify_batch(self, batch: ScheduleBatch, draft_tokens: torch.Tensor) -> ScheduleBatch:  
        """Create verification batch with draft tokens concatenated."""  
        import torch  
        from sglang.srt.model_executor.forward_batch_info import ForwardMode  
        
        # Ensure draft_tokens has the right shape (batch_size, num_draft_tokens)  
        if draft_tokens.dim() == 1:  
            draft_tokens = draft_tokens.unsqueeze(0)  
        
        batch_size = batch.input_ids.size(0)  
        num_draft_tokens = draft_tokens.size(1)  
        
        # Concatenate original input with all draft tokens  
        verify_input_ids = torch.cat([batch.input_ids, draft_tokens], dim=-1)  
        
        # Create copy of batch for verification  
        verify_batch = batch.copy()  
        verify_batch.input_ids = verify_input_ids  
        verify_batch.seq_lens = batch.seq_lens + num_draft_tokens  
        verify_batch.forward_mode = ForwardMode.TARGET_VERIFY  
        
        # Update positions for the entire sequence  
        total_seq_len = verify_input_ids.size(1)  
        verify_batch.positions = torch.arange(total_seq_len, device=verify_input_ids.device).unsqueeze(0).expand(batch_size, -1)  
        
        # Update sequence length sum  
        verify_batch.seq_lens_sum = verify_batch.seq_lens.sum().item()  
        
        # Update output cache locations for verification  
        if hasattr(verify_batch, 'out_cache_loc') and verify_batch.out_cache_loc is not None:  
            # Extend cache locations for draft tokens  
            draft_cache_locs = torch.arange(  
                batch.out_cache_loc.max() + 1,  
                batch.out_cache_loc.max() + 1 + num_draft_tokens,  
                device=batch.out_cache_loc.device  
            ).unsqueeze(0).expand(batch_size, -1)  
            
            verify_batch.out_cache_loc = torch.cat([batch.out_cache_loc, draft_cache_locs], dim=-1)  
        
        # Set appropriate flags for verification  
        verify_batch.return_logprob = batch.return_logprob  
        verify_batch.return_hidden_states = False  
        
        return verify_batch


    def _find_acceptance_length(self, accepted_mask: torch.Tensor) -> int:  
        """Find first rejection point in acceptance mask."""  
        import torch  
        
        # Handle different input shapes  
        if accepted_mask.dim() == 0:  
            # Single boolean value  
            return 1 if accepted_mask.item() else 0  
        
        if accepted_mask.dim() == 2:  
            # Batch dimension - take first batch item for simplicity  
            # In a more sophisticated implementation, you might handle each batch item separately  
            accepted_mask = accepted_mask[0]  
        
        # Find first False in the mask (first rejection)  
        if accepted_mask.dtype != torch.bool:  
            # Convert to boolean if needed  
            accepted_mask = accepted_mask.bool()  
        
        # Find positions where mask is False (rejected tokens)  
        rejected_positions = (~accepted_mask).nonzero(as_tuple=True)[0]  
        
        if len(rejected_positions) > 0:  
            # Return position of first rejection  
            return rejected_positions[0].item()  
        else:  
            # All tokens accepted  
            return accepted_mask.size(0)


    def init_attention_backend(self):  
        """Initialize attention backend for simple speculative decoding."""  
        from sglang.srt.utils import empty_context  
        
        # Simple spec doesn't need complex multi-step backends like EAGLE  
        # We'll use the standard attention backend from the target worker  
        self.draft_attn_backend = None  
        self.draft_extend_attn_backend = None  
        
        # Get the attention backend type from server args  
        if self.server_args.attention_backend == "flashinfer":  
            from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend  
            
            self.draft_attn_backend = FlashInferAttnBackend(  
                self.draft_model_runner,  
                skip_prefill=False,  
            )  
            
        elif self.server_args.attention_backend == "triton":  
            from sglang.srt.layers.attention.triton_backend import TritonAttnBackend  
            
            self.draft_attn_backend = TritonAttnBackend(  
                self.draft_model_runner,  
                skip_prefill=False,  
            )  
            
        elif self.server_args.attention_backend == "fa3":  
            from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend  
            
            self.draft_attn_backend = FlashAttentionBackend(  
                self.draft_model_runner,  
                skip_prefill=False,  
            )  
            
        elif self.server_args.attention_backend == "torch_native":  
            from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend  
            
            self.draft_attn_backend = TorchNativeAttnBackend(  
                self.draft_model_runner,  
            )  
            
        else:  
            raise ValueError(  
                f"Simple spec is not supported with attention backend {self.server_args.attention_backend}"  
            )  
        
        # Set the backend on the model runner  
        self.draft_model_runner.attn_backend = self.draft_attn_backend  
  
    def init_cuda_graphs(self):  
        """Initialize CUDA graphs for simple speculative decoding."""  
        import time  
        from sglang.srt.utils import get_available_gpu_memory  
        import logging  
        
        logger = logging.getLogger(__name__)  
        
        self.cuda_graph_runner = None  
        
        if self.server_args.disable_cuda_graph:  
            return  
        
        # Simple spec uses basic CUDA graph runner (not the complex EAGLE one)  
        tic = time.perf_counter()  
        before_mem = get_available_gpu_memory(self.device, self.gpu_id)  
        logger.info(  
            f"Capture simple spec cuda graph begin. avail mem={before_mem:.2f} GB"  
        )  
        
        # Create a simple CUDA graph runner for basic operations  
        self.cuda_graph_runner = SimpleSpecCudaGraphRunner(self)  
        
        after_mem = get_available_gpu_memory(self.device, self.gpu_id)  
        logger.info(  
            f"Capture simple spec cuda graph end. Time elapsed: {time.perf_counter() - tic:.2f} s. "  
            f"mem usage={(before_mem - after_mem):.2f} GB. avail mem={after_mem:.2f} GB."  
        )

    def handle_forward_mode(self, batch: ScheduleBatch, forward_batch: ForwardBatch):  
        """Handle different forward modes for simple speculative decoding."""  
        from sglang.srt.model_executor.forward_batch_info import ForwardMode  
        
        if forward_batch.forward_mode == ForwardMode.EXTEND:  
            # Handle prefill/extend mode - no speculation needed  
            return self.target_worker.forward_batch_generation(batch)  
        
        elif forward_batch.forward_mode == ForwardMode.DECODE:  
            # Handle decode mode - use speculation  
            return self.forward_batch_speculative_generation(batch)  
        
        elif forward_batch.forward_mode == ForwardMode.TARGET_VERIFY:  
            # Handle verification mode  
            return self.target_worker.model_runner.forward(forward_batch)  
        
        elif forward_batch.forward_mode == ForwardMode.DRAFT_EXTEND:  
            # Handle draft extend mode  
            return self.draft_model_runner.forward(forward_batch)  
        
        else:  
            raise ValueError(f"Unsupported forward mode: {forward_batch.forward_mode}")  
    
    def get_model_worker_batch(self, batch: ScheduleBatch):  
        """Create ModelWorkerBatch with proper forward mode handling."""  
        model_worker_batch = batch.get_model_worker_batch()  
        
        # Set appropriate forward mode based on batch state  
        if batch.forward_mode.is_extend():  
            model_worker_batch.forward_mode = ForwardMode.EXTEND  
        else:  
            model_worker_batch.forward_mode = ForwardMode.DECODE  
        
        # Add simple spec specific metadata  
        model_worker_batch.spec_algorithm = SpeculativeAlgorithm.SIMPLE_SPEC  
        model_worker_batch.spec_num_draft_tokens = self.num_draft_tokens  
        
        return model_worker_batch


    def validate_batch(self, batch: ScheduleBatch):  
        """Validate batch for simple speculative decoding."""  
        if batch is None:  
            raise ValueError("Batch cannot be None")  
        
        if len(batch.reqs) == 0:  
            raise ValueError("Batch cannot be empty")  
        
        if batch.input_ids is None or batch.input_ids.size(0) == 0:  
            raise ValueError("Batch input_ids cannot be empty")  
        
        if batch.seq_lens is None or batch.seq_lens.size(0) == 0:  
            raise ValueError("Batch seq_lens cannot be empty")  
        
        # Check for memory allocation issues  
        try:  
            available_memory = self.token_to_kv_pool_allocator.available_size()  
            required_memory = len(batch.reqs) * self.num_draft_tokens  
            
            if available_memory < required_memory:  
                raise RuntimeError(  
                    f"Insufficient memory for speculation. "  
                    f"Available: {available_memory}, Required: {required_memory}"  
                )  
        except Exception as e:  
            raise RuntimeError(f"Memory allocation check failed: {e}")  
    
    def validate_draft_tokens(self, draft_tokens: torch.Tensor):  
        """Validate draft tokens."""  
        if draft_tokens is None:  
            raise ValueError("Draft tokens cannot be None")  
        
        if draft_tokens.size(0) == 0:  
            raise ValueError("Draft tokens cannot be empty")  
        
        if draft_tokens.size(1) != self.num_draft_tokens:  
            raise ValueError(  
                f"Expected {self.num_draft_tokens} draft tokens, "  
                f"got {draft_tokens.size(1)}"  
            )  
        
        # Check for invalid token IDs  
        vocab_size = self.target_worker.model_runner.model_config.vocab_size  
        if (draft_tokens >= vocab_size).any() or (draft_tokens < 0).any():  
            raise ValueError(  
                f"Invalid token IDs in draft tokens. "  
                f"Must be in range [0, {vocab_size})"  
            )  
    
    def safe_forward_batch_speculative_generation(self, batch: ScheduleBatch):  
        """Safe wrapper for speculative generation with error handling."""  
        try:  
            # Validate inputs  
            self.validate_batch(batch)  
            
            # Perform speculative generation  
            return self.forward_batch_speculative_generation(batch)  
            
        except torch.cuda.OutOfMemoryError as e:  
            # Handle CUDA OOM  
            torch.cuda.empty_cache()  
            raise RuntimeError(f"CUDA out of memory during speculation: {e}")  
            
        except Exception as e:  
            # Log error and fallback to target model only  
            import logging  
            logger = logging.getLogger(__name__)  
            logger.warning(f"Speculation failed, falling back to target model: {e}")  
            
            # Fallback to non-speculative generation  
            return self.target_worker.forward_batch_generation(batch)


  
    def forward_batch_speculative_generation(self, batch: ScheduleBatch):  
        """Enhanced version with performance monitoring."""  
        import time  
        
        if batch.forward_mode.is_extend():  
            return self.target_worker.forward_batch_generation(batch)  
        
        # Draft phase with timing  
        draft_start = time.perf_counter()  
        spec_info = self.draft(batch)  
        draft_time = time.perf_counter() - draft_start  
        
        # Verify phase with timing  
        verify_start = time.perf_counter()  
        logits_output, verify_output, model_worker_batch, can_run_cuda_graph = self.verify(batch, spec_info)  
        verify_time = time.perf_counter() - verify_start  
        
        # Record performance metrics  
        self.performance_monitor.record_speculation_round(  
            num_draft_tokens=spec_info.num_draft_tokens,  
            num_accepted_tokens=verify_output.num_accepted,  
            draft_time=draft_time,  
            verify_time=verify_time  
        )  
        
        return logits_output, verify_output.accepted_tokens, model_worker_batch.bid, verify_output.num_accepted, can_run_cuda_graph

    def get_performance_metrics(self):  
        """Get current performance metrics."""  
        return self.performance_monitor.get_metrics_dict()