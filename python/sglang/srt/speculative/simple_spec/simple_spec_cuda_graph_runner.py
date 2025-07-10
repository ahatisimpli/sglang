import bisect  
import torch  
from typing import TYPE_CHECKING  
  
from sglang.srt.model_executor.cuda_graph_runner import (  
    CUDA_GRAPH_CAPTURE_FAILED_MSG,  
    get_batch_sizes_to_capture,  
    get_global_graph_memory_pool,  
    model_capture_mode,  
    set_global_graph_memory_pool,  
    set_torch_compile_config,  
)  
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode  
from sglang.srt.layers.logits_processor import LogitsProcessorOutput  
  
if TYPE_CHECKING:  
    from sglang.srt.speculative.simple_spec.simple_spec_worker import SimpleSpecWorker  
  
import logging  
logger = logging.getLogger(__name__)  
  
  
class SimpleSpecCudaGraphRunner:  
    """CUDA Graph runner for simple speculative decoding."""  
      
    def __init__(self, simple_spec_worker: "SimpleSpecWorker"):  
        self.simple_spec_worker = simple_spec_worker  
        self.model_runner = simple_spec_worker.draft_model_runner  
        self.graphs = {}  
        self.output_buffers = {}  
        self.enable_torch_compile = self.model_runner.server_args.enable_torch_compile  
        self.disable_padding = self.model_runner.server_args.disable_cuda_graph_padding  
        self.num_draft_tokens = simple_spec_worker.num_draft_tokens  
          
        # Get batch sizes to capture  
        self.capture_bs, self.compile_bs = get_batch_sizes_to_capture(self.model_runner)  
        self.max_bs = max(self.capture_bs)  
          
        # For simple spec, we capture graphs for single token generation  
        self.num_tokens_per_bs = 1  
        self.max_num_token = self.max_bs * self.num_tokens_per_bs  
          
        # Initialize attention backend state  
        if hasattr(self.model_runner.attn_backend, 'init_cuda_graph_state'):  
            self.model_runner.attn_backend.init_cuda_graph_state(  
                self.max_bs, self.max_num_token  
            )  
          
        if self.enable_torch_compile:  
            set_torch_compile_config()  
          
        # Graph inputs - simplified for basic speculative decoding  
        with torch.device("cuda"):  
            self.input_ids = torch.zeros((self.max_num_token,), dtype=torch.int64)  
            self.req_pool_indices = torch.zeros((self.max_bs,), dtype=torch.int32)  
            self.seq_lens = torch.ones((self.max_bs,), dtype=torch.int32)  
            self.out_cache_loc = torch.zeros((self.max_num_token,), dtype=torch.int64)  
            self.positions = torch.zeros((self.max_num_token,), dtype=torch.int64)  
          
        # Capture graphs  
        try:  
            with model_capture_mode():  
                self.capture()  
        except RuntimeError as e:  
            raise Exception(  
                f"Capture simple spec cuda graph failed: {e}\n{CUDA_GRAPH_CAPTURE_FAILED_MSG}"  
            )  
      
    def capture(self):  
        """Capture CUDA graphs for different batch sizes."""  
        for bs in self.capture_bs:  
            self.capture_one_batch_size(bs)  
      
    def capture_one_batch_size(self, bs: int):  
        """Capture CUDA graph for a specific batch size."""  
          
        def run_once():  
            # Simple forward pass for draft token generation  
            logits_output = self.model_runner.model.forward(  
                self.input_ids[:bs],  
                self.positions[:bs],   
                ForwardBatch(  
                    input_ids=self.input_ids[:bs],  
                    positions=self.positions[:bs],  
                    req_pool_indices=self.req_pool_indices[:bs],  
                    seq_lens=self.seq_lens[:bs],  
                    out_cache_loc=self.out_cache_loc[:bs],  
                    forward_mode=ForwardMode.DECODE,  
                    batch_size=bs,  
                    seq_lens_sum=bs,  # Simple case: one token per sequence  
                )  
            )  
            return logits_output  
          
        # Warmup runs  
        for _ in range(3):  
            torch.cuda.synchronize()  
            run_once()  
          
        # Capture the graph  
        graph = torch.cuda.CUDAGraph()  
        stream = torch.cuda.Stream()  
          
        with torch.cuda.graph(graph, pool=get_global_graph_memory_pool(), stream=stream):  
            output = run_once()  
          
        set_global_graph_memory_pool(graph.pool())  
          
        self.graphs[bs] = graph  
        self.output_buffers[bs] = output  
          
        logger.info(f"Captured CUDA graph for simple spec batch size {bs}")  
      
    def can_run(self, forward_batch: ForwardBatch) -> bool:  
        """Check if CUDA graph can be used for this batch."""  
        bs = forward_batch.batch_size  
          
        # Check if batch size is supported  
        is_bs_supported = (  
            bs in self.graphs  
            if self.disable_padding  
            else bs <= self.max_bs  
        )  
          
        # Simple spec only supports decode mode  
        is_mode_supported = forward_batch.forward_mode.is_decode()  
          
        # Check if it's a single token per sequence (simple case)  
        is_simple_case = forward_batch.input_ids.size(0) == bs  
          
        return is_bs_supported and is_mode_supported and is_simple_case  
      
    def replay(self, forward_batch: ForwardBatch) -> LogitsProcessorOutput:  
        """Replay CUDA graph for the given batch."""  
        bs = forward_batch.batch_size  
          
        # Find appropriate batch size  
        if self.disable_padding:  
            graph_bs = bs  
        else:  
            index = bisect.bisect_left(self.capture_bs, bs)  
            graph_bs = self.capture_bs[index]  
          
        # Copy input data  
        self.input_ids[:bs].copy_(forward_batch.input_ids)  
        self.positions[:bs].copy_(forward_batch.positions)  
        self.req_pool_indices[:bs].copy_(forward_batch.req_pool_indices)  
        self.seq_lens[:bs].copy_(forward_batch.seq_lens)  
        self.out_cache_loc[:bs].copy_(forward_batch.out_cache_loc)  
          
        # Initialize attention backend metadata if needed  
        if hasattr(self.model_runner.attn_backend, 'init_forward_metadata_replay_cuda_graph'):  
            self.model_runner.attn_backend.init_forward_metadata_replay_cuda_graph(  
                bs=graph_bs,  
                req_pool_indices=self.req_pool_indices,  
                seq_lens=self.seq_lens,  
                seq_lens_sum=forward_batch.seq_lens_sum,  
                encoder_lens=None,  
                forward_mode=ForwardMode.DECODE,  
                spec_info=None,  
            )  
          
        # Replay the graph  
        self.graphs[graph_bs].replay()  
        output = self.output_buffers[graph_bs]  
          
        # Return appropriate slice if padding was used  
        if graph_bs != bs:  
            return LogitsProcessorOutput(  
                next_token_logits=output.next_token_logits[:bs],  
                hidden_states=output.hidden_states[:bs] if hasattr(output, 'hidden_states') else None,  
            )  
          
        return output