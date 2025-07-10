

import unittest  
import torch  
import requests  
from unittest.mock import Mock, patch  
  
from sglang.srt.speculative.simple_spec.simple_spec_worker import SimpleSpecWorker  
from sglang.srt.speculative.simple_spec.simple_spec_utils import SimpleSpecDraftInput, SimpleSpecVerifyInput  
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm  
from sglang.srt.server_args import ServerArgs  
from sglang.test.test_utils import (  
    DEFAULT_MODEL_NAME_FOR_TEST,  
    DEFAULT_URL_FOR_TEST,  
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,  
    popen_launch_server,  
    kill_process_tree,  
    CustomTestCase,  
)  
  
class TestSimpleSpecWorkerMethods(unittest.TestCase):  
    """Unit tests for SimpleSpecWorker methods."""  
      
    def setUp(self):  
        # Mock server args  
        self.server_args = Mock()  
        self.server_args.simple_spec_num_draft_tokens = 4  
        self.server_args.simple_spec_acceptance_threshold = 0.8  
        self.server_args.disable_cuda_graph = True  
          
        # Mock target worker  
        self.target_worker = Mock()  
        self.target_worker.get_memory_pool.return_value = (Mock(), Mock())  
          
    def test_find_acceptance_length(self):  
        """Test _find_acceptance_length method."""  
        worker = SimpleSpecWorker(  
            self.server_args, 0, 0, 0, 0, 12345, self.target_worker  
        )  
          
        # Test all accepted  
        accepted_mask = torch.tensor([True, True, True, True])  
        length = worker._find_acceptance_length(accepted_mask)  
        self.assertEqual(length, 4)  
          
        # Test partial acceptance  
        accepted_mask = torch.tensor([True, True, False, True])  
        length = worker._find_acceptance_length(accepted_mask)  
        self.assertEqual(length, 2)  
          
        # Test no acceptance  
        accepted_mask = torch.tensor([False, True, True, True])  
        length = worker._find_acceptance_length(accepted_mask)  
        self.assertEqual(length, 0)  
      
    def test_validate_draft_tokens(self):  
        """Test draft token validation."""  
        worker = SimpleSpecWorker(  
            self.server_args, 0, 0, 0, 0, 12345, self.target_worker  
        )  
          
        # Mock vocab size  
        worker.target_worker.model_runner.model_config.vocab_size = 1000  
          
        # Valid tokens  
        valid_tokens = torch.tensor([[1, 2, 3, 4]])  
        worker.validate_draft_tokens(valid_tokens)  # Should not raise  
          
        # Invalid tokens (out of vocab)  
        invalid_tokens = torch.tensor([[1, 2, 1000, 4]])  
        with self.assertRaises(ValueError):  
            worker.validate_draft_tokens(invalid_tokens)  
          
        # Wrong number of tokens  
        wrong_size_tokens = torch.tensor([[1, 2, 3]])  # Only 3 tokens  
        with self.assertRaises(ValueError):  
            worker.validate_draft_tokens(wrong_size_tokens)  
  
class TestSimpleSpecIntegration(CustomTestCase):  
    """Integration tests for simple speculative decoding."""  
      
    @classmethod  
    def setUpClass(cls):  
        cls.base_url = DEFAULT_URL_FOR_TEST  
        cls.process = popen_launch_server(  
            DEFAULT_MODEL_NAME_FOR_TEST,  
            cls.base_url,  
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,  
            other_args=[  
                "--speculative-algorithm", "SIMPLE_SPEC",  
                "--simple-spec-num-draft-tokens", "2",  
                "--simple-spec-acceptance-threshold", "0.7",  
                "--simple-spec-draft-model-path", DEFAULT_MODEL_NAME_FOR_TEST,  
            ],  
        )  
      
    @classmethod  
    def tearDownClass(cls):  
        kill_process_tree(cls.process.pid)  
      
    def test_simple_generation(self):  
        """Test basic text generation with simple spec."""  
        url = self.base_url + "/generate"  
        data = {  
            "text": "The capital of France is",  
            "sampling_params": {  
                "temperature": 0,  
                "max_new_tokens": 10,  
            },  
        }  
          
        response = requests.post(url, json=data)  
        self.assertEqual(response.status_code, 200)  
          
        result = response.json()  
        self.assertIn("text", result)  
        self.assertTrue(len(result["text"]) > len(data["text"]))  
      
    def test_performance_metrics(self):  
        """Test that performance metrics are being tracked."""  
        # Generate some text to populate metrics  
        url = self.base_url + "/generate"  
        for _ in range(5):  
            data = {  
                "text": f"Test prompt {_}",  
                "sampling_params": {"temperature": 0, "max_new_tokens": 5},  
            }  
            requests.post(url, json=data)  
          
        # Check server info for metrics  
        info_response = requests.get(self.base_url + "/get_server_info")  
        self.assertEqual(info_response.status_code, 200)  
          
        server_info = info_response.json()  
        # Should contain simple spec metrics  
        self.assertIn("internal_states", server_info)  
  
class TestSimpleSpecPerformance(unittest.TestCase):  
    """Performance benchmark tests."""  
      
    def test_acceptance_rate_calculation(self):  
        """Test acceptance rate calculation."""  
        monitor = SimpleSpecPerformanceMonitor()  
          
        # Record some speculation rounds  
        monitor.record_speculation_round(4, 3, 0.01, 0.02)  # 75% acceptance  
        monitor.record_speculation_round(4, 2, 0.01, 0.02)  # 50% acceptance  
        monitor.record_speculation_round(4, 4, 0.01, 0.02)  # 100% acceptance  
          
        # Should average to 75% acceptance rate  
        self.assertAlmostEqual(monitor.get_acceptance_rate(), 0.75, places=2)  
      
    def test_speedup_calculation(self):  
        """Test speedup ratio calculation."""  
        monitor = SimpleSpecPerformanceMonitor()  
          
        # Record rounds with good acceptance  
        monitor.record_speculation_round(4, 3, 0.01, 0.02)  
        monitor.record_speculation_round(4, 3, 0.01, 0.02 