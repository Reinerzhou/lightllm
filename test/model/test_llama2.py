import os
import sys
import unittest
from model_infer import test_model_inference
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch_dipu

class TestLlama2Infer(unittest.TestCase):

    def test_llama2_infer(self):
        from lightllm.models.llama.model import LlamaTpPartModel
        test_model_inference(world_size=8, 
                             model_dir="/path/llama2-7b-chat", 
                             model_class=LlamaTpPartModel, 
                             batch_size=20, 
                             input_len=10, 
                             output_len=20,
                             mode=[])
        return


if __name__ == '__main__':
    unittest.main()