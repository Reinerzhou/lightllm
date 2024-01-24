import os
import sys
import unittest

import torch
import torch.fx.graph_module
import torch._dynamo
import torch_dipu

torch._dynamo.config.suppress_errors = False
torch._dynamo.config.cache_size_limit = 3000

from model_infer import test_model_inference
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# class TestLlama2Infer(unittest.TestCase):
def test_llama2_infer():
    from lightllm.models.llama.model import LlamaTpPartModel
    model_dir = "/data/share_data/llama_model_data/llama-2-7b-chat-hf"
    test_model_inference(world_size=1,
                         model_dir=model_dir,
                         model_class=LlamaTpPartModel,
                         batch_size=1,
                         input_len=8,
                         output_len=6,
                         mode=[])
    return


if __name__ == '__main__':
    # unittest.main()
    from dicp.tools.op_collector import InnerCompilerOpCollectorContext
    with InnerCompilerOpCollectorContext(
        inner_commpiler_func="dicp.dynamo_bridge.compile_fx.compile_fx_inner",
        compile_fx_func="dicp.dynamo_bridge.compile_fx.compile_fx",
        collector_name="llama2",
        inner_compiler_param_key="inner_compile",
        write_file=True,
        bypass_graph_module=False,
        cache_graph_module=True,
    ) as ctx:
        test_llama2_infer()
