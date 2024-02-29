import torch
import torch.distributed as dist
from torch.profiler import record_function

from lightllm.common.basemodel.splitfuse_infer_struct import SplitFuseInferStateInfo
from lightllm.models.llama.layer_weights.pre_and_post_layer_weight import LlamaPreAndPostLayerWeight
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.common.basemodel import PreLayerInferTpl
from lightllm.utils.infer_utils import mark_cost_time


class LlamaPreLayerInfer(PreLayerInferTpl):
    """
    """

    def __init__(self, tp_rank, world_size, network_config, mode):
        super().__init__(tp_rank, world_size, network_config, mode)
        tp_vocab_size_ = network_config["vocab_size"] // self.world_size_
        self.vob_start_id_ = tp_vocab_size_ * self.tp_rank_
        self.vob_end_id_ = tp_vocab_size_ * (self.tp_rank_ + 1)

        self.opt_pre_context_forward = torch.compile(self.pre_context_forward, backend='ascendgraph', dynamic=False)
        self.opt_pre_token_forward = torch.compile(self.pre_token_forward, backend='ascendgraph', dynamic=False)
        return

    def pre_context_forward(self, input_ids, layer_weight):
        bool1 = input_ids.lt(self.vob_start_id_)
        bool2 = input_ids.ge(self.vob_end_id_)
        input_mask = torch.logical_or(bool1, bool2)
        tmp_input_ids = (input_ids - self.vob_start_id_)
        tmp_input_ids[input_mask] = 0

        input_embdings = torch.embedding(layer_weight, tmp_input_ids, padding_idx=-1)
        input_embdings[input_mask] = 0.0
        if self.world_size_ > 1:
            dist.all_reduce(input_embdings, op=dist.ReduceOp.SUM, async_op=False)
        return input_embdings

    def pre_token_forward(self, input_ids, layer_weight):
        bool1 = input_ids.lt(self.vob_start_id_)
        bool2 = input_ids.ge(self.vob_end_id_)
        input_mask = torch.logical_or(bool1, bool2)
        tmp_input_ids = (input_ids - self.vob_start_id_)
        tmp_input_ids[input_mask] = 0
        input_embdings = torch.embedding(layer_weight, tmp_input_ids, padding_idx=-1)

        input_embdings[input_mask] = 0.0
        if self.world_size_ > 1:
            dist.all_reduce(input_embdings, op=dist.ReduceOp.SUM, async_op=False)
        return input_embdings

    @record_function('pre_context_forward')
    def context_forward(self, input_ids, infer_state: LlamaInferStateInfo, layer_weight: LlamaPreAndPostLayerWeight):
        input_embdings = self.opt_pre_context_forward(input_ids, layer_weight.wte_weight_)
        return input_embdings

    @record_function('pre_token_forward')
    def token_forward(self, input_ids, infer_state: LlamaInferStateInfo, layer_weight: LlamaPreAndPostLayerWeight):
        input_embdings = self.opt_pre_token_forward(input_ids, layer_weight.wte_weight_)
        return input_embdings
    
    @mark_cost_time("splitfuse forward")
    def splitfuse_forward(self, input_ids, infer_state: SplitFuseInferStateInfo, layer_weight: LlamaPreAndPostLayerWeight):
        return self.token_forward(input_ids, infer_state, layer_weight)
    
