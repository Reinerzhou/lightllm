export DIPU_MOCK_CUDA=True

export DIPU_KEEP_TORCHOP_DEFAULT_IMPL_OPS="rsqrt.out,mm,linear,_softmax.out"

export DICP_TOPS_DIPU=True

python test/model/test_llama2.py
