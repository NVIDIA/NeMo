from dataclasses import dataclass


@dataclass
class TPOverlapCfg:
    pass


@dataclass
class PipelineOverlapCfg(TPOverlapCfg):
    num_sm: int
    cga_size: int
    num_splits: int
    set_sm_margin: bool
    fp8_buf: bool = (False,)
    method: str = 'pipeline'


@dataclass
class RingExchangeOverlapCfg(TPOverlapCfg):
    aggregate: bool = False
    method: str = 'ring_exchange'


@dataclass
class BulkOverlapCfg(TPOverlapCfg):
    num_sm: int
    cga_size: int
    set_sm_margin: bool
    method: str = 'bulk'


@dataclass
class TransformerLayerTPOverlapCfg:
    qkv_dgrad: TPOverlapCfg
    qkv_wgrad: TPOverlapCfg
    fc1_dgrad: TPOverlapCfg
    fc1_wgrad: TPOverlapCfg
    qkv_fprop: TPOverlapCfg
    proj_dgrad: TPOverlapCfg
    fc1_fprop: TPOverlapCfg
    fc2_dgrad: TPOverlapCfg
    proj_fprop: TPOverlapCfg
    fc2_fprop: TPOverlapCfg


# Model configs: H100/70B/TP8/MBS1/SeqLen8K
userbuffers_bf16_h100_h8192_tp4_mbs1_seqlen8192 = TransformerLayerTPOverlapCfg(
    qkv_dgrad=BulkOverlapCfg(num_sm=4, cga_size=2, set_sm_margin=False),
    qkv_wgrad=BulkOverlapCfg(num_sm=24, cga_size=2, set_sm_margin=False),
    fc1_dgrad=BulkOverlapCfg(num_sm=2, cga_size=2, set_sm_margin=False),
    fc1_wgrad=BulkOverlapCfg(num_sm=4, cga_size=2, set_sm_margin=False),
    qkv_fprop=RingExchangeOverlapCfg(aggregate=False),
    proj_dgrad=RingExchangeOverlapCfg(aggregate=False),
    fc1_fprop=RingExchangeOverlapCfg(aggregate=False),
    fc2_dgrad=RingExchangeOverlapCfg(aggregate=False),
    proj_fprop=PipelineOverlapCfg(num_sm=24, cga_size=2, num_splits=4, set_sm_margin=True),
    fc2_fprop=PipelineOverlapCfg(num_sm=16, cga_size=2, num_splits=4, set_sm_margin=True),
)

userbuffers_fp8_h100_h8192_tp4_mbs1_seqlen8192 = TransformerLayerTPOverlapCfg(
    qkv_dgrad=BulkOverlapCfg(num_sm=4, cga_size=2, set_sm_margin=False),
    qkv_wgrad=BulkOverlapCfg(num_sm=24, cga_size=2, set_sm_margin=False),
    fc1_dgrad=BulkOverlapCfg(num_sm=2, cga_size=2, set_sm_margin=False),
    fc1_wgrad=BulkOverlapCfg(num_sm=4, cga_size=2, set_sm_margin=False),
    qkv_fprop=RingExchangeOverlapCfg(aggregate=False),
    proj_dgrad=RingExchangeOverlapCfg(aggregate=False),
    fc1_fprop=RingExchangeOverlapCfg(aggregate=False),
    fc2_dgrad=RingExchangeOverlapCfg(aggregate=False),
    proj_fprop=PipelineOverlapCfg(num_sm=24, cga_size=2, num_splits=4, set_sm_margin=True, fp8_buf=True),
    fc2_fprop=PipelineOverlapCfg(num_sm=16, cga_size=2, num_splits=4, set_sm_margin=True, fp8_buf=True),
)