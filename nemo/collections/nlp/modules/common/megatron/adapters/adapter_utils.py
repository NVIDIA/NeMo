import enum


class AdapterName(str, enum.Enum):
    """
    Names for adapters used in NLP Adapters and IA3. Note: changing this will break backward compatibility. 
    """

    MLP_INFUSED = "mlp_infused_adapter"
    KEY_INFUSED = "key_infused_adapter"
    VALUE_INFUSED = "value_infused_adapter"
    PRE_ATTN_ADAPTER = 'adapter_1'
    POST_ATTN_ADAPTER = 'adapter_2'
    TINY_ATTN_ADAPTER = "tiny_attn_adapter"
