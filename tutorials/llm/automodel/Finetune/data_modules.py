"""
Flexible data modules for various datasets with HuggingFace integration.
"""
import json
from typing import Any, Dict, List, Optional

from datasets import load_dataset

from nemo.collections.llm.gpt.data.hf_dataset import HFDatasetDataModule


class CustomHFDataModule(HFDatasetDataModule):
    """
    Flexible data modules for various datasets with HuggingFace integration.
    """

    def __init__(self, tokenizer, seq_length: int, path_or_dataset=None, **kwargs):
        # Handle loading JSON/JSONL files as HF datasets
        if path_or_dataset and type(path_or_dataset) == list and (path_or_dataset[0].endswith('.jsonl') or path_or_dataset[0].endswith('.json')):
            dataset = load_dataset('json', data_files={'train': path_or_dataset[0], 'validation': path_or_dataset[1]})
            kwargs['path_or_dataset'] = dataset
        elif path_or_dataset:
            kwargs['path_or_dataset'] = path_or_dataset
            
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
        self.seq_length = seq_length

    def get_chat_template(self, tokenizer):
        # attempt to unwrap NeMo's tokenizer wrapper and check if wrapped tokenizer has chat_template
        tmp_tokenizer = getattr(tokenizer, 'tokenizer', tokenizer)
        has_chat_template = getattr(tmp_tokenizer, 'chat_template', None) is not None
        if has_chat_template:
            return tmp_tokenizer, getattr(tmp_tokenizer, 'eos_token_id', None), has_chat_template
        else:
            return tokenizer, getattr(tokenizer, 'eos_id', None), has_chat_template

    def formatting_prompts_func_with_chat_template(self, example: Dict[str, Any], start_of_turn_token: Optional[str] = None) -> Dict[str, List[int]]:
        """
        Format any conversation example using Mistral chat template.
        
        Args:
            example: Dataset example, preferably with a 'messages' list and optional 'tools'.
            start_of_turn_token: Token marking start of assistant response
            
        Returns:
            Dictionary with input_ids, labels, and loss_mask
        """
        tools = example.get('tools', [])

        formatted_text: List[Dict[str, str]] = []
        raw_messages = example.get('messages')

        # Build system prompt that includes the available tools
        tools_block = '[AVAILABLE_TOOLS]' + json.dumps(tools, separators=(',', ':')) + '[/AVAILABLE_TOOLS]'
        if raw_messages[0].get('role') == 'system':
            system_content = str(raw_messages[0].get('content', ''))
            if '[AVAILABLE_TOOLS]' not in system_content:
                system_content = (system_content + '\n' + tools_block).strip()
            formatted_text.append({'role': 'system', 'content': system_content})
            start_index = 1
        else:
            system_content = 'You are a helpful assistant.'
            formatted_text.append({'role': 'system', 'content': (system_content + '\n' + tools_block)})
            start_index = 0

        # Process the rest of the turns
        for msg in raw_messages[start_index:]:
            role = msg.get('role')
            if role in ('user', 'assistant'):
                tool_calls = msg.get('tool_calls')
                if role == 'assistant' and isinstance(tool_calls, list) and len(tool_calls) > 0:
                    # Normalize tool calls -> keep id, type, function {name, arguments}
                    normalized_calls: List[Dict[str, Any]] = []
                    for call in tool_calls:
                        call_id = call.get('id')
                        call_type = call.get('type', 'function')
                        fn = call.get('function', {}) or {}
                        fn_name = fn.get('name')
                        fn_args = fn.get('arguments')
                        if isinstance(fn_args, str):
                            try:
                                fn_args = json.loads(fn_args)
                            except Exception:
                                pass
                        normalized_calls.append({
                            'id': call_id,
                            'type': call_type,
                            'function': {
                                'name': fn_name,
                                'arguments': fn_args
                            }
                        })
                    formatted_text.append({
                        'role': 'assistant',
                        'content': '[TOOL_CALLS]' + json.dumps(normalized_calls, separators=(',', ':'))
                    })
                else:
                    content = msg.get('content', '')
                    formatted_text.append({'role': role, 'content': str(content)})
            elif role == 'tool':
                tool_call_id = msg.get('tool_call_id')
                tool_content = msg.get('content', '')
                result_obj = {'tool_call_id': tool_call_id, 'content': str(tool_content)}
                formatted_text.append({
                    'role': 'assistant',
                    'content': '[TOOL_RESULTS]' + json.dumps(result_obj, separators=(',', ':')) + '[/TOOL_RESULTS]'
                })
            else:
                continue

        input_ids = self.get_chat_template(self.tokenizer)[0].apply_chat_template(formatted_text, tools=tools, add_generation_prompt=True)
        
        if isinstance(start_of_turn_token, str):
            start_of_turn_token_id = self.tokenizer(start_of_turn_token, add_special_tokens=False)['input_ids'][0]
            first_start_of_turn_token_id = input_ids.index(start_of_turn_token_id)
            response_start = input_ids.index(start_of_turn_token_id, first_start_of_turn_token_id + 1) + 1
        else:
            response_start = 0
            
        loss_mask = [0] * response_start + [1] * (len(input_ids) - response_start)
        
        return dict(
            input_ids=input_ids,
            labels=input_ids[1:] + [getattr(self.tokenizer, 'eos_token_id', None) or input_ids[-1]],
            loss_mask=loss_mask,
        )

    def setup(self, stage):
        """
        Setup the dataset with datasetspecific formatting.
        
        Args:
            stage: Training stage
        """
        super().setup(stage)
        
        # Determine which columns to remove based on dataset structure
        remove_columns = ["conversation_id", "messages", "tools"]

        self.map(
            self.formatting_prompts_func_with_chat_template,
            batched=False,
            batch_size=2,
            remove_columns=remove_columns,
        )