import re

class ChatTemplateMixin:
    def apply_chat_template(self, messages):
        assert self.chat_template is not None
        return tokenize_with_chat_template(self, messages, self.chat_template)

    @property
    def has_chat_template(self):
        return self.chat_template is not None


def render_chat_turn(message, template):
    """ Renders a chat turn based on template

    Args:
        message (Dict)
        e.g. {'role': ['user'], 'content': ['What is your favourite fruit?']},
        template (Dict): {
            'roles': {
                'user': "[INST] {_content_} [/INST]",
                'assistant': "{_content_}{_eos_}"
            }
        }

    Returns:
        (str, token_id/None): the template formatted message
        e.g.
            "[INST] What is your favourite fruit? [/INST]", None
    """
    assert message['role'] in template['roles'], (message['role'], template['roles'])
    template = template['roles'][message['role']]
    content = message['content']
    # Parse format and split into meta-vars vs regular strings.
    template_parts = re.split('({_[^}]+_})', template)
    is_meta_var = lambda x: x.startswith('{_') and x.endswith('_}') and not '}' in x[:-2]
    ans = []
    for i, template_part in enumerate(template_parts):
        # skip empty parts
        if template_part == '':
            continue
        if is_meta_var(template_part):
            # Strip the "{_" prefix and the "_}" suffix
            template_part = template_part[2:-2]
            if template_part == 'content':
                ans.append(message['content'])
            else:
                assert i == len(template_parts) - 1, "unsupported"
                return ''.join(ans), template_part
        else:
            ans.append(template_part)

        return ''.join(ans), None


def tokenize_with_chat_template(tokenizer, messages, template):
    assert is_chat_input(messages), "Expected input to be chat-template"
    assert len(messages) > 0, "Expected non-empty messages"
    assert 'roles' in template, "Expected template to have key `roles`."
    ans = []
    buffer = []
    render_chat_part = lambda x: render_chat_turn(x, template)
    for templated_messages, special_token in map(render_chat_part, messages):
        buffer += [templated_messages]
        if special_token is not None:
            ans.append(tokenizer.text_to_ids(''.join(buffer)) + [special_token])
            buffer = []
    if len(buffer) > 0:
        ans.append(tokenizer.text_to_ids(''.join(buffer)))
    ans = sum(ans, [])
    assert len(ans) > 0, 'Expected non-empty output'
    return ans


def extract_turns(messages, axis):
    """
    a collated messages can have multiple chat messages in each dict,
    this extracts (vertically) one of them, for example:

    messages = [
        {'role': ['user', 'user'], 'content': ['What is your favourite condiment?', 'What is your favourite fruit?']},
        {'role': ['assistant', 'assistant'], 'content': ["Well, I'm quite partial to a ", "good squeeze of fresh lemon"]},
        {'role': ['user', 'user'], 'content': ['Do you have mayonnaise recipes?', 'Do you have tomato salad recipes?']}
    ]
    ans = extract_turns(messages, axis=1)

    ans = [
        {'role': ['user'], 'content': ['What is your favourite fruit?']},
        {'role': ['assistant'], 'content': ["good squeeze of fresh lemon"]},
        {'role': ['user'], 'content': ['Do you have tomato salad recipes?']}
    ]
    """
    ans = []
    for turn in messages:
        ans.append({k: v[axis] for k,v in turn.items()})
    return ans

def explode_chat_template_input(messages):
    # [
    #   {'role': ['user', 'user'], 'content': ['What is your favourite condiment?', 'What is your favourite fruit?']},
    #   {'role': ['assistant', 'assistant'], 'content': ["Well, I'm quite partial to a ", "good squeeze of fresh lemon"]},
    #   {'role': ['user', 'user'], 'content': ['Do you have mayonnaise recipes?', 'Do you have tomato salad recipes?']}
    # ]
    assert isinstance(messages, list), "Expected messages to be a list"
    assert len(messages) > 0, "Expected non empty messages"
    assert all(map(lambda x: isinstance(x, dict), messages)), "Expected messages to contain dicts"
    assert all(map(lambda x: 'role' in x and 'content' in x, messages)), "Expected messages each dict to contain 'role' and 'content' fields"
    n = len(messages[0]['role'])
    assert all(map(lambda x: len(x['role']) == n, messages)), "Expected all batch messages to contain equal number of roles in all turns"
    for i in range(n):
        yield extract_turns(messages, axis=i)


def is_chat_input(messages):
    return isinstance(messages, list) and len(messages) > 0 and isinstance(messages[0], dict)

# /mnt/4tb/chat_template