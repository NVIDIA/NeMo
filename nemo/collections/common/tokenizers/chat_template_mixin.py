class ChatTemplateMixin:
    def apply_chat_template(self, input):
        assert self.chat_template is not None
        return tokenize_with_chat_template(self, input, self.chat_template, False)


    @property
    def has_chat_template(self):
        return self.chat_template is not None




def render_chat_turn(template, input):
    assert input['role'] in template['roles']
    template = template['roles'][input['role']]
    content = input['content']
    prefix = template.get('prefix', '')
    suffix = template.get('suffix', '')
    decorated_prompt = f"{prefix}{content}{suffix}"
    return decorated_prompt, template.get('add_eos_suffix', False)


def tokenize_with_chat_template(tokenizer, inputs, template, add_bos):
    assert len(inputs) > 0, "Expected non-empty inputs"
    assert 'roles' in template, "Expected template to have key `roles`."
    ans = []
    tmp_buffer = []
    render_chat_part = lambda x: render_chat_turn(template, x)
    for templated_input, has_eos in map(render_chat_part, inputs):
        tmp_buffer += [templated_input]
        if has_eos:
            ans.append(tokenizer.text_to_ids(''.join(tmp_buffer)) + [tokenizer.eos_id])
            tmp_buffer = []
    if len(tmp_buffer) > 0:
        ans.append(tokenizer.text_to_ids(''.join(tmp_buffer)))
    ans = sum(ans, [])
    assert len(ans) > 0, 'Expected non-empty output'
    if add_bos:
        return [tokenizer.bos_id] + ans
    return ans


def extract_turns(sentences, axis):
    """
    a collated input can have multiple chat prompts in each dict,
    this extracts (vertically) one of them, for example:

    sentences = [
        {'role': ['user', 'user'], 'content': ['What is your favourite condiment?', 'What is your favourite fruit?']},
        {'role': ['assistant', 'assistant'], 'content': ["Well, I'm quite partial to a ", "good squeeze of fresh lemon"]},
        {'role': ['user', 'user'], 'content': ['Do you have mayonnaise recipes?', 'Do you have tomato salad recipes?']}
    ]
    ans = extract_turns(sentences, axis=1)

    ans = [
        {'role': ['user'], 'content': ['What is your favourite fruit?']},
        {'role': ['assistant'], 'content': ["good squeeze of fresh lemon"]},
        {'role': ['user'], 'content': ['Do you have tomato salad recipes?']}
    ]
    """
    ans = []
    for turn in sentences:
        ans.append({k: v[axis] for k,v in turn.items()})
    return ans

def explode_chat_template_input(sentences):
    # [
    #   {'role': ['user', 'user'], 'content': ['What is your favourite condiment?', 'What is your favourite fruit?']},
    #   {'role': ['assistant', 'assistant'], 'content': ["Well, I'm quite partial to a ", "good squeeze of fresh lemon"]},
    #   {'role': ['user', 'user'], 'content': ['Do you have mayonnaise recipes?', 'Do you have tomato salad recipes?']}
    # ]
    assert isinstance(sentences, list), "Expected input to be a list"
    assert len(sentences) > 0, "Expected non empty input"
    assert all(map(lambda x: isinstance(x, dict), sentences)), "Expected prompts to contain dicts"
    assert all(map(lambda x: 'role' in x and 'content' in x, sentences)), "Expected prompts each dict to contain 'role' and 'content' fields"
    n = len(sentences[0]['role'])
    assert all(map(lambda x: len(x['role']) == n, sentences)), "Expected all batch prompts to contain equal number of roles in all turns"
    for i in range(n):
        yield extract_turns(sentences, axis=i)


def is_chat_input(sentences):
    return isinstance(sentences, list) and len(sentences) > 0 and isinstance(sentences[0], dict)