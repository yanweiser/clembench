from typing import Dict

# Define chat templates here for multimodal models
# Cannot use tokenizers to apply chat template, apply them manually for now


def apply_ua_template(current_messages: Dict) -> str:
    '''
    Return the chat template in the following format
    # prompt template
    # USER:
    # <image>\n<prompt> 
    # ASSISTANT:
    # <answer1>
    # USER:
    # <response1>
    # ASSISTANT:
    # ...

    Args: 
        current_messages - The base message passed to the backend
    Returns:
        prompt_text - The message in required template
    '''

    if current_messages[0]['role'] == 'system':
        del current_messages[0]
    prompt_text = ""

    assert current_messages, "messages cannot only contain a system prompt"
    assert current_messages[0]['role'] == 'user', "You need to start dialogue on a User entry"
    
    current_messages[-1]['content'] = f"<image>\n{current_messages[-1]['content']}"
    prompt_text = ""

    for msg in current_messages:
        if msg['role'] == 'user':
            prompt_text += f"USER:  {msg['content']}\n"
        else:
            prompt_text += f"ASSISTANT:  {msg['content']}\n"
    prompt_text += "ASSISTANT:  "
    
    return prompt_text

def apply_qa_template(current_messages: Dict) -> str:
    '''
    No chat template for models like BLIP2, applying VQA template 
    template - "Question : question_str. Answer: "

    Args: 
        current_messages - The base message passed to the backend
    Returns:
        prompt_text - The message in required template
    '''

    if current_messages[0]['role'] == 'system':
        del current_messages[0]
    prompt_text = ""

    assert current_messages, "messages cannot only contain a system prompt"
    assert current_messages[0]['role'] == 'user', "You need to start dialogue on a User entry"

    prompt_text += f"Question: {current_messages[0]['content']}"

    for msg in current_messages[1:]:
        if msg['role'] == 'user':
            prompt_text = f"Question:  {msg['content']}\n"
        else:
            prompt_text = f"Answer:  {msg['content']}\n"
    prompt_text += "Answer:  "
    
    return prompt_text