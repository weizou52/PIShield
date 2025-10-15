def clean_text(text, tokenizer):
    """
    Clean the input text by replacing special tokens.

    Args:
        text: The input text to be cleaned.

    Returns:
        The cleaned text with special tokens replaced.
    """
    if not text:
        return text

    def insert_vline(token: str) -> str:
        if len(token) < 2:
            return " "
        elif len(token) == 2:
            return f"{token[0]}|{token[1]}"
        else:
            return f"{token[:1]}|{token[1:-1]}|{token[-1:]}"

    if tokenizer.bos_token:
        text = text.replace(tokenizer.bos_token, insert_vline(tokenizer.bos_token))
    if tokenizer.eos_token:
        text = text.replace(tokenizer.eos_token, insert_vline(tokenizer.eos_token))
    if tokenizer.pad_token:
        text = text.replace(tokenizer.pad_token, insert_vline(tokenizer.pad_token))
    if tokenizer.unk_token:
        text = text.replace(tokenizer.unk_token, insert_vline(tokenizer.unk_token))

    return text


def get_formatted_data1(examples, tokenizer, use_chat_template=True, use_system_prompt=True):
    # print(f"Processing {len(examples)} examples")
    formatted_dataset = []
    for example in examples:
        if use_chat_template:
            if use_system_prompt:   
                message = [{"role": "system", "content": f"{example['instruction']}"}, {"role": "user", "content": f"{clean_text(example['data_prompt'], tokenizer)}"}]
            else:
                message = [{"role": "user", "content": f"{clean_text(example['data_prompt'], tokenizer)}"}]
            formated_data = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        else:
            formated_data = clean_text(f"{example['instruction']}\n{example['data_prompt']}", tokenizer)
        
        formatted_dataset.append(formated_data)

    # print(f"Processed {len(formatted_dataset)} examples")
    return formatted_dataset


def get_formatted_data2(examples, tokenizer, use_chat_template=True, use_system_prompt=True):
    print(f"Processing {len(examples)} examples")
    formatted_dataset = []
    for example in examples:
        if use_chat_template:
            if use_system_prompt:       
                message = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": clean_text(example['data_prompt'], tokenizer)}]
            else:
                message = [{"role": "user", "content": clean_text(example['data_prompt'], tokenizer)}]
            formated_data = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        else:
            formated_data = clean_text(example['data_prompt'], tokenizer)
        
        formatted_dataset.append(formated_data)

    print(f"Processed {len(formatted_dataset)} examples")
    return formatted_dataset