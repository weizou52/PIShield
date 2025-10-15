from .utils import *
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from pishield.utils import *

class Extractor:
    def __init__(self, model_name, format_id, token_position, batch_size=1, device='auto'):
        self.model_name = model_name
        self.format = format_id
        self.token_position = token_position
        self.batch_size = batch_size
        self.device = device 
        self.name = f"{self.model_name}_{self.format}_{self.token_position}"
        self.model = None
        self.model_config = self._load_model_config()
        self.layer_ids = list(range(self.model_config[self.model_name]['num_hidden_layers']))

    def _load_model_config(self):
        return jload('pishield/config')['models']

    def _load_model(self):
        model_path = self.model_config[self.model_name]['path']
        self.use_system_prompt = self.model_config[self.model_name]['use_system_prompt']
        self.use_chat_template = self.model_config[self.model_name]['use_chat_template']
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map=self.device, torch_dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id


    def get_formatted_data(self, examples):
        if self.format==1:
            return get_formatted_data1(examples, self.tokenizer, self.use_chat_template, self.use_system_prompt)
        elif self.format==2:
            return get_formatted_data2(examples, self.tokenizer, self.use_chat_template, self.use_system_prompt)
        else:
            raise ValueError(f"Invalid format: {self.format}")


    def extract_hidden_states(self, inputs):
        """
        Retrieve the hidden states from the specified layers of the language model for the given input strings.

        Args:
            model: The model to get hidden states from.
            tokenizer: The tokenizer associated with the model.
            inputs: A list of input strings.
            layer_ids: The IDs of hidden layers to get states from.
            batch_size: The batch size to use when processing inputs.
            token_position: How many tokens to accumulate for the hidden state.
            suffixes: List of suffixes to use when accumulating hidden states.

        Returns:
            A dictionary mapping layer IDs to numpy arrays of hidden states.
        """
        print(f"Extracting hidden states for {len(inputs)} inputs...")
        # Split the input strings into batches based on the specified batch size
        batched_inputs = [
            inputs[p : p + self.batch_size] for p in range(0, len(inputs), self.batch_size)
        ]  
        # Initialize an empty dictionary to store the hidden states for each specified layer
        hidden_states = {layer: [] for layer in self.layer_ids}
        
        # Disable gradient computation for efficiency
        with torch.no_grad():
            # Iterate over each batch of input strings
            for b_idx, batch in enumerate(batched_inputs):
                print(f"batch {b_idx}/{len(batched_inputs)}")
                # Get tokenized input and attention mask
                special_token_flag = not self.use_chat_template
                tokenized = self.tokenizer(batch, padding=True, add_special_tokens=special_token_flag, return_tensors="pt").to(self.model.device)
                attention_mask = tokenized.attention_mask
                # Pass the batch through the language model and retrieve the hidden states
                out = self.model(
                    **tokenized,
                    output_hidden_states=True,
                )
                
                # Iterate over each specified layer ID
                for layer_id in self.layer_ids:
                    # Adjust the layer index if it is negative
                    hidden_idx = layer_id + 1 if layer_id >= 0 else layer_id
                    
                    # Iterate over each batch of hidden states
                    for i, batch_hidden in enumerate(out.hidden_states[hidden_idx]):
                        # Get the actual sequence length (excluding padding)
                        # if b_idx==0 and layer_id==0 and i==0:
                            # print(batch[i])
                        seq_len = attention_mask[i].sum().item()
                        if self.token_position == "all":
                            accumulated_hidden_state = torch.mean(batch_hidden, dim=0)
                        elif self.token_position == "last":
                            accumulated_hidden_state = batch_hidden[seq_len-1, :]
                        else:
                            token_position=int(self.token_position)
                            accumulated_hidden_state = torch.mean(batch_hidden[max(0,seq_len-token_position):seq_len, :], dim=0)
                        
                        hidden_states[layer_id].append(accumulated_hidden_state.squeeze().cpu().numpy())
                
                # Delete the model output to free up memory
                del out
        
        # Stack the hidden states for each layer into a numpy array
        # Return the dictionary mapping layer IDs to their corresponding stacked hidden states
        print(f"Extraction finished.\n\n")
        return {k: np.vstack(v) for k, v in hidden_states.items()}

    def __call__(self, examples) -> dict[int, np.ndarray]:
        if self.model is None:
            self._load_model()
        return self.extract_hidden_states(self.get_formatted_data(examples))
    
