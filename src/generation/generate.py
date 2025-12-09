import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict
import numpy as np
from tqdm import tqdm


def load_model_and_tokenizer(model_name: str, device: str = "cuda", dtype=torch.float16):
    """
    Load model and toknizer from HuggingFace.
    Supported Models:
       - GPT-2: "gpt2"
        - GPT-J: "EleutherAI/gpt-j-6b"
        - Llama-2: "meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-13b-hf"
        - Llama-3: "meta-llama/Llama-3.1-8B"
        - Phi: "microsoft/phi-2

    Args:
        model_name: HuggingFace model name (e.g., "roberta-base", "gpt2")
        device: Device to load model on (Cuda or CPU)
        dtype: Data type (torch.float 16 for memory efficiency) float32 for full precision

    Returns:
        model, tokenizer: Loaded model and tokenizer
    """
    print (f"Loading model and tokenizer for {model_name} on {device}...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load casual language model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype, # Use half precision for memory efficiency
        device_map = "auto", # Automatically map model to available devices
        low_cpu_mem_usage=True # Reduce CPU memory usage during load
    )

    model.eval() # set model to evaluation mode (disables droput, grads etc)
    print ("Model and tokenizer loaded.")

    # Set pad token if not already set (needed for batching)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # for left padding
    tokenizer.padding_side = "left"

    print(f" Model loaded successfully")
    print(f"  - Device: {model.device}")
    print(f"  - dtype: {model.dtype}")
    print(f"  - Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    
    return model, tokenizer

def generate_responses(
        model,
        tokenizer,
        prompt: str,
        num_generations: int = 20,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        do_sample: bool = True,
) -> List[str]:
    """
    Generate multiple responses for a single prompt. 
    
    This is the main function that creates the N responses we need for uncertainty quantification (N=20)

    Args:
        model : The Causal language model
        tokenizer : Corresponding tokenizer
        prompt : Input prompt/question
        num_generations : how many responses to generate (default 20)
        max_new_tokens : Maximum length of generated text
        temperature : Sampling temperature (higher = more diverse) 
                      1.0 = normal, 0.7 = focused, 1.5 = more diverse
        top_p : Nucleus sampling threshold (0.9 = keep top 90% probability mass.
        top_k : Top-K sampling - keep only top k tokens (0 = disabled)
        do_sample : If True, sample randomly. If False, use greedy decoding. 

    Returns:
        responses: List of generated text strings

    """

    device = model.device

    # Tokenize the input prompt
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
    ).to(device)
    
    input_length = inputs.input_ids.shape[1] # track where the prompt ends

    responses = []

    # generate responses in batches to save time.
    batch_size = min(4, num_generations) # generate up to 4 at a time
    num_batches = (num_generations + batch_size - 1) // batch_size

    with torch.no_grad(): # no gradients for inference
        for batch_idx in range(num_batches):

            # How many to generate in this batch
            current_batch_size = min(batch_size, num_generations - len(responses)) # either generate full batch or remaining

            # Generate outputs
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k if top_k > 0 else None,
                num_return_sequences=current_batch_size, # generate multiple sequences per batch
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            # Decode the generated tokens (skip the input prompt part) - stored location above
            for output in outputs:
                generated_text = tokenizer.decode(
                    output[input_length:], # skip the prompt part
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                responses.append(generated_text.strip())

    return responses[:num_generations] # Return exactly num_generations reposne


def generate_for_dataset(
        model,
        tokenizer,
        dataset,
        num_generations: int = 20,
        save_every : int = 10,
        output_file : str = None,
        **generation_kwargs
) -> List[Dict]:
    """
    Generate responses for an entire datset.
    
    This function processes all samples in a dataset and generates multiple responses for each one. 

    Args:
        model : LM
        tokenizer : Tokenizer
        dataset : Dataset object (from datasets.py)
        num_generations : Number of responses per question
        save_every : Save intermediate results every N samples (default 10)
        output_file : Path to save results  (optional) 
        **generation_kwargs : Additional arguments passed to generate_responses()
        (temperature, top_p, max_new_tokens etc)

    Returns:
       all_results: List of dicts, each containing:
            'id' : sample identifier
            'prompt' : input prompt
            'question' : original question
            'answer' : gold answer
            'generations' : List of generated responses 

    """

    all_results = []

    # Process each sample with a progress bar
    for idx, sample in enumerate(tqdm(dataset, desc="Generating responses")):

        # Generate multiple responses for this sample
        responses = generate_responses(
            model = model,
            tokenizer = tokenizer,
            prompt = sample['prompt'],
            num_generations = num_generations,
            **generation_kwargs
        )

        # Store results
        result = {
            'id' : sample['id'],
            'prompt' : sample['prompt'],
            'question' : sample['question'],
            'answer' : sample['answer'],
            'context' : sample.get('context', None), # optional
            'responses' : responses
        }

        all_results.append(result)

        # Save intermediate results
        if output_file and (idx + 1) % save_every == 0:
            import pickle
            with open(output_file, 'wb') as f:
                pickle.dump(all_results, f)
            print(f"\nSaved intermediate results ({len(all_results)} samples)")

    # Final save
    if output_file:
        import pickle
        with open(output_file, 'wb') as f:
            pickle.dump(all_results, f)
        print(f"\nSaved final results ({len(all_results)} samples) to {output_file}")
    
    
    return all_results


def seed_everything(seed: int = 42):
    """
    Set random seed for reproducibility.

    Args:
        seed: Seed value
    """
    
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    
    # to get reproducible results on CUDA (may be slower)
    
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False