import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Tuple
from tqdm import tqdm


# Marker for degenerate/gibberish responses
DEGENERATE_MARKER = "[DEGENERATE]"
NO_FACTS_MARKER = "NO_FACTS_FOUND"


def is_degenerate_response(response: str) -> bool:
    """
    Detect if a response is degenerate (gibberish/collapsed).

    This pre-filter catches obvious gibberish before sending to the LLM,
    saving compute and preventing hallucinated extractions.

    Args:
        response: The model response to check

    Returns:
        True if response is degenerate and should be skipped
    """
    if not response or len(response.strip()) == 0:
        return True

    words = response.lower().split()

    # Check 1: Very low unique word ratio (e.g., "the the the the")
    if len(words) > 5:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.3:
            return True

    # Check 2: Repetitive word patterns (same word 5+ times in a row)
    if re.search(r'\b(\w+)(\s+\1){4,}', response.lower()):
        return True

    # Check 3: Very short repeated character patterns
    if re.search(r'(.{1,3})\1{5,}', response):
        return True

    return False


class KnowledgeExtractor:
    """
    Extract factual knowledge claims from responses using an auxiliary LLM.

    From MD-UQ paper Section 2.2.2
    Given question Q and response A_i, generate knowledge representation K_i
    by extracting explicit claims that are standalone and independent of wording.
    """

    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-hf", device : str = "cuda"):
        """
        Args:
            model_name : Auxiliary LLM for knowledge extraction
            device: Device to run on
        """
        print(f"Loading knowledge extraction model: {model_name} on {device}...")

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        self.model.eval()

        # Set pad token if not already set (needed for batching)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        print("Knowledge extraction model loaded.")

    def extract_knowledge(self, question: str, response: str, skip_prefilter: bool = False) -> str:
        """
        Extract factual claims from a response.

        Uses a strict prompt that prevents hallucination when the response
        is incoherent or contains no factual information.

        Args:
            question: Original question
            response: Model's response to extract knowledge from
            skip_prefilter: If True, skip the degenerate response check

        Returns:
            extracted_knowledge: Factual claims as text, or DEGENERATE_MARKER/NO_FACTS_MARKER
        """
        # Layer 1: Pre-filter for degenerate responses
        if not skip_prefilter and is_degenerate_response(response):
            return DEGENERATE_MARKER

        # Layer 2: Strict prompt that prevents hallucination
        # Key changes from original:
        # - Explicit instruction to NOT use own knowledge
        # - Explicit instruction to return NO_FACTS_FOUND if incoherent
        prompt = f"""Extract factual claims from the response below that answer the question.
- ONLY extract facts explicitly stated in the response
- Do NOT use your own knowledge to answer the question
- Do NOT add information not present in the response
- If the response is incoherent or contains no relevant facts, respond with exactly: NO_FACTS_FOUND

Question: {question}
Response: {response}

Extracted claims (or NO_FACTS_FOUND):"""

        # Tokenize WITH attention mask
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(self.device)

        # Generate knowledge
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=150,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode (skip the prompt)
        input_length = inputs.input_ids.shape[1]
        extracted = self.tokenizer.decode(
            outputs[0][input_length:],
            skip_special_tokens=True
        ).strip()

        return extracted
    
    def extract_knowledge_batch(
            self,
            question: str,
            responses: List[str],
            return_stats: bool = False
    ):
        """
        Extract knowledge from multiple responses.

        Args:
            question: Original question
            responses: List of responses to extract knowledge from
            return_stats: If True, also return stats about degenerate/no_facts counts

        Returns:
            knowledge_list: List of extracted knowledge strings
            stats (if return_stats=True): Dict with counts of degenerate and no_facts responses
        """
        knowledge_list = []
        degenerate_count = 0
        no_facts_count = 0

        for response in tqdm(responses, desc="Extracting knowledge"):
            knowledge = self.extract_knowledge(question, response)
            knowledge_list.append(knowledge)

            # Track stats
            if knowledge == DEGENERATE_MARKER:
                degenerate_count += 1
            elif NO_FACTS_MARKER in knowledge.upper():
                no_facts_count += 1

        if return_stats:
            stats = {
                'degenerate_count': degenerate_count,
                'no_facts_count': no_facts_count,
                'valid_count': len(responses) - degenerate_count - no_facts_count,
                'total_count': len(responses)
            }
            return knowledge_list, stats

        return knowledge_list
    
def extract_knowledge_for_dataset(
        results: List[dict],
        extractor: KnowledgeExtractor = None,
        model_name : str = "meta-llama/Llama-2-7b-hf",
        device : str = "cuda"
) -> List[dict]:
    """
    Extract knowledge for all samples in a dataset.

    Args:
        results: List of dicts from generate_for_dataset()
        extractor: Pre-loaded knowledge extractor (optional)
        model_name : Model to use for extraction
        device: Device to run on.
    
    Returns: 
        results_with_knowledge: Same list with 'knowledge_responses' added 
    """
    # Initialize extractor once for all samples
    if extractor is None:
        extractor = KnowledgeExtractor(model_name=model_name, device=device)

    # Process each sample
    for result in tqdm(results, desc="Extracting knowledge for dataset"):

        # Extract knowledge form all responses for this sample
        knowledge_responses = extractor.extract_knowledge_batch(
            question=result['question'],
            responses=result['responses']
        )

        # Add to result dict
        result['knowledge_responses'] = knowledge_responses

    return results

