import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
from tqdm import tqdm

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

    def extract_knowledge(self, question: str, response: str) -> str:
        """
        Extract factual claims from a response.
        
        Uses the prompt from MD-UQ paper Section 2.2.2:
        "Extract all factual claims from this response, phrased as standalone
        statements independent of specific wording. Include only information
        directly relevant to answering the question."
        
        Args:
            question: Original question
            response: Model's response to extract knowledge from
        
        Returns:
            extracted_knowledge: Factual claims as text
        """
        # Build prompt for knowledge extraction
        prompt = f"""Extract all factual claims from the following response. Present them as standalone statements that are independent of specific wording. Only include information directly relevant to answering the question.

    Question: {question}
    Response: {response}

    Extracted factual claims:"""
        
        # Tokenize WITH attention mask
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        # Generate knowledge
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,  # ← Add this!
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
            responses : List[str]
    ) -> List[str]:
        """
        Extract knowlede from multiple responses.

        Args:
            question : Original question
            responses : List of responses to extract knowledge from

        Returns:
            Knowledge_list : List of extracted knowledge strings
        """

        knowledge_list = []
        for response in tqdm(responses, desc="Extracting knowledge"):
            knowledge = self.extract_knowledge(question, response)
            knowledge_list.append(knowledge)

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

