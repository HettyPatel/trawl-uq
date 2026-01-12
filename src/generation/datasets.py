from datasets import load_dataset
from typing import Dict, List

class QADataset:
    """
    Base class for question-answering datasets.
    This defines the common interface that all QA datasets should follow.
    """

    def __init__(self, dataset_name: str, split: str = "validation", num_samples: int = None):
        """

        Args:
            dataset_name: Name of the dataset (e.g 'coqa')
            split: Which split to use ('train', 'validation', 'test)
            num_samples: How many samples to load (None = Load all)
        """
        self.dataset_name = dataset_name
        self.split = split
        self.num_samples = num_samples
        self.data = None # will store processed data after load()

    def load(self):
        """
        Load and process the dataset.
        Each subclass implements this differently based on dataset structure.
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    def __len__(self):
        """
        Return number of samples in dataset
        """
        return len(self.data) if self.data else 0
    
    def __getitem__(self, idx):
        """Get single sample by index (allows dataset[0], dataset[1] etc)"""
        return self.data[idx]
    


class CoQADataset(QADataset):
    """
    CoQA (Conversational Question Answering) dataset.
    Structure: story (context) + questions + answers.
    Example: 
        Story about students becming heros,
        Q: How many?
        A: Three.
    """

    def load(self, tokenizer):
        """
        Load and process CoQA dataset from HuggingFace.
        
        Returns: 
            self: For method chaining (dataset.load(tokenizer).data)
        """

        # Download CoQA from HuggingFace (cahched after first downlaod)
        dataset = load_dataset("coqa", split=self.split)

        # Limit number of samples if specified (e.g., for quick testing)
        if self.num_samples:
            dataset = dataset.select(range(min(self.num_samples, len(dataset))))

        # Process each item into a standard format
        processed_data = []

        for idx, item in enumerate(dataset):
            # Extract fields from CoQA format
            story = item['story'] # background context paragraph
            questions = item['questions'][0] #First question in the conversation
            answer = item['answers']['input_text'][0] #First answer in the conversation

            # Format as prompt for the LLM
            # This is what we'll feed to Roberta / GPT-J etc. 

            prompt = f"Context: {story}\nQuestion: {questions}\nAnswer:"

            # Store in standard format (same across all datasets for easier downstream processing)
            processed_data.append({
                'id' : f"coqa_{idx}", # Unique identifier
                'prompt': prompt, # Full prompt for the model
                'question' : questions, # Question Only
                'answer' : answer, # Gold answer (for evaluation)
                'context' : story, # Original context
            })

        self.data = processed_data
        return self # return self for chaining.
        
class HotpotQADataset(QADataset):
    """
    HotpotQA dataset - requires multi-hop reasoning across mulitple paragraphs.
    Example: ' What nationality is the director of the 2018 film Arctic?
    Needs to find: (1) Director of Arctic, (2) Nationality of that director
    """

    def load(self, tokenizer):
        """Load and process HotpotQA dataset."""
        # 'distractor version includes irrelevant paragraphs to increase difficulty
        dataset = load_dataset("hotpot_qa", "distractor", split=self.split)

        if self.num_samples:
            dataset = dataset.select(range(min(self.num_samples, len(dataset))))

            processed_data = []

        for idx, item in enumerate(dataset):

            question = item['question']
            answer = item['answer']

            # HotpotQA has multiple context paragraphs from different wikipedia articles
            # Each has a title and list of sentences
            # We combine them into one context (taking first 4 sentences to avoid too long inputs)

            context_sentences = []
            for title, sentences in zip(item['context']['title'], item['context']['sentences']):
                context_sentences.extend(sentences)
            context = " ".join(context_sentences[:5]) # take first 5 sentences

            # Format prompt
            prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"

            processed_data.append({
                'id': f"hotpotqa_{idx}",
                'prompt': prompt,
                'question': question,
                'answer': answer,
                'context': context
            })

        self.data = processed_data
        return self # return self for chaining.


class NQOpenDataset(QADataset):
    """
    Natural Questions Open dataset - open domain QA from Google Search queries.
    Questions are real user queries, answers are short (<=5 tokens).
    No context provided - tests model's parametric knowledge.
    """

    def load(self, tokenizer):
        """Load and process NQ-Open dataset from HuggingFace."""
        dataset = load_dataset("google-research-datasets/nq_open", split=self.split)

        if self.num_samples:
            dataset = dataset.select(range(min(self.num_samples, len(dataset))))

        processed_data = []

        for idx, item in enumerate(dataset):
            question = item['question']
            # NQ-Open can have multiple valid answers
            answers = item['answer']  # List of valid answers
            answer = answers[0] if answers else ""  # Use first answer as primary

            # No context for open-domain QA - just the question
            prompt = f"Question: {question}\nAnswer:"

            processed_data.append({
                'id': f"nq_open_{idx}",
                'prompt': prompt,
                'question': question,
                'answer': answer,
                'all_answers': answers,  # Store all valid answers for evaluation
                'context': None  # No context for open-domain QA
            })

        self.data = processed_data
        return self


####======================================================####

def get_dataset(dataset_name: str, split: str = "validation", num_samples: int = None):
    """
    Factory function: returns the appropriate dataset class.
    Main function used to load datasets.
    
    Usage:
        # Load 100 samples from CoQA validation set
        dataset = get_dataset("coqa", split="validation", num_samples=100)
        dataset.load(tokenizer)

        #Access Data:
        print(len(dataset)) # 100
        print(dataset[0])  # First Sample

    Args:
        dataset_name: 'coqa' or 'hotpotqa'
        split: 'train', 'validation', 'test'
        num_samples: How many samples to load (None = all)
    
    Returns:
        Dataset instance (not yet loaded - most call .load(tokenizer) on it first)

    """

    # Map dataset names to classes
    datasets = {
        "coqa": CoQADataset,
        "hotpotqa": HotpotQADataset,
        "nq_open": NQOpenDataset
    }

    # check if requested dataset is supported
    if dataset_name.lower() not in datasets:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: {list(datasets.keys())}")
    
    # Create and reutn instance of the appropriate class
    return datasets[dataset_name](dataset_name, split, num_samples)


