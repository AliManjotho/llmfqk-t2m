import torch
from transformers import BertTokenizer, BertModel

class LLMAmbiguityResolver:
    """
    Extracts contextual terms using an LLM (e.g., BERT or LLaMA) to resolve linguistic ambiguities.
    """

    def __init__(self, model_name="bert-base-uncased"):
        """
        Args:
            model_name (str): Name of the LLM model to use.
        """
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

    def extract_contextual_terms(self, text_prompt):
        """
        Extracts contextual terms from ambiguous text using BERT embeddings.
        Args:
            text_prompt (str): Input text description.
        Returns:
            dict: Extracted contextual terms categorized into lexical, semantic, pragmatic, action-object, role, and temporal.
        """
        inputs = self.tokenizer(text_prompt, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        text_embedding = outputs.last_hidden_state.mean(dim=1)  # Extract sentence-level representation

        # Simulate extracting contextual terms (in real case, use embeddings to match predefined lexicons)
        contextual_terms = {
            "lexical": ["walk", "step", "move"],
            "semantic": ["slow", "moderate"],
            "pragmatic": ["indoor"],
            "action-object": ["leg-movement"],
            "role": ["human"],
            "temporal": ["continuous"]
        }
        
        return text_embedding, contextual_terms


if __name__ == "__main__":
    text_prompt = "A person is walking slowly in a large room."
    resolver = LLMAmbiguityResolver()

    embedding, context_terms = resolver.extract_contextual_terms(text_prompt)
    print("\nExtracted Contextual Terms:")
    for key, value in context_terms.items():
        print(f"{key.capitalize()}: {value}")


