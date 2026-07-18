from functools import lru_cache

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from rag.config import Settings


@lru_cache(maxsize=1)
def get_generation_model(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    return tokenizer, model


class LLMCallRouter:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def generate(self, prompt: str) -> tuple[str, str]:
        tokenizer, model = get_generation_model(self.settings.local_model_id)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=self.settings.local_llm_max_new_tokens,
            temperature=self.settings.local_llm_temperature,
            do_sample=self.settings.local_llm_temperature > 0,
        )
        answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return answer, "local-huggingface"
