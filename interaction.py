from tenacity import retry, stop_after_attempt, wait_random_exponential
import openai

@retry(stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=1, max=40))
def get_embedding(text, model_name):
    return openai.Embedding.create(input=text, model=model_name)


