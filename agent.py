from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_random_exponential
import config


# @retry(stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=1, max=40))
# def chat_request()