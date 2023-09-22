from tenacity import retry, stop_after_attempt, wait_random_exponential
import openai
import tiktoken
import logging
import concurrent
from tqdm import tqdm
from paperutil import parse_pdf, create_chunks
import config

@retry(stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=1, max=40))
def get_embedding(text, model_name):
    return openai.Embedding.create(input=text, model=model_name)


def prompt_wrapper(prompt, text):
    return prompt + text


def get_api_response(prompt, text, model_name):
    prompt = prompt_wrapper(prompt, text)
    response = openai.ChatCompletion.create(
        model = model_name,
        messages = [{"role": "user", "content": prompt}],
        temperature = 0
    )
    return response.choices[0].message.content


def summarize(pdf_path, file_type='paper'):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    text = parse_pdf(pdf_path)
    chunks = create_chunks(text, 1500, tokenizer)
    text_chunks = [tokenizer.decode(chunk) for chunk in chunks]
    logging.info("Summarizing.")
    # summary_prompt = """Summarize this text from an academic paper. Extract any key points with reasoning.\n\nContent:"""
    sub_sum_prompts = {"survey": "你是一个专业的计算机专业研究人员，请按格式总结此综述：\n研究的问题\n研究方法\n关键技术创新点\n研究进展，如果没有此部分可以不写，并以markdown形式列出。",
                        "paper": "你是一个专业的计算机专业研究人员，请按格式总结文献：\n基础理论\n技术框架\n研究方法\n前沿进展\n创新建议，如果没有此部分可以不写，并以markdown形式列出。"}
    results = ""
    # 多线程总结每个chunk
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(text_chunks)) as executor:
        futures = [executor.submit(get_api_response, sub_sum_prompts[file_type], chunk, config.CHAT_MODEL) for chunk in text_chunks]
        with tqdm(total=len(futures)) as pbar:
            for _ in concurrent.futures.as_completed(futures):
                pbar.update(1)
        for future in futures:
            summary = future.result()
            results += summary

    sum_sum_prompts = {"survey": f"当前总结的内容为{results}，请按格式总结此综述：\n研究的问题\n研究方法\n关键技术创新点\n研究进展，并以markdown形式列出。",
                        "paper": f"当前总结的内容为{results}，请按格式总结文献：\n基础理论\n技术框架\n研究方法\n前沿进展\n创新建议，并以markdown形式列出。"}

    
    response = openai.ChatCompletion.create(
        model = config.LONG_MODEL,
        # messages = [{"role": "user", "content": f"""Write a summary collated from this collection of key points extracted from an academic paper.
        #                 The summary should highlight the core argument, conclusions and evidence.
        #                 The summary should be structured in bulleted lists following the headings Core Argument, Evidence, and Conclusions.
        #                 Key points:\n{results}\nSummary:\n"""
        # }],
        messages = [{"role": "user", "content": sum_sum_prompts[file_type]}],
        temperature = 0,
    )
    return response.choices[0].message.content



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(summarize("/workspaces/ChatPDF/downloads/1605.08386v1.Heat_bath_random_walks_with_Markov_bases.pdf"))