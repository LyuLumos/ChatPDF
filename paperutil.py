import arxiv
import os
from typing import Generator
import logging
from PyPDF2 import PdfReader
from tqdm import tqdm


def make_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        logging.info("Directory created: %s", path)
    else:
        logging.info("Directory already exists: %s", path)


def get_relevant_arxiv_papers(query, output_path, max_results=1, sort_by=arxiv.SortCriterion.Relevance):
    res = arxiv.Search(
        query=query, max_results=max_results, sort_by=sort_by
    )
    save_paper_info(res.results(), output_path)


def get_arxiv_papers(id_list, output_path):
    # 返回指定的arxiv论文信息，例如 get_arxiv_papers("1605.08386v1", "downloads")
    res = arxiv.Search(id_list=[id_list])
    save_paper_info(res.results(), output_path)


def save_paper_info(search_res: Generator, output_path):
    papers_infos = []
    for paper in search_res:
        paper_dict = {
            "title": paper.title,
            "authors": [author.name for author in paper.authors],
            "summary": paper.summary,
            "pdf_url": [link.href for link in paper.links if link.title == "pdf"][0],
            "arxiv_url": [link.href for link in paper.links][0],
            "journal_ref": paper.journal_ref,
            "primary_category": paper.primary_category,
            "categories": paper.categories,
        }
        papers_infos.append(paper_dict)
        # for i in paper_dict:
        #     print(i, ':', paper_dict[i])

        pdf_path = paper.download_pdf(output_path)
        paper_dict["pdf_path"] = pdf_path
        logging.info("Downloaded paper: %s", paper.title)

    # write to csv
    with open(os.path.join(output_path, 'paper.csv'), "w", encoding='utf-8') as f:
        f.write(
            "title,authors,summary,pdf_url,arxiv_url,journal_ref,primary_category,categories,pdf_path\n")
        for paper in papers_infos:
            f.write(
                f"{paper['title']},{paper['authors']},{repr(paper['summary'])},{paper['pdf_url']},{paper['arxiv_url']},{paper['journal_ref']},{paper['primary_category']},{paper['categories']},{paper['pdf_path']}\n"
            )
    return papers_infos


def parse_pdf(filepath):
    reader = PdfReader(filepath)
    pdf_text = ""
    logging.info(f"Parsing %s", filepath)
    for page in enumerate(tqdm(reader.pages)):
        pdf_text += page[1].extract_text() + f"\nPage {page[0] + 1}\n"
    return pdf_text


def create_chunks(text, chunk_size, tokenizer):
    logging.info("Creating chunks.")
    tokens = tokenizer.encode(text)
    i = 0
    while i < len(tokens):
        j = min(i + int(1.5 * chunk_size), len(tokens))
        while j > i + chunk_size//2:
            chunk = tokenizer.decode(tokens[i:j])
            if chunk.endswith(('.', '?', '!', '\n')):
                break
            j -= 1
        if j == i + chunk_size//2:
            j = min(i + chunk_size, len(tokens))
        yield tokens[i:j]
        i = j


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # make_directory("downloads")
    # get_relevant_arxiv_papers("quantum computing", "downloads")
    get_arxiv_papers("1605.08386v1", "downloads")
    # print(parse_pdf("downloads/1605.08386v1.Heat_bath_random_walks_with_Markov_bases.pdf"))
    # test()
