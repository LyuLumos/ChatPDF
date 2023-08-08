import arxiv
import os

def make_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print("Directory created: " + path)
    else:
        print("Directory already exists: " + path)


def get_relevant_arxiv_papers(query, output_path, max_results=1, sort_by=arxiv.SortCriterion.Relevance):
    res = arxiv.Search(
        query=query, max_results=max_results, sort_by=sort_by
    )
    papers_infos = []
    for paper in res.results():
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
        #     print(i, paper_dict[i])
        


def get_arxiv_papers(id_list, output_path):
    # 返回指定的arxiv论文信息，例如 get_arxiv_papers("1605.08386v1", "downloads")
    res = arxiv.Search(id_list=[id_list]).results()
    paper = next(res)
    print(paper)



if __name__ == "__main__":
    make_directory("downloads")
    get_relevant_arxiv_papers("quantum computing", "downloads")
    # get_arxiv_papers("1605.08386v1", "downloads")
    # test()
