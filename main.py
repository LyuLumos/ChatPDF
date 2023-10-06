from interaction import summarize
import logging
import os


logging.basicConfig(level=logging.INFO)

path = "/workspaces/ChatPDF/futureinternet-12-00027-v2.pdf"
filename = path.split('/')[-1].split('.')[0]

if os.path.exists(f'{filename}_summary.md'):
    os.remove(f'{filename}_summary.md')

with open(f'{filename}_summary.md', 'w') as f:
    print(summarize(path, 'paper-interview'), file=f)
logging.info("Done.")