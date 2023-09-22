from interaction import summarize
import logging
import os


logging.basicConfig(level=logging.INFO)

path = "/workspaces/ChatPDF/downloads/Gemini.pdf"
filename = path.split('/')[-1].split('.')[0]

if os.path.exists(f'{filename}_summary.md'):
    os.remove(f'{filename}_summary.md')

with open(f'{filename}_summary.md', 'w') as f:
    print(summarize(path, 'paper'), file=f)
logging.info("Done.")