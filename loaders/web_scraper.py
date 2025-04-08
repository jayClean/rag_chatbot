# loaders/web_scraper.py
import requests
from bs4 import BeautifulSoup

def extract_website_chunks(url: str):
    res = requests.get(url)
    soup = BeautifulSoup(res.text, "html.parser")
    content = "\n".join(p.get_text() for p in soup.find_all(['p', 'li']))
    return [{"text": content, "metadata": {"source": url}}]
