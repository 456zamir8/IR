'''Web Crawling and Indexing 
A) Develop a web crawler to fetch and index web pages. 
B) Handle challenges such as robots.txt, dynamic content, and crawling delays. '''

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time

def crawl(start_url, max_pages=10):
    visited = set()
    to_visit = [start_url]
    page_count = 0

    while to_visit and page_count < max_pages:
        url = to_visit.pop(0)
        if url in visited:
            continue

        try:
            response = requests.get(url, timeout=5)
            if response.status_code != 200:
                continue

            soup = BeautifulSoup(response.text, 'html.parser')
            title = soup.title.string.strip() if soup.title else "No title"
            print(f"[{page_count + 1}] {title} — {url}")

            visited.add(url)
            page_count += 1

            for link in soup.find_all('a', href=True):
                full_url = urljoin(url, link['href'])
                if urlparse(full_url).netloc == urlparse(start_url).netloc:  # same domain
                    if full_url not in visited and full_url not in to_visit:
                        to_visit.append(full_url)

            time.sleep(1)  # Be nice to the server

        except Exception as e:
            print(f"Error accessing {url}: {e}")

# Example usage
start_url = "https://google.com"
crawl(start_url, max_pages=20)
