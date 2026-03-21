from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup


def fetch_url_text(url):
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Please enter a valid URL starting with http:// or https://")

    try:
        response = requests.get(
            url,
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response.raise_for_status()
    except requests.RequestException as error:
        raise ConnectionError(f"Could not fetch the webpage: {error}") from error

    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = " ".join(soup.stripped_strings)
    if not text:
        raise ValueError("Could not extract readable text from the URL.")

    return text
