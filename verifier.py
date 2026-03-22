import re

import requests


COMMON_NEWS_KEYWORDS = {
    "government",
    "president",
    "minister",
    "election",
    "court",
    "police",
    "report",
    "news",
    "breaking",
    "official",
    "statement",
    "international",
    "india",
    "world",
    "economy",
    "health",
    "education",
    "technology",
    "market",
    "sports",
}


def verify_news(text):
    words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())

    # Keep a few unique words as simple "search keywords".
    keywords = []
    for word in words:
        if word not in keywords:
            keywords.append(word)
        if len(keywords) == 5:
            break

    search_query = " ".join(keywords)

    try:
        response = requests.get(
            "https://www.google.com/search",
            params={"q": search_query},
            timeout=5,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        search_success = response.status_code == 200
    except requests.RequestException:
        search_success = False

    matched_keywords = [word for word in keywords if word in COMMON_NEWS_KEYWORDS]

    if not search_query:
        score = 0.5
        explanation = "No useful keywords were found, so verification stayed neutral."
    elif matched_keywords:
        score = 0.8 if search_success else 0.65
        explanation = (
            f"Includes trusted news-style words like {', '.join(matched_keywords)}, "
            "so it looks closer to a normal news report."
        )
    else:
        score = 0.45 if search_success else 0.4
        explanation = (
            "Strong news-specific keywords were limited, so the verification signal stayed cautious."
        )

    if search_query and search_success:
        explanation += " A basic web search request also succeeded."
    elif search_query:
        explanation += " No reliable sources were confirmed from the simple web request."

    return round(score, 2), explanation
