import os
import requests
from fastapi import APIRouter, Query, HTTPException
from dotenv import load_dotenv
router = APIRouter()
load_dotenv() 
# Load News API Key
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

# Ensure the API key is loaded
if not NEWS_API_KEY:
    raise ValueError("Missing NEWS_API_KEY. Ensure it is set in the environment variables.")

# News API Endpoint
NEWS_API_URL = "https://newsapi.org/v2/everything/"

def fetch_trending_news_by_topic(topic):
    """
    Fetches news articles based on a specific topic.
    """
    if not topic or len(topic) < 2:
        return {"error": "Invalid topic. Please provide a valid topic."}

    params = {
        "q": topic,
        "apiKey": NEWS_API_KEY,
        "sortBy": "relevancy",
        "language": "en",
    }

    try:
        response = requests.get(NEWS_API_URL, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data.get("status") != "ok":
            return {"error": f"News API Error: {data.get('message', 'Unknown error')}"}

        return [
            {
                "title": article.get("title", "No title available"),
                "description": article.get("description", "No description available."),
                "url": article.get("url", "#"),
                "source": article.get("source", {}).get("name", "Unknown Source"),
                "publishedAt": article.get("publishedAt", "Unknown Date"),
                "content": article.get("content", "Full article content is unavailable."),
            }
            for article in data.get("articles", [])
        ]

    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to fetch news: {str(e)}"}

@router.get("/api/news")
def get_news(topic: str = Query(..., description="The topic to search for")):
    """
    Fetches relevant news articles based on a given topic.
    Example usage: /news?topic=Technology
    """
    articles = fetch_trending_news_by_topic(topic)

    if "error" in articles:
        raise HTTPException(status_code=500, detail=articles["error"])

    return {"articles": articles}