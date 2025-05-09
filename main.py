from fastapi import FastAPI, HTTPException, Request,Body
from dotenv import load_dotenv
from pydantic import BaseModel#----------task3Correction
from pinecone import Pinecone, ServerlessSpec # Pinecone - NEW USAGE
from google.cloud import firestore # Firestore (Firebase)
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Query
from services.news_api_service import fetch_trending_news_by_topic
from openai import OpenAI  
from textstat import flesch_reading_ease
from deepeval.metrics import FaithfulnessMetric, ContextualRelevancyMetric, GEval
from deepeval.test_case import LLMTestCase
from deepeval import evaluate
from typing import  Dict, Optional, List
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import jaccard_score
import openai  # Import OpenAI SDK
import os
import requests
import re
import nltk
import string
import time
import json
import cohere



import ssl
import nltk

# ─── macOS SSL workaround so nltk.download can talk HTTPS ────────────────
ssl._create_default_https_context = ssl._create_unverified_context

# ─── make sure every needed model is present before anything else runs ────
for pkg in (
    "punkt",
    "punkt_tab",
    "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng",
    "stopwords",
):
    nltk.download(pkg, quiet=True)








#  Load .env file
load_dotenv(override=True)

#  Debug Print (BEFORE Importing news_api_service)
print("DEBUG: NEWS_API_KEY =", os.getenv("NEWS_API_KEY"))
print("DEBUG: OPENAI_API_KEY =", os.getenv("OPENAI_API_KEY"))

from services.news_api_service import fetch_trending_news_by_topic 

env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path, override=True) 

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
if not NEWS_API_KEY:
    raise ValueError("NEWS_API_KEY is missing! Check your .env file.")


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY is missing! Check your .env file.")

openai.api_key = OPENAI_API_KEY  # Set OpenAI API Key

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")  # e.g. "us-east-1"

if not PINECONE_API_KEY or not PINECONE_ENVIRONMENT:
    raise ValueError(
        "Pinecone API Key or Environment is missing in the .env file.\n"
        "Check PINECONE_API_KEY and PINECONE_ENVIRONMENT."
    )

# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ Only for debugging, NOT for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ArticleRequest(BaseModel):
    prompt: str
    user_id: str
    keywords: str = ""

# Set the path for Google credentials (for Firebase/Firestore)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./credentials/gennews-2e5b4-e2350b747f87.json"
if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
    raise EnvironmentError(
        "The GOOGLE_APPLICATION_CREDENTIALS environment variable is not set. "
        "Please set it to point to your service account JSON file."
    )

# Initialize Firestore (Firebase)
firestore_client = firestore.Client()

###############################################################################
# PINECONE INITIALIZATION - NEW SYNTAX
###############################################################################
# Create a Pinecone instance (no more pinecone.init())
pc = Pinecone(api_key=PINECONE_API_KEY)

# Define your Pinecone Index name
INDEX_NAME = "retrieval-engine"

# Check if the index exists; if not, create it.
existing_indexes = pc.list_indexes()  # returns a list of IndexSummary objects
existing_index_names = [i.name for i in existing_indexes]

if INDEX_NAME not in existing_index_names:
    # If you're on a Serverless (Starter) plan, you'll typically use ServerlessSpec
    # Make sure your region is correct, e.g., "us-east-1" or "us-west-2"
    # If you have a Dedicated plan or a different setup, adapt accordingly.
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,       # adjust dimension to match your embeddings
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",            # or 'gcp' depending on your Pinecone project
            region=PINECONE_ENVIRONMENT
        )
    )

# Now connect to the existing (or newly created) index
index = pc.Index(INDEX_NAME)

print(f"Successfully set up Pinecone Index: {INDEX_NAME}")


#implementation of evaluation metrics
# Load model for semantic similarity (Style Matching)

# 1. Faithfulness Evaluation (using QAG)
def evaluate_faithfulness(article, context):
    test_case = LLMTestCase(input=context, actual_output=article, retrieval_context=[context])
    metric = FaithfulnessMetric()
    result = metric.measure(test_case)
    return result

# 2. Contextual Relevancy Evaluation
def evaluate_contextual_relevancy(article, context):
    test_case = LLMTestCase(input=context, actual_output=article, retrieval_context=[context])
    metric = ContextualRelevancyMetric()
    result = metric.measure(test_case)
    return result

# 3. Style Matching Evaluation (Cosine Similarity + G-Eval)
#osine similarity from evaluate_style_matching and rely only on G-Eval

from sentence_transformers import SentenceTransformer, util

# Load model globally (if not already)
style_model = SentenceTransformer('all-MiniLM-L6-v2')  # Good balance of speed and accuracy

# ✅ New evaluate_style_matching using BERT
def evaluate_style_matching(generated_article: str, past_articles_for_evaluation: list):
    if not past_articles_for_evaluation:
        return {
            "style_similarity_score": None,
            "style_reasoning": "No past articles available for comparison."
        }

    try:
        # Combine past articles into one string
        combined_past = " ".join(past_articles_for_evaluation[:5])

        # ✅ Use SentenceTransformer embeddings
        embeddings = style_model.encode([generated_article, combined_past], convert_to_tensor=True)

        # Calculate cosine similarity
        cosine_sim = util.cos_sim(embeddings[0], embeddings[1]).item()

        return {
            "style_similarity_score": float(cosine_sim),
            "style_reasoning": "Similarity measured using BERT embeddings and cosine similarity."
        }

    except Exception as e:
        return {
            "style_similarity_score": None,
            "style_reasoning": f"Style Matching failed: {str(e)}"
        }

###############################################################################
# FASTAPI ROUTES
###############################################################################

@app.get("/")
async def read_root():
    """
    Health-check endpoint to verify the server is running.
    """
    return {"message": "FastAPI backend for retrieval engine and DeepSeek integration is running!"}

@app.post("/ingest-data/")
async def ingest_data(items: list[dict]):
    """
    Accepts a list of items with the structure:
    {
        "id": "unique_id",
        "content": "text to embed",
        "metadata": { "source": "Firecrawl", "category": "news" }
    }

    Then upserts them into the Pinecone index.
    """
    try:
        vectors = []
        for item in items:
            item_id = item["id"]
            text_content = item["content"]
            metadata = item.get("metadata", {})

            # Convert text to a vector (embedding)
            vector_values = generate_vector(text_content)  # Replace with real embedding

            vectors.append({
                "id": item_id,
                "values": vector_values,
                "metadata": metadata
            })

        # Upsert vectors into Pinecone
        index.upsert(vectors=vectors)
        return {"message": "Data ingested successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/")
async def search_pinecone(query: str, top_k: int = 5):
    """
    Searches Pinecone for relevant vectors given a query text.
    Returns the top_k matches from the index.
    """
    try:
        # Convert the query to an embedding vector
        query_vector = generate_vector(query)
        results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


openai.api_key = os.getenv("OPENAI_API_KEY")
UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY")

# ✅ Initialize OpenAI Client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # ✅ Set API key correctly
import re
import datetime

def remove_existing_signature(article_text, signature_keyword="By"):
    """
    Removes any trailing line that starts with the signature keyword (e.g., "By") from the article.
    """
    # This regex looks for a newline followed by optional whitespace and the signature_keyword,
    # then any characters until the end of the string.
    cleaned_text = re.sub(r'\n\s*' + re.escape(signature_keyword) + r'.*$', '', article_text, flags=re.IGNORECASE)
    return cleaned_text

@app.post("/generate-article/")
async def generate_article(request: ArticleRequest):
    """
    Generates a news article using OpenAI GPT-3.5, incorporating the journalist's writing style.
    At the end of the article, appends the journalist's title, first name, and last name as a signature.
    """
    print(f"🔎 Received request: {request}")

    prompt = request.prompt
    user_id = request.user_id
    keywords = request.keywords

    # Fetch Linguistic Print and journalist details from Firestore
    doc_ref = firestore_client.collection("Journalists").document(user_id)
    doc = doc_ref.get()
    linguistic_print = {}
    first_name = "Anonymous"
    last_name = ""
    user_title = ""

    if doc.exists:
        data = doc.to_dict()
        first_name = data.get("firstName", "Anonymous")
        last_name = data.get("lastName", "")
        user_title = data.get("title", "")
        previous_articles = data.get("previousArticles", [])
        exported_articles = [article.get("content", "") for article in data.get("exportedArticles", [])]
        if previous_articles or exported_articles:
            linguistic_print = extract_linguistic_print(previous_articles, exported_articles)
            print("\n✅ DEBUG: Extracted Linguistic Print:", linguistic_print)
            style_samples = "\n\n".join(previous_articles + exported_articles[:1])  # limit if too many
        else:
            style_samples=""

    # Build the GPT prompt
    gpt_prompt = f"""
You are an AI journalist writing in the exact style of the user based on their past work.

Below is a sample of the journalist’s writing style:
\"\"\"
{style_samples}
\"\"\"

Instructions:
- Match the writing style above as closely as possible.
- Use similar vocabulary, sentence structure, tone, and flow.
- If their style is poetic, use dramatic metaphors and Early Modern English.
- If their style is formal, keep it formal.
- If their style is casual, use modern expressions.

 Topic: {prompt}
 Keywords: {keywords}
"""

    try:
        # Call the OpenAI API using the constructed prompt
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": gpt_prompt}],
            max_tokens=2048,
            temperature=0.7,
            frequency_penalty=0.1,
            presence_penalty=0.2,
        )



        generated_article = response.choices[0].message.content.strip()
        # Remove any existing signature to avoid duplication.
        generated_article = remove_existing_signature(generated_article, "By")


        
        # Build a single signature string from the journalist's details.
        signature = f"By {user_title} {first_name} {last_name}".strip()
        
        # Append the signature only if it is not already present.
        if not generated_article.endswith(signature):
            generated_article_with_signature = f"{generated_article}\n\n{signature}"
        else:
            generated_article_with_signature = generated_article

        print("\n📝 Full Article with Signature:\n", generated_article_with_signature)

        # (Optional) Evaluate the generated article with your metrics here…
        retrieval_context = prompt
        faithfulness_score = evaluate_faithfulness(generated_article_with_signature, retrieval_context)
        contextual_relevancy_score = evaluate_contextual_relevancy(generated_article_with_signature, retrieval_context)
        past_articles_for_evaluation = previous_articles + exported_articles
        style_matching_results = evaluate_style_matching(generated_article_with_signature, past_articles_for_evaluation)

        evaluation_metrics = {
            "faithfulness_score": faithfulness_score,
            "contextual_relevancy_score": contextual_relevancy_score,
            "style_matching": style_matching_results
        }

        newChat = {
            "title":prompt,
            "versions": [generated_article_with_signature],
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "linguisticPrint": linguistic_print,
            "evaluationMetrics": evaluation_metrics,
        }

        generated_response = {
            "article": generated_article_with_signature,
            "linguistic_print": linguistic_print,
            "evaluation_metrics": evaluation_metrics,
        }

        # ✅ Save generated article to Firestore
        try:
            doc_ref = firestore_client.collection("Journalists").document(user_id)
            doc_ref.update({
                "savedArticles": firestore.ArrayUnion([{
        "title": prompt,
        "versions": [generated_article_with_signature],
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "linguisticPrint": linguistic_print
    }])
           })
            print("✅ Successfully saved generated article to Firestore (previousArticles).")
        except Exception as save_error:
         print(f"❌ Error saving article to Firestore: {str(save_error)}")

        print("\n📊  Final Evaluation Metrics:")
        print(f"faithfulness_score: {evaluation_metrics['faithfulness_score']}")
        print(f"contextual_relevancy_score: {evaluation_metrics['contextual_relevancy_score']}")
        print(f"style_similarity_score: {evaluation_metrics['style_matching']['style_similarity_score']}")

        return generated_response

    except openai.OpenAIError as e:
        print(f"❌ OpenAI API Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"OpenAI API Error: {str(e)}")
    except Exception as e:
        print(f"❌ General Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.get("/get-preferences/{user_id}")
async def get_preferences(user_id: str):
    """
    Retrieves user preferences from Firestore.
    """
    try:
        doc_ref = firestore_client.collection("users").document(user_id)
        doc = doc_ref.get()
        if doc.exists:
            return doc.to_dict()
        else:
            return {"message": "No preferences found for this user."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


###############################################################################
# HELPER FUNCTION: generate_vector
###############################################################################
def generate_vector(text: str):
    """
    Replace this function with a real embedding model call.
    E.g. using OpenAI:
    
    import openai
    openai.api_key = "YOUR_OPENAI_KEY"
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=text
    )
    embedding = response['data'][0]['embedding']
    return embedding
    """
    # Return a mock vector of zeros for demonstration (1536 dims).
    return [0.0] * 1536

#----------------------------------------------------------------------------task3

import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firebase Admin SDK
if not firebase_admin._apps:  # Prevent initializing multiple times
    cred = credentials.Certificate("./credentials/gennews-2e5b4-firebase-adminsdk-k3adz-af7308d3ec2.json")
    firebase_admin.initialize_app(cred)

#--------------------retrave articles
@app.get("/get-user-articles/{user_id}")
async def get_user_articles(user_id: str):
    """
    Retrieves all saved articles, previous articles, and exported articles from Firestore.
    """
    try:
        doc_ref = firestore_client.collection("Journalists").document(user_id)
        doc = doc_ref.get()

        if doc.exists:
            data = doc.to_dict()
            return {
                "previousArticles": data.get("previousArticles", []),  # Ensure default empty list
                "exportedArticles": data.get("exportedArticles", [])   # Ensure default empty list
            }
        else:
            return {"message": "No articles found for this user."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


#----------------------------extract linguistic print

#----------------------------extract linguistic print UPDATED

text = "This is a test sentence. Let's see if NLTK works properly!" #working check
print(sent_tokenize(text))

# Tell NLTK where to find the data
nltk.data.path.append(os.path.join(os.getcwd(), "nltk_data"))

# Set the path where NLTK should look for the data
NLTK_DATA_PATH = os.path.join(os.getcwd(), "venv", "lib", "python3.12", "site-packages", "nltk_data")
nltk.data.path.append(NLTK_DATA_PATH)

# Ensure necessary NLTK components are available
nltk.download("punkt", download_dir=NLTK_DATA_PATH)
nltk.download("averaged_perceptron_tagger", download_dir=NLTK_DATA_PATH)
nltk.download("stopwords", download_dir=NLTK_DATA_PATH)


def extract_linguistic_print(previous_articles, exported_articles):
    """
    Extracts a highly detailed linguistic profile from both `previousArticles` and `exportedArticles`.

    Parameters:
    - `previous_articles`: A list of strings (each article is a string).
    - `exported_articles`: A list of dictionaries where each has a `"content"` key.

    Returns:
    - A dictionary containing linguistic patterns such as sentence length, tone, common words, etc.
    """

    # Extract content from `exported_articles`
    exported_contents = [article.get("content", "") for article in exported_articles if "content" in article]

    # Combine both article sources
    all_text = " ".join(previous_articles + exported_contents)

    if not all_text.strip():  # Ensure there's meaningful content
        return {}

    # Tokenize sentences & words
    sentences = sent_tokenize(all_text)
    words = word_tokenize(all_text.lower())

    # **Sentence Length & Complexity**
    avg_sentence_length = sum(len(word_tokenize(sentence)) for sentence in sentences) / len(sentences)
    readability_score = flesch_reading_ease(all_text)  # Measures writing complexity

    # **Most Common Words (Expanded to 50)**
    word_freq = Counter(words)
    most_common_words = word_freq.most_common(50)  # 🔹 Increased from 15 → 50

    # **Extract Phrases: Bigrams, Trigrams, Quadgrams**
    bigrams = Counter(zip(words[:-1], words[1:])).most_common(30)  # 🔹 Increased from 15 → 30
    trigrams = Counter(zip(words[:-2], words[1:-1], words[2:])).most_common(20)
    quadgrams = Counter(zip(words[:-3], words[1:-2], words[2:-1], words[3:])).most_common(10)

    # **Punctuation & Writing Style**
    punctuation_freq = Counter(char for char in all_text if char in string.punctuation)
    most_common_punctuation = punctuation_freq.most_common(10)  # 🔹 Increased from 5 → 10

    # **Break Down Words by Part of Speech (POS)**
    pos_tags = nltk.pos_tag(words)  # Assign POS tags (noun, verb, etc.)
    nouns = [word for word, tag in pos_tags if tag.startswith("NN")]
    verbs = [word for word, tag in pos_tags if tag.startswith("VB")]
    adjectives = [word for word, tag in pos_tags if tag.startswith("JJ")]
    adverbs = [word for word, tag in pos_tags if tag.startswith("RB")]

    noun_freq = Counter(nouns).most_common(20)
    verb_freq = Counter(verbs).most_common(20)
    adj_freq = Counter(adjectives).most_common(15)
    adv_freq = Counter(adverbs).most_common(15)

    # **Tone Analysis**
    formal_words = {"hence", "thus", "therefore", "moreover", "consequently", "furthermore"}
    casual_words = {"lol", "hey", "gonna", "wanna", "gotta", "idk", "omg"}
    storytelling_words = {"once", "upon", "suddenly", "moment", "felt", "realized", "narrative"}

    formal_count = sum(1 for word in words if word in formal_words)
    casual_count = sum(1 for word in words if word in casual_words)
    storytelling_count = sum(1 for word in words if word in storytelling_words)

    if max(formal_count, casual_count, storytelling_count) == formal_count:
        tone = "Formal"
    elif max(formal_count, casual_count, storytelling_count) == casual_count:
        tone = "Casual"
    else:
        tone = "Storytelling"

    # **Active vs. Passive Voice**
    passive_voice_count = sum(1 for word in words if word in {"by", "was", "were", "had been"})
    active_voice_count = sum(1 for word in words if word in {"do", "does", "did", "has", "have", "had"})

    if active_voice_count > passive_voice_count:
        voice_preference = "Active Voice"
    else:
        voice_preference = "Passive Voice"

    # **Personal vs. Impersonal Writing**
    personal_words = {"i", "we", "me", "my", "mine", "our", "ours"}
    personal_count = sum(1 for word in words if word in personal_words)

    if personal_count > len(words) * 0.02:
        personal_vs_impersonal = "Personal"
    else:
        personal_vs_impersonal = "Impersonal"

    return {
        "avg_sentence_length": avg_sentence_length,
        "readability_score": readability_score,
        "most_common_words": most_common_words,
        "most_common_punctuation": most_common_punctuation,
        "most_common_bigrams": bigrams,
        "most_common_trigrams": trigrams,
        "most_common_quadgrams": quadgrams,
        "most_common_nouns": noun_freq,
        "most_common_verbs": verb_freq,
        "most_common_adjectives": adj_freq,
        "most_common_adverbs": adv_freq,
        "tone": tone,
        "voice_preference": voice_preference,
        "personal_vs_impersonal": personal_vs_impersonal,
    }

#-------------Modify the API to return the linguistic print
#hhhhhhh

@app.get("/get-linguistic-print/{user_id}")
async def get_linguistic_print(user_id: str):
    """
    Retrieves past articles from both 'previousArticles' and 'exportedArticles'
    and analyzes linguistic style based on their content.
    """
    try:
        doc_ref = firestore_client.collection("Journalists").document(user_id)
        doc = doc_ref.get()
        
        if doc.exists:
            data = doc.to_dict()
            previous_articles = data.get("previousArticles", [])
            exported_articles = data.get("exportedArticles", [])

            linguistic_print = extract_linguistic_print(previous_articles, exported_articles)

            return {"linguistic_print": linguistic_print}

        else:
            return {"message": "No articles found for this user."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/news")
async def get_news_by_topic(
    topic: Optional[str] = Query(None, description="Single topic query"),
    user_id: Optional[str] = Query(None, description="User ID for selected topics")
):
    def safe_get(d, key, default=None):
        return d.get(key, default) if isinstance(d, dict) else default

    def format_sources(newsapi_articles, google_data):
        sources = []

        for article in newsapi_articles[:2]:
            if isinstance(article, dict):
                sources.append({
                    "source": safe_get(article.get("source"), "name", "Unknown"),
                    "url": article.get("url", "#")
                })

        if isinstance(google_data, dict):
            for item in safe_get(google_data, "news_results", [])[:2]:
                if isinstance(item, dict):
                    sources.append({
                        "source": "Google News",
                        "url": item.get("link", "#")
                    })

        return sources

    try:
        if topic:
            newsapi_articles = fetch_trending_news_by_topic(topic)
            google_data = fetch_google_news(topic)

            summary = (
                newsapi_articles[0]["description"]
                if newsapi_articles and isinstance(newsapi_articles[0], dict)
                else "No summary available."
            )

            sources = format_sources(newsapi_articles, google_data)

            return {
                "title": newsapi_articles[0]["title"] if newsapi_articles else topic,
                "summary": summary,
                "sources": sources
            }

        elif user_id:
            doc_ref = firestore_client.collection("Journalists").document(user_id)
            doc = doc_ref.get()

            if not doc.exists:
                return {"message": "User not found."}

            user_data = doc.to_dict()
            selected_topics = user_data.get("selectedTopics", [])
            if not selected_topics:
                return {"message": "No selected topics for user."}

            articles = []
            for topic in selected_topics:
                newsapi_articles = fetch_trending_news_by_topic(topic)
                google_data = fetch_google_news(topic)

                summary = (
                    newsapi_articles[0]["description"]
                    if newsapi_articles and isinstance(newsapi_articles[0], dict)
                    else "No summary available."
                )

                sources = format_sources(newsapi_articles, google_data)

                articles.append({
                    "title": newsapi_articles[0]["title"] if newsapi_articles else topic,
                    "content": summary,
                    "sources": sources
                })

            return {"articles": articles}

        else:
            raise HTTPException(status_code=422, detail="Missing required query: topic or user_id")

    except Exception as e:
        print(f"❌ Error in /news: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")



@app.delete("/delete-articles")
async def delete_articles(user_id: str = Query(..., description="User ID of the journalist")):
    """
    Deletes all previous articles for a given user.
    """
    try:
        # Reference Firestore document
        doc_ref = firestore_client.collection("Journalists").document(user_id)

        # Update Firestore document to remove previous articles
        doc_ref.update({"previousArticles": []})

        return {"message": "All previous articles deleted successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting articles: {str(e)}")
    # the new API ////////////////////////////////////////////////
    

def fetch_google_news(q: str, gl: str = "us", hl: str = "en"):
    """
    Fetches news articles from Google News via the SerpApi with a freshness filter.
    """
    url = "https://serpapi.com/search.json"
    params = {
        "engine": "google_news",
        "q": q,
        "gl": gl,
        "hl": hl,
        "tbs": "qdr:h",  # Filter to news from the past hour
        "api_key": os.getenv("SERPAPI_API_KEY"),
        "output": "json"
    }
    response = requests.get(url, params=params)
    if response.ok:
        return response.json()
    else:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    
    #X open AI 

TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")
twitter_cache = {}  # ✅ Cache to store Twitter results (topic -> {tweets, timestamp})

def get_twitter_news(topic: str):
    """
    Fetches high-quality tweets (from trusted accounts with at least 1,000 likes) about the topic.
    Uses caching to reduce API calls.
    """
    current_time = time.time()

    # Use cached data if it is not older than 30 minutes.
    if topic in twitter_cache:
        last_fetch_time = twitter_cache[topic]["timestamp"]
        if current_time - last_fetch_time < 1800:
            print(f"🟡 Using Cached Twitter Data for: {topic}")
            return {"tweets": twitter_cache[topic]["tweets"]}

    url = "https://api.twitter.com/2/tweets/search/recent"

    # Build a query that targets tweets with the topic AND from trusted news sources:
    # (e.g., from:nytimes OR from:bbc OR from:reuters OR from:business)
    query = f'"{topic}" (from:nytimes OR from:bbc OR from:reuters OR from:business) -is:retweet -is:reply -is:quote'

    params = {
        "query": query,
        "tweet.fields": "public_metrics,created_at,text,author_id",
        "expansions": "author_id",
        "max_results": 15
    }

    headers = {"Authorization": f"Bearer {TWITTER_BEARER_TOKEN}"}
    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        data = response.json()
        if "data" not in data or not data["data"]:
            print("❌ No tweets found.")
            return {"tweets": []}
        tweets = data["data"]
        includes = data.get("includes", {}).get("users", [])

        # ✅ Build a dictionary of user data
        user_dict = {user["id"]: user for user in includes}

        # Filter tweets: only from verified users and with at least 1,000 likes
        filtered_tweets = sorted(
            [
                {
                    "text": tweet["text"],
                    "likes": tweet["public_metrics"]["like_count"],
                    "created_at": tweet["created_at"],
                    "author": user_dict.get(tweet["author_id"], {}).get("username", "Unknown"),
                    "verified": user_dict.get(tweet["author_id"], {}).get("verified", False)
                }
                for tweet in tweets
                if tweet["public_metrics"]["like_count"] >= 1000 and user_dict.get(tweet["author_id"], {}).get("verified", False)
            ],
            key=lambda x: x["likes"],
            reverse=True
        )[:5]

        twitter_cache[topic] = {"tweets": filtered_tweets, "timestamp": current_time}
        return {"tweets": filtered_tweets}

    elif response.status_code == 429:
        print("⏳ Twitter API Rate Limit Exceeded. Returning cached data if available.")
        return {"tweets": twitter_cache.get(topic, {}).get("tweets", [])}
    else:
        print(f"❌ Twitter API Error: {response.status_code} - {response.text}")
        return {"tweets": []}


# ✅ Generate prompt for image search from title and content
def generate_unsplash_prompt(title: str, content: str) -> str:
    words = title.split() + content.split()
    keywords = [w for w in words if w.isalpha()]
    return " ".join(keywords[:5])  # Use top 5 keywords

# Function to fetch images from Unsplash
UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY")

def fetch_unsplash_images(query: str, count: int = 9):
    try:
        url = "https://api.unsplash.com/search/photos"
        params = {
            "query": query,
            "client_id": UNSPLASH_ACCESS_KEY,
            "per_page": count,
            "orientation": "landscape"
        }

        response = requests.get(url, params=params)  # Fetch data from Unsplash
        response.raise_for_status()

        data = response.json()  # Parse the response
        results = data.get("results", [])

        image_urls = [img["urls"]["regular"] for img in results if "urls" in img]

        if image_urls:
            print(f"\n🖼️ Successfully fetched {len(image_urls)} image(s) from Unsplash for query: '{query}'")
            for url in image_urls:
                print(f"  - {url}")  # Print each image URL
        else:
            print(f"⚠️ No images found for query: '{query}'")

        return image_urls

    except Exception as e:
        print(f"⚠️ Error fetching images from Unsplash: {str(e)}")
        return []

# ✅ FastAPI endpoint to fetch Unsplash images based on article
@app.post("/unsplash-images/")
async def get_unsplash_images(request: Request):
    data = await request.json()  # Get the request body
    title = data.get("title")
    content = data.get("content")

    # Now you can use the title or content to generate your search query
    search_query = generate_unsplash_prompt(title, content)
    print(f"🔍 Searching Unsplash with: {search_query}")

    image_urls = fetch_unsplash_images(search_query)
    return {"images": image_urls}

# Pydantic model to receive article data
class ArticleDataNew(BaseModel):
    title: str
    content: str

# Function to generate a prompt for Unsplash search
def generate_unsplash_prompt(title: str, content: str) -> str:
    # You can use the title or content of the article to generate the search query for Unsplash
    return title + " " + content[:50]  # Example: combine title with first 50 characters of content

# FastAPI endpoint to fetch Unsplash images based on article
@app.post("/unsplash-images/")
async def get_unsplash_images(request: Request):
    data = await request.json()  # Get the request body
    title = data.get("title")
    content = data.get("content")

    # Generate the search query for Unsplash using article title and content
    search_query = generate_unsplash_prompt(title, content)
    print(f"🔍 Searching Unsplash with query: {search_query}")

    # Fetch images from Unsplash
    image_urls = fetch_unsplash_images(search_query)

    return {"images": image_urls}

# Function to fetch tweet images

# Twitter API configurations
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")
TWITTER_API_URL = "https://api.twitter.com/2/tweets/search/recent"
twitter_image_cache = {}

# Model to receive the topic from the frontend
class TopicImageRequest(BaseModel):
    topic: str

def get_twitter_images(topic: str):
    current_time = time.time()

    # ✅ Use cache if data is less than 30 minutes old
    if topic in twitter_image_cache:
        last_time = twitter_image_cache[topic]["timestamp"]
        if current_time - last_time < 1800:
            print(f"🟡 Using Cached Twitter Images for: {topic}")
            return {"images": twitter_image_cache[topic]["images"]}

    # ✅ Construct the image-focused query for Twitter API
    query = f'"{topic}" (verified OR from:nytimes OR from:bbc OR from:reuters) has:images -is:retweet -is:reply -is:quote'

    params = {
        "query": query,
        "tweet.fields": "public_metrics,created_at,text,author_id",
        "expansions": "author_id,attachments.media_keys",
        "media.fields": "url,type",
        "max_results": 30
    }

    headers = {"Authorization": f"Bearer {TWITTER_BEARER_TOKEN}"}
    response = requests.get(TWITTER_API_URL, headers=headers, params=params)

    if response.status_code == 200:
        data = response.json()
        
        if "data" not in data or not data["data"]:
            print("❌ No tweets found for the given topic.")
            return {"images": []}

        tweets = data["data"]
        media = data.get("includes", {}).get("media", [])
        users = data.get("includes", {}).get("users", [])

        user_dict = {user["id"]: user for user in users}
        media_dict = {item["media_key"]: item["url"] for item in media if item["type"] == "photo"}

        # ✅ Filter and sort tweets with images and verified authors
        tweet_images = []
        for tweet in tweets:
            author = user_dict.get(tweet["author_id"], {})
            if not author.get("verified", False):
                continue  # Skip if the author is not verified

            if "attachments" in tweet and "media_keys" in tweet["attachments"]:
                for media_key in tweet["attachments"]["media_keys"]:
                    image_url = media_dict.get(media_key)
                    if image_url:
                        tweet_images.append({
                            "url": image_url,
                            "likes": tweet["public_metrics"]["like_count"]
                        })

        # ✅ Sort images by likes and take the top 9 images
        sorted_images = sorted(tweet_images, key=lambda x: x["likes"], reverse=True)[:9]
        image_urls = [img["url"] for img in sorted_images]

        # ✅ Print out the fetched image URLs
        if image_urls:
            print(f"\n🖼️ Successfully fetched {len(image_urls)} image(s) for the topic '{topic}':")
            for url in image_urls:
                print(f"  - {url}")
        else:
            print(f"✅ No images found for the topic '{topic}'.")

        # ✅ Cache the result for future use
        twitter_image_cache[topic] = {"images": image_urls, "timestamp": current_time}

        return {"images": image_urls}

    elif response.status_code == 429:  # ✅ Handle Rate Limit Exceeded
        print("⏳ Twitter API Rate Limit Exceeded. Returning cached data if available.")
        return {"images": twitter_image_cache.get(topic, {}).get("images", [])}

    else:  # ❌ Handle other API errors
        print(f"❌ Twitter API Error: {response.status_code} - {response.text}")
        return {"images": []}

# Define the route for getting tweet images based on the topic
@app.post("/tweet-images/")
def get_tweet_images_endpoint(request: TopicImageRequest):
    return get_twitter_images(request.topic)


# -------- enhance article --------

class EnhancementFeedback(BaseModel):
    intended: str
    style: str

class ArticleEnhancementRequest(BaseModel):
    article: str
    userId: Optional[str] = None
    feedback: EnhancementFeedback

    # -------- Helper Functions --------
# Fetch user's linguistic profile from Firestore
def get_user_linguistic_profile(user_id: str) -> Dict:
    db = firestore.Client()  # assuming Firestore is set up
    doc_ref = db.collection('linguistic_profiles').document(user_id)
    doc = doc_ref.get()

    if doc.exists:
        return doc.to_dict()  # Fetches the linguistic profile from Firestore
    else:
        return {  # Return default values if no profile is found
            "tone": "Neutral",
            "avg_sentence_length": "Moderate",
            "readability_score": "Standard",
            "voice_preference": "Balanced",
            "personal_vs_impersonal": "Neutral",
            "most_common_words": [("example", 5)],
            "most_common_punctuation": [(".", 10)],
        }

# Function to remove existing signature from article (to avoid duplication)
def remove_existing_signature(article_text: str, keyword="By"):
    lines = article_text.strip().split("\n")
    return "\n".join(line for line in lines if not line.strip().startswith(keyword))

def build_enhancement_prompt(original_article, linguistic_print, feedback):
    reasons = []
    if feedback.get("intended") != "yes":
        reasons.append("it did not fully reflect the user's intended message")
    if feedback.get("style") != "similar":
        reasons.append("it did not match the user's writing style")
    reason_text = " and ".join(reasons) if reasons else "minor improvements are needed"

    return f"""
Context:
You are an experienced journalist and professional editor.

Writing Style Profile:
- Tone: {linguistic_print.get('tone', 'Neutral')}
- Sentence Length: {linguistic_print.get('avg_sentence_length', 'Moderate')}
- Readability Score: {linguistic_print.get('readability_score', 'Standard')}
- Voice Preference: {linguistic_print.get('voice_preference', 'Balanced')}
- Personal vs. Impersonal: {linguistic_print.get('personal_vs_impersonal', 'Neutral')}
- Common Words: {", ".join([w[0] for w in linguistic_print.get("most_common_words", [])[:10]])}
- Frequent Punctuation: {", ".join([p[0] for p in linguistic_print.get("most_common_punctuation", [])[:5]])}

Task:
The user has provided feedback that the article below needs improvements because {reason_text}.
Please revise the article to better align with the intended message and match the user's style.

Article to Enhance:
\"\"\"
{original_article}
\"\"\"

Output:
Return only the enhanced article.
"""

@app.post("/enhance-article/")
async def enhance_article(request: ArticleEnhancementRequest):
    try:
        print("🚀 Request received:", request.dict())

        article = request.article
        user_id = request.userId
        feedback = request.feedback

        print("✍️ Article:", article[:100])  # Print first 100 chars for brevity
        print("🧑‍💻 User ID:", user_id)
        print("📝 Feedback:", feedback)

        # Fetch user's style profile from Firestore
        linguistic_print = get_user_linguistic_profile(user_id)

        # Build GPT enhancement prompt dynamically
        enhancement_prompt = build_enhancement_prompt(
            original_article=request.article,
            linguistic_print=linguistic_print,
            feedback=request.feedback.dict()
        )

        # Call OpenAI API to enhance the article
        response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": enhancement_prompt}],
        max_tokens=2048,
        temperature=0.7,
        frequency_penalty=0.1,
        presence_penalty=0.2,
    )

        # Clean & finalize the article
        enhanced_article = response['choices'][0]['message']['content'].strip()
        enhanced_article = remove_existing_signature(enhanced_article, "By")

        # Optionally: append journalist signature
        signature = "By Journalist Name"
        if not enhanced_article.endswith(signature):
            enhanced_article += f"\n\n{signature}"

        return {"enhanced_article": enhanced_article}

    except Exception as e:
        print("❌ Error enhancing article:", e)
        raise HTTPException(status_code=500, detail="Enhancement failed")
    
@app.post("/generate-title/")
async def generate_title(request: dict = Body(...)):
    topic = request.get("topic", "")
    user_id = request.get("user_id", "")
    existing_titles = request.get("existing_titles", [])

    if not topic:
        raise HTTPException(status_code=400, detail="Topic is required")

    try:
        prompt = f"""
    Generate a unique, catchy, and relevant news article title based on the topic below:
    - Topic: {topic}
    Avoid reusing any of these titles:
    {existing_titles}

    The title should be short, professional, and related to current events.
    Only return the new title as a string.
    """

        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": prompt}],
            max_tokens=50,
            temperature=0.7,
        )

        title = response.choices[0].message.content.strip()
        return {"title": title}

    except Exception as e:
        print(f"❌ Error generating title: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate title")
    
    # News Api photo by wijdan 

def generate_newsapi_prompt(title: str, content: str) -> str:
    text = f"{title} {content}"
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())  # كلمات من 4 أحرف أو أكثر
    common_words = Counter(words).most_common(5)  # أكثر 5 كلمات تكرارًا
    keywords = [word for word, _ in common_words]
    return " ".join(keywords) if keywords else "latest news"

def fetch_newsapi_images(query: str, min_required: int = 3, max_results: int = 9):
    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "apiKey": NEWS_API_KEY,
            "pageSize": 30,
            "language": "en",
            "sortBy": "relevancy"
        }

        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        articles = data.get("articles", [])
        image_urls = []

        for article in articles:
            url = article.get("urlToImage")
            if url and url not in image_urls:
                image_urls.append(url)
            if len(image_urls) >= max_results:
                break

        if len(image_urls) < min_required:
            print(f"⚠️ Only found {len(image_urls)} image(s), less than required minimum.")
        else:
            print(f"🖼️ Found {len(image_urls)} image(s) for query: '{query}'")

        return image_urls

    except Exception as e:
        print(f"❌ Error fetching images from NewsAPI: {e}")
        return []
@app.post("/newsapi-images/")
async def get_newsapi_images(request: Request):
    data = await request.json()
    title = data.get("title", "")
    content = data.get("content", "")

    search_query = generate_newsapi_prompt(title, content)
    print(f"🔍 Generated prompt: {search_query}")

    image_urls = fetch_newsapi_images(search_query)
    return {"images": image_urls}

    # SerpApi photo by wijdan

def generate_serpapi_prompt(title: str, content: str) -> str:
    text = f"{title} {content}"
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    common_words = Counter(words).most_common(5)
    keywords = [word for word, _ in common_words]
    return " ".join(keywords) if keywords else "latest news"

def fetch_serpapi_images(query: str, min_required: int = 3, max_results: int = 9):
    try:
        url = "https://serpapi.com/search.json"
        params = {
            "engine": "google_images",
            "q": query,
            "api_key": os.getenv("SERPAPI_API_KEY"),
            "ijn": "0"  # أول صفحة من الصور
        }

        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        images_results = data.get("images_results", [])
        image_urls = []

        for image in images_results:
            link = image.get("original")
            if link and link not in image_urls:
                image_urls.append(link)
            if len(image_urls) >= max_results:
                break

        if len(image_urls) < min_required:
            print(f"⚠️ Only found {len(image_urls)} image(s), less than required minimum.")
        else:
            print(f"🖼️ Found {len(image_urls)} image(s) for query: '{query}'")

        return image_urls

    except Exception as e:
        print(f"❌ Error fetching images from SerpAPI: {e}")
        return []

@app.post("/serpapi-images/")
async def get_serpapi_images(request: Request):
    data = await request.json()
    title = data.get("title", "")
    content = data.get("content", "")

    search_query = generate_serpapi_prompt(title, content)
    print(f"🔍 Generated prompt: {search_query}")

    image_urls = fetch_serpapi_images(search_query)
    return {"images": image_urls}
