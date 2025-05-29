# main.py
import os
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import datetime
from collections import Counter
import pickle, re, numpy as np, pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from mastodon_client import get_user_info_and_posts
from startup import download_and_extract_models
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Political Leaning Dashboard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify your frontend URL(s)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

download_and_extract_models()

print("Current directory:", os.getcwd())
print("Files in model dir:", os.listdir("./models"))

# -----------------------------------------------------------------------------
# 1. Load & register models
# -----------------------------------------------------------------------------
# Classical pipeline
with open("./models/tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)
with open("./models/lr_model.pkl", "rb") as f:
    lr_model = pickle.load(f)

# Transformer pipeline (fine-tuned)
best_ckpt = "./models/distilroberta-base/content/results/checkpoint-368"  # or the folder with the best model
tokenizer = AutoTokenizer.from_pretrained(best_ckpt)
roberta_pipe = pipeline(
    "text-classification",
    model=best_ckpt,
    tokenizer=tokenizer,
    return_all_scores=False
)

# You can register more here...
MODELS = {
    "classical": {
        "predict": lambda texts: (
            lr_model.predict(tfidf.transform(texts)),
            lr_model.predict_proba(tfidf.transform(texts)).max(axis=1)
        )
    },
    "transformer": {
        "predict": lambda texts: (
            [int(r["label"].split("_")[-1]) for r in roberta_pipe(texts, truncation=True)],
            [r["score"] for r in roberta_pipe(texts, truncation=True)]
        )
    }
}

# -----------------------------------------------------------------------------
# Category Classification
# -----------------------------------------------------------------------------
CATEGORY_KEYWORDS = {
    "politics": ["election", "president", "government", "senate", "policy", "political", "vote"],
    "technology": ["ai", "machine learning", "software", "tech", "python", "github", "programming"],
    "health": ["covid", "vaccine", "mental health", "healthcare", "hospital", "fitness"],
    "economy": ["inflation", "stock market", "recession", "economy", "jobs", "gdp", "budget"],
    "environment": ["climate", "global warming", "sustainability", "carbon", "pollution", "renewable"],
    "social_issues": [
        "racism", "equality", "equity", "diversity", "inclusion",
        "lgbt", "lgbtq", "trans rights", "gender", "feminism",
        "abortion", "women's rights", "civil rights", "immigration",
        "refugees", "discrimination", "hate crime", "social justice",
        "police brutality", "black lives matter", "blm", "human rights",
        "pride", "same-sex marriage", "reproductive rights"
    ]
}

def assign_category(text: str) -> Optional[str]:
    text_lower = text.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in text_lower for keyword in keywords):
            return category
    return "Other"


# -----------------------------------------------------------------------------
# 2. Request & Data Models
# -----------------------------------------------------------------------------
def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[@#]\w+", "", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()

class Post(BaseModel):
    text: str
    timestamp: datetime
    engagement: int = Field(..., ge=0)
    category: Optional[str] = None

class AnalyzeRequest(BaseModel):
    username: str
    model: Literal["classical","transformer"]
    posts: List[Post]

def mastodon_post_to_post(p):
    text = p.get("content", "")
    return Post(
        text=text,
        timestamp=p.get("created_at"),
        engagement=p.get("favourites_count", 0) + p.get("reblogs_count", 0) + p.get("replies_count", 0),
        category=assign_category(text)
    )

# -----------------------------------------------------------------------------
# 3. Core analysis using selected model
# -----------------------------------------------------------------------------
def analyze_posts(posts: List[Post], model_key: str):
    if model_key not in MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model '{model_key}'")
    predictor = MODELS[model_key]["predict"]
    texts = [preprocess(p.text) for p in posts]
    labels, scores = predictor(texts)

    # Prepare original posts for lookup (result is always defined in analyze_posts caller)
    orig_posts = []
    import inspect
    frame = inspect.currentframe()
    while frame:
        if 'result' in frame.f_locals:
            orig_posts = frame.f_locals['result'].get('posts', [])
            break
        frame = frame.f_back
    per_post = []
    for p, lab, sc in zip(posts, labels, scores):
        orig = None
        for post_dict in orig_posts:
            if (getattr(p, "timestamp", None) == post_dict.get("created_at") and
                getattr(p, "text", None) == post_dict.get("content")):
                orig = post_dict
                break
        per_post.append({
            "text": p.text,
            "timestamp": p.timestamp,
            "engagement": p.engagement,
            "category": p.category,
            "label": int(lab),
            "confidence": float(sc),
            "id": orig["id"] if orig else None,
            "favourites_count": orig["favourites_count"] if orig else None,
            "reblogs_count": orig["reblogs_count"] if orig else None
        })

    overall = float(np.mean(labels)) if len(labels) > 0 else None
    cnt = Counter(labels)
    breakdown = {str(label): cnt[label] / len(labels) for label in range(5)}

    return per_post, overall, breakdown

# -----------------------------------------------------------------------------
# 4. Endpoints (with model selection)
# -----------------------------------------------------------------------------
@app.post("/analyze/user")
def analyze_user(
    username: str = Query(...),
    model_name: str = Query(default="classical")  # match your MODELS keys
):
    result = get_user_info_and_posts(username)
    posts = result.get("posts", []) if isinstance(result, dict) else []

    if not posts:
        return {"error": "User not found or no posts"}

    # Convert dicts to Post objects if needed
    post_objs = [mastodon_post_to_post(p) if isinstance(p, dict) else p for p in posts]

    per_post, overall, breakdown = analyze_posts(post_objs, model_name)

    # Compute bias label from overall score
    bias_map = {
        0: "extreme right",
        1: "center right",
        2: "center",
        3: "center left",
        4: "extreme left"
    }
    avg_label = round(overall) if overall is not None else None
    bias_label = bias_map.get(avg_label, None)

    # Compute average confidence
    avg_confidence = float(np.mean([p["confidence"] for p in per_post])) if per_post else None

    # --- Engagement by Bias Label ---
    df = pd.DataFrame(per_post)
    if not df.empty:
        engagement_by_label = {str(i): None for i in range(5)}
        group = df.groupby("label")["engagement"].mean()
        for i in range(5):
            if i in group.index:
                engagement_by_label[str(i)] = float(group[i])
    else:
        engagement_by_label = {str(i): None for i in range(5)}

    # --- Time Series Analysis ---
    if not df.empty:
        df["date"] = pd.to_datetime(df["timestamp"]).dt.date
        daily = df.groupby("date").agg(
            post_count=("text", "size"),
            avg_bias=("label", "mean")
        ).reset_index()
        time_series = {
            "dates": daily["date"].astype(str).tolist(),
            "post_counts": daily["post_count"].tolist(),
            "avg_bias": daily["avg_bias"].tolist()
        }
    else:
        time_series = {"dates": [], "post_counts": [], "avg_bias": []}

    # --- Category Bias ---
    if not df.empty:
        cat_bias = df.dropna(subset=["category"]).groupby("category")["label"].mean().to_dict()
    else:
        cat_bias = {}

    # Merge all user info except posts, and add analysis results
    user_info = {k: v for k, v in result.items() if k != "posts"}
    # Add num_posts from statuses_count if present
    if "statuses_count" in result:
        user_info["num_posts"] = result["statuses_count"]
    user_info.update({
        "username": username,
        "model": model_name,
        "overall_score": overall,
        "breakdown": breakdown,
        "per_post": per_post,
        "bias_label": bias_label,
        "confidence": avg_confidence,
        "engagement_by_label": engagement_by_label,
        "time_series": time_series,
        "category_bias": cat_bias
    })
    return user_info

@app.post("/analyze/text")
def analyze_text(
    text: str = Query(...),
    model_name: str = Query(default="classical")
):
    # Preprocess and create a Post object
    processed_text = preprocess(text)
    category = assign_category(text)
    post = Post(
        text=text,
        timestamp=datetime.now(),
        engagement=0,
        category=category
    )
    # Run model prediction
    labels, scores = MODELS[model_name]["predict"]([processed_text])
    label = int(labels[0])
    confidence = float(scores[0])
    bias_map = {
        0: "extreme right",
        1: "center right",
        2: "center",
        3: "center left",
        4: "extreme left"
    }
    bias_label = bias_map.get(label, None)
    return {
        "text": text,
        "model": model_name,
        "label": label,
        "bias_label": bias_label,
        "confidence": confidence,
        "category": category
    }

@app.get("/user/raw")
def get_user_raw(username: str = Query(...)):
    """
    Fetch raw user info and posts from Mastodon.
    """
    result = get_user_info_and_posts(username)
    return result
