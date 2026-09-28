# -*- coding: utf-8 -*-
"""
Dataset-specific configuration for KHALRS preprocessing.
"""

GOODREADS_KEEP = {
    "academic", "education", "educational", "science", "scientific",
    "technology", "engineering", "computer", "computing", "mathematics",
    "math", "medicine", "medical", "health", "economics", "finance",
    "business", "law", "professional", "reference", "textbook",
    "knowledge", "history", "philosophy", "psychology", "social science",
    "sociology", "politics", "policy", "career",
}

MIND_KEEP = {
    "education", "science", "technology", "career", "finance",
    "health", "policy", "academic", "school", "college", "university",
    "research", "engineering", "business", "economics",
}

MIND_DROP = {
    "entertainment", "sports", "sport", "crime", "celebrity",
    "music", "movie", "movies", "tv", "television", "games", "gaming",
}

DOUBAN_KEEP = {
    "academic", "professional", "textbook", "science", "technology",
    "medicine", "medical", "economics", "law", "education",
    "social science", "social sciences", "engineering", "computer",
    "mathematics", "finance", "business", "research",
}

RESOURCE_TYPE_MAP = {
    "book": "book",
    "books": "book",
    "ebook": "book",
    "e-book": "book",
    "journal": "journal",
    "e-journal": "journal",
    "article": "article",
    "news": "article",
    "paper": "article",
    "video": "multimedia",
    "audio": "multimedia",
    "multimedia": "multimedia",
    "courseware": "multimedia",
}

MIN_USER_INTERACTIONS = 5
MIN_ITEM_INTERACTIONS = 5

TRAIN_RATIO = 0.70
VALID_RATIO = 0.10
TEST_RATIO = 0.20
