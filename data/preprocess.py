# -*- coding: utf-8 -*-
"""
Core preprocessing utilities for the KHALRS dataset construction pipeline.

The implementation follows the manuscript-level processing rules:
- remove duplicate user-item-timestamp records;
- retain users and items with at least five interactions;
- normalize metadata;
- construct dataset-specific academic/educational subsets;
- construct KG triples;
- split interactions chronologically at user level using 70/10/20.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from config import (
    GOODREADS_KEEP,
    MIND_KEEP,
    MIND_DROP,
    DOUBAN_KEEP,
    RESOURCE_TYPE_MAP,
    TRAIN_RATIO,
    VALID_RATIO,
)


# ---------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------

def read_table(path: Path) -> pd.DataFrame:
    suffixes = "".join(path.suffixes).lower()

    if suffixes.endswith(".parquet"):
        return pd.read_parquet(path)

    if suffixes.endswith(".jsonl") or suffixes.endswith(".jsonl.gz"):
        return pd.read_json(path, lines=True, compression="infer")

    if suffixes.endswith(".json") or suffixes.endswith(".json.gz"):
        try:
            return pd.read_json(path, lines=True, compression="infer")
        except ValueError:
            return pd.read_json(path, compression="infer")

    if ".tsv" in suffixes:
        return pd.read_csv(path, sep="\t", compression="infer", low_memory=False)

    return pd.read_csv(path, compression="infer", low_memory=False)


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


# ---------------------------------------------------------------------
# Text and metadata normalization
# ---------------------------------------------------------------------

def normalize_text(x) -> str:
    if pd.isna(x):
        return ""

    text = unicodedata.normalize("NFKC", str(x))
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_resource_type(x) -> str:
    value = normalize_text(x)
    if not value:
        return "unknown"
    return RESOURCE_TYPE_MAP.get(value, value)


def split_multi_value(x) -> List[str]:
    text = normalize_text(x)
    if not text:
        return []

    values = re.split(r"[|;,/、，；]+", text)
    return [value.strip() for value in values if value.strip()]


def merge_text_fields(df: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    columns = [c for c in columns if c and c in df.columns]

    if not columns:
        return pd.Series([""] * len(df), index=df.index, dtype="object")

    merged = pd.Series([""] * len(df), index=df.index, dtype="object")
    for column in columns:
        merged = merged + " " + df[column].fillna("").map(normalize_text)

    return merged.str.strip()


def normalize_metadata(
    metadata: pd.DataFrame,
    title_col: Optional[str],
    category_col: Optional[str],
    type_col: Optional[str],
    author_col: Optional[str],
    abstract_col: Optional[str],
    discipline_col: Optional[str],
) -> pd.DataFrame:
    df = metadata.copy()

    rename_targets = {
        title_col: "title",
        category_col: "category",
        type_col: "resource_type",
        author_col: "author",
        abstract_col: "abstract",
        discipline_col: "discipline",
    }

    for source, target in rename_targets.items():
        if source and source in df.columns:
            df[target] = df[source].map(normalize_text)

    if "resource_type" in df.columns:
        df["resource_type"] = df["resource_type"].map(normalize_resource_type)

    return df


# ---------------------------------------------------------------------
# Dataset-specific domain filtering
# ---------------------------------------------------------------------

def contains_any(text: str, terms: Set[str]) -> bool:
    return any(term in text for term in terms)


def dataset_domain_filter(
    metadata: pd.DataFrame,
    dataset: str,
    text_cols: Sequence[str],
) -> pd.DataFrame:
    text = merge_text_fields(metadata, text_cols)
    dataset = dataset.lower()

    if dataset == "goodreads":
        mask = text.map(lambda x: contains_any(x, GOODREADS_KEEP))

    elif dataset == "mind-edu":
        keep = text.map(lambda x: contains_any(x, MIND_KEEP))
        drop = text.map(lambda x: contains_any(x, MIND_DROP))
        mask = keep & ~drop

    elif dataset == "douban-acad":
        mask = text.map(lambda x: contains_any(x, DOUBAN_KEEP))

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    return metadata.loc[mask].copy()


# ---------------------------------------------------------------------
# Interaction preprocessing
# ---------------------------------------------------------------------

def parse_timestamp(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(series, errors="coerce")
        median = values.dropna().median() if values.notna().any() else np.nan

        if pd.notna(median):
            if median > 1e14:
                unit = "ns"
            elif median > 1e11:
                unit = "ms"
            else:
                unit = "s"

            return pd.to_datetime(values, unit=unit, utc=True, errors="coerce")

    return pd.to_datetime(series, utc=True, errors="coerce")


def hash_id(prefix: str, value: str, salt: str) -> str:
    raw = f"{salt}|{prefix}|{value}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:20]


def anonymize_ids(
    interactions: pd.DataFrame,
    user_col: str,
    item_col: str,
    salt: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    users = sorted(interactions[user_col].astype(str).unique())
    items = sorted(interactions[item_col].astype(str).unique())

    user_mapping = pd.DataFrame({
        "raw_user_id": users,
        "user_id": [hash_id("user", x, salt) for x in users],
    })

    item_mapping = pd.DataFrame({
        "raw_item_id": items,
        "item_id": [hash_id("item", x, salt) for x in items],
    })

    df = interactions.copy()
    df[user_col] = df[user_col].astype(str)
    df[item_col] = df[item_col].astype(str)

    df = df.merge(
        user_mapping,
        left_on=user_col,
        right_on="raw_user_id",
        how="left",
    )

    df = df.merge(
        item_mapping,
        left_on=item_col,
        right_on="raw_item_id",
        how="left",
    )

    df = df.drop(columns=[
        user_col,
        item_col,
        "raw_user_id",
        "raw_item_id",
    ])

    return df, user_mapping, item_mapping


def iterative_k_core(
    interactions: pd.DataFrame,
    min_user: int = 5,
    min_item: int = 5,
) -> pd.DataFrame:
    df = interactions.copy()

    while True:
        before = len(df)

        user_counts = df["user_id"].value_counts()
        valid_users = user_counts[user_counts >= min_user].index
        df = df[df["user_id"].isin(valid_users)]

        item_counts = df["item_id"].value_counts()
        valid_items = item_counts[item_counts >= min_item].index
        df = df[df["item_id"].isin(valid_items)]

        if len(df) == before:
            break

    return df.reset_index(drop=True)


def chronological_split(
    interactions: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_parts = []
    valid_parts = []
    test_parts = []

    for _, group in (
        interactions
        .sort_values(["user_id", "timestamp"])
        .groupby("user_id", sort=False)
    ):
        group = group.sort_values("timestamp").reset_index(drop=True)
        n = len(group)

        train_end = int(np.floor(TRAIN_RATIO * n))
        valid_end = int(np.floor((TRAIN_RATIO + VALID_RATIO) * n))

        train_end = max(1, min(train_end, n - 2))
        valid_end = max(train_end + 1, min(valid_end, n - 1))

        train_parts.append(group.iloc[:train_end])
        valid_parts.append(group.iloc[train_end:valid_end])
        test_parts.append(group.iloc[valid_end:])

    return (
        pd.concat(train_parts, ignore_index=True),
        pd.concat(valid_parts, ignore_index=True),
        pd.concat(test_parts, ignore_index=True),
    )


# ---------------------------------------------------------------------
# Metadata mapping and concept extraction
# ---------------------------------------------------------------------

def map_item_ids(
    metadata: pd.DataFrame,
    raw_item_col: str,
    item_mapping: pd.DataFrame,
) -> pd.DataFrame:
    df = metadata.copy()
    df[raw_item_col] = df[raw_item_col].astype(str)

    df = df.merge(
        item_mapping,
        left_on=raw_item_col,
        right_on="raw_item_id",
        how="inner",
    )

    return df.drop(columns=[raw_item_col, "raw_item_id"])


def extract_concepts(
    row: pd.Series,
    category_col: Optional[str] = "category",
    keyword_col: Optional[str] = "keywords",
    title_col: Optional[str] = "title",
    abstract_col: Optional[str] = "abstract",
    max_text_terms: int = 12,
) -> List[str]:

    concepts = set()

    for column in [category_col, keyword_col]:
        if column and column in row.index:
            concepts.update(split_multi_value(row[column]))

    text = ""
    for column in [title_col, abstract_col]:
        if column and column in row.index and pd.notna(row[column]):
            text += " " + normalize_text(row[column])

    tokens = re.findall(
        r"[a-z][a-z0-9+#.-]{2,}|[\u4e00-\u9fff]{2,8}",
        text,
    )

    stopwords = {
        "the", "and", "for", "with", "from", "this", "that", "into",
        "using", "based", "study", "book", "books", "article", "news",
    }

    concepts.update(
        token for token in tokens[:max_text_terms]
        if token not in stopwords
    )

    return sorted(x for x in concepts if x)


# ---------------------------------------------------------------------
# Knowledge graph construction
# ---------------------------------------------------------------------

def entity(kind: str, value: str) -> str:
    return f"{kind}:{normalize_text(value)}"


def build_kg(
    metadata: pd.DataFrame,
    courses: Optional[pd.DataFrame] = None,
    cocited: Optional[pd.DataFrame] = None,
    keyword_col: str = "keywords",
    publisher_col: str = "publisher",
    course_col: str = "course",
    course_concept_col: str = "concept",
    curriculum_col: str = "curriculum",
    prerequisite_col: str = "prerequisite",
    cocited_src_col: str = "item_a",
    cocited_dst_col: str = "item_b",
) -> pd.DataFrame:

    triples = set()
    discipline_map = {}

    for _, row in metadata.iterrows():
        item = entity("item", row["item_id"])

        concepts = extract_concepts(
            row,
            category_col="category" if "category" in metadata.columns else None,
            keyword_col=keyword_col if keyword_col in metadata.columns else None,
            title_col="title" if "title" in metadata.columns else None,
            abstract_col="abstract" if "abstract" in metadata.columns else None,
        )

        for concept in concepts:
            triples.add((
                item,
                "hasConcept",
                entity("concept", concept),
            ))

        if "author" in metadata.columns:
            for author in split_multi_value(row.get("author", "")):
                triples.add((
                    item,
                    "hasConcept",
                    entity("author", author),
                ))

        if publisher_col in metadata.columns:
            for publisher in split_multi_value(row.get(publisher_col, "")):
                triples.add((
                    item,
                    "hasConcept",
                    entity("publisher", publisher),
                ))

        if "discipline" in metadata.columns:
            disciplines = split_multi_value(row.get("discipline", ""))

            for concept in concepts:
                discipline_map.setdefault(concept, set()).update(disciplines)

    # sameDiscipline
    by_discipline = {}
    for concept, disciplines in discipline_map.items():
        for discipline in disciplines:
            by_discipline.setdefault(discipline, []).append(concept)

    for concepts in by_discipline.values():
        concepts = sorted(set(concepts))

        for a, b in zip(concepts[:-1], concepts[1:]):
            triples.add((
                entity("concept", a),
                "sameDiscipline",
                entity("concept", b),
            ))
            triples.add((
                entity("concept", b),
                "sameDiscipline",
                entity("concept", a),
            ))

    # coCited
    if cocited is not None and not cocited.empty:
        for _, row in cocited.iterrows():
            a = entity("item", row[cocited_src_col])
            b = entity("item", row[cocited_dst_col])

            triples.add((a, "coCited", b))
            triples.add((b, "coCited", a))

    # prerequisiteOf and alignedWithCourse
    if courses is not None and not courses.empty:
        for _, row in courses.iterrows():
            course = entity("course", row[course_col])

            if course_concept_col in row.index:
                for concept in split_multi_value(row.get(course_concept_col, "")):
                    triples.add((
                        course,
                        "alignedWithCourse",
                        entity("concept", concept),
                    ))

            if curriculum_col in row.index:
                for curriculum in split_multi_value(row.get(curriculum_col, "")):
                    triples.add((
                        course,
                        "alignedWithCourse",
                        entity("curriculum", curriculum),
                    ))

            if prerequisite_col in row.index:
                for prerequisite in split_multi_value(row.get(prerequisite_col, "")):
                    triples.add((
                        entity("course", prerequisite),
                        "prerequisiteOf",
                        course,
                    ))

    return pd.DataFrame(
        sorted(triples),
        columns=["head", "relation", "tail"],
    )


# ---------------------------------------------------------------------
# Main preprocessing routine
# ---------------------------------------------------------------------

def process_dataset(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    interactions = read_table(Path(args.interactions))
    metadata = read_table(Path(args.metadata))

    # Validate required columns
    for column in [args.user_col, args.item_col, args.time_col]:
        if column not in interactions.columns:
            raise KeyError(f"Missing interaction column: {column}")

    if args.meta_item_col not in metadata.columns:
        raise KeyError(
            f"Missing metadata item column: {args.meta_item_col}"
        )

    # Parse timestamps and remove exact duplicates
    interactions = interactions.copy()
    interactions[args.time_col] = parse_timestamp(
        interactions[args.time_col]
    )

    interactions = interactions.dropna(subset=[args.time_col])

    interactions = interactions.drop_duplicates(
        subset=[
            args.user_col,
            args.item_col,
            args.time_col,
        ],
        keep="first",
    )

    # Normalize metadata
    metadata = normalize_metadata(
        metadata,
        title_col=args.title_col,
        category_col=args.category_col,
        type_col=args.type_col,
        author_col=args.author_col,
        abstract_col=args.abstract_col,
        discipline_col=args.discipline_col,
    )

    text_cols = [
        column
        for column in [
            "title",
            "category",
            "abstract",
            "discipline",
            args.keyword_col
            if args.keyword_col in metadata.columns
            else None,
        ]
        if column
    ]

    # Dataset-specific academic/educational filtering
    metadata = dataset_domain_filter(
        metadata,
        args.dataset,
        text_cols,
    )

    allowed_items = set(
        metadata[args.meta_item_col].astype(str)
    )

    interactions[args.item_col] = (
        interactions[args.item_col].astype(str)
    )

    interactions = interactions[
        interactions[args.item_col].isin(allowed_items)
    ].copy()

    # Anonymize identifiers
    interactions, user_mapping, item_mapping = anonymize_ids(
        interactions,
        args.user_col,
        args.item_col,
        args.salt,
    )

    interactions = interactions.rename(
        columns={args.time_col: "timestamp"}
    )

    # 5-core filtering
    interactions = iterative_k_core(
        interactions,
        min_user=args.min_user_interactions,
        min_item=args.min_item_interactions,
    )

    retained_users = set(interactions["user_id"])
    retained_items = set(interactions["item_id"])

    user_mapping = user_mapping[
        user_mapping["user_id"].isin(retained_users)
    ].reset_index(drop=True)

    item_mapping = item_mapping[
        item_mapping["item_id"].isin(retained_items)
    ].reset_index(drop=True)

    # Map metadata IDs
    metadata = map_item_ids(
        metadata,
        args.meta_item_col,
        item_mapping,
    )

    metadata = (
        metadata[
            metadata["item_id"].isin(retained_items)
        ]
        .drop_duplicates("item_id")
        .reset_index(drop=True)
    )

    # User-level chronological split
    train, valid, test = chronological_split(interactions)

    courses = (
        read_table(Path(args.courses))
        if args.courses
        else None
    )

    cocited = (
        read_table(Path(args.cocited))
        if args.cocited
        else None
    )

    if cocited is not None:
        raw_to_anon = dict(
            zip(
                item_mapping["raw_item_id"],
                item_mapping["item_id"],
            )
        )

        for column in [
            args.cocited_src_col,
            args.cocited_dst_col,
        ]:
            cocited[column] = (
                cocited[column]
                .astype(str)
                .map(raw_to_anon)
            )

        cocited = cocited.dropna(
            subset=[
                args.cocited_src_col,
                args.cocited_dst_col,
            ]
        )

    kg = build_kg(
        metadata,
        courses=courses,
        cocited=cocited,
        keyword_col=args.keyword_col,
        publisher_col=args.publisher_col,
        course_col=args.course_col,
        course_concept_col=args.course_concept_col,
        curriculum_col=args.curriculum_col,
        prerequisite_col=args.prerequisite_col,
        cocited_src_col=args.cocited_src_col,
        cocited_dst_col=args.cocited_dst_col,
    )

    # Save outputs
    write_table(train, output_dir / "train.csv")
    write_table(valid, output_dir / "valid.csv")
    write_table(test, output_dir / "test.csv")
    write_table(metadata, output_dir / "metadata.csv")
    write_table(kg, output_dir / "kg_triples.csv")
    write_table(user_mapping, output_dir / "user_mapping.csv")
    write_table(item_mapping, output_dir / "item_mapping.csv")

    stats = {
        "dataset": args.dataset,
        "users": int(interactions["user_id"].nunique()),
        "items": int(interactions["item_id"].nunique()),
        "interactions": int(len(interactions)),
        "train": int(len(train)),
        "valid": int(len(valid)),
        "test": int(len(test)),
        "kg_triples": int(len(kg)),
        "relations": (
            sorted(kg["relation"].unique().tolist())
            if len(kg)
            else []
        ),
    }

    with open(
        output_dir / "stats.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            stats,
            file,
            ensure_ascii=False,
            indent=2,
        )

    return stats
