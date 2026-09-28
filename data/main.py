#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Command-line entry point for KHALRS data preprocessing.
"""

import argparse
import json

from config import (
    MIN_USER_INTERACTIONS,
    MIN_ITEM_INTERACTIONS,
)
from preprocess import process_dataset


def build_parser():
    parser = argparse.ArgumentParser(
        description="KHALRS data preprocessing only"
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=[
            "goodreads",
            "mind-edu",
            "douban-acad",
        ],
    )

    parser.add_argument(
        "--interactions",
        required=True,
    )

    parser.add_argument(
        "--metadata",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    # Interaction schema
    parser.add_argument(
        "--user-col",
        default="user_id",
    )

    parser.add_argument(
        "--item-col",
        default="item_id",
    )

    parser.add_argument(
        "--time-col",
        default="timestamp",
    )

    # Metadata schema
    parser.add_argument(
        "--meta-item-col",
        default="item_id",
    )

    parser.add_argument(
        "--title-col",
        default="title",
    )

    parser.add_argument(
        "--category-col",
        default="category",
    )

    parser.add_argument(
        "--keyword-col",
        default="keywords",
    )

    parser.add_argument(
        "--type-col",
        default="resource_type",
    )

    parser.add_argument(
        "--author-col",
        default="author",
    )

    parser.add_argument(
        "--publisher-col",
        default="publisher",
    )

    parser.add_argument(
        "--abstract-col",
        default="abstract",
    )

    parser.add_argument(
        "--discipline-col",
        default="discipline",
    )

    # Filtering thresholds
    parser.add_argument(
        "--min-user-interactions",
        type=int,
        default=MIN_USER_INTERACTIONS,
    )

    parser.add_argument(
        "--min-item-interactions",
        type=int,
        default=MIN_ITEM_INTERACTIONS,
    )

    # Anonymization
    parser.add_argument(
        "--salt",
        default="khalrs-2026",
    )

    # Optional course/curriculum mappings
    parser.add_argument(
        "--courses",
        default=None,
    )

    parser.add_argument(
        "--course-col",
        default="course",
    )

    parser.add_argument(
        "--course-concept-col",
        default="concept",
    )

    parser.add_argument(
        "--curriculum-col",
        default="curriculum",
    )

    parser.add_argument(
        "--prerequisite-col",
        default="prerequisite",
    )

    # Optional co-citation graph
    parser.add_argument(
        "--cocited",
        default=None,
    )

    parser.add_argument(
        "--cocited-src-col",
        default="item_a",
    )

    parser.add_argument(
        "--cocited-dst-col",
        default="item_b",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    stats = process_dataset(args)

    print(
        json.dumps(
            stats,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
