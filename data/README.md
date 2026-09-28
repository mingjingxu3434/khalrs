# KHALRS Data Preprocessing

This repository contains **only the data processing and dataset construction code** for the KHALRS study. It does not include model training, recommendation inference, or evaluation code.

The implementation follows the dataset construction procedure described in the manuscript:

1. remove duplicated user-item-timestamp interactions;
2. retain users and items with at least five interactions;
3. normalize metadata fields;
4. construct academic or education-oriented subsets;
5. generate knowledge graph triples;
6. split interactions chronologically for each user using a 70/10/20 train/validation/test ratio.

---

## 1. Project structure

```text
khalrs_data_processing/
├── config.py
├── preprocess.py
├── main.py
└── README.md
```

### `config.py`

Contains dataset-specific filtering vocabularies and global preprocessing settings.

### `preprocess.py`

Contains the main data processing functions:

- file loading;
- metadata normalization;
- domain filtering;
- user/item anonymization;
- 5-core filtering;
- chronological splitting;
- concept extraction;
- knowledge graph construction;
- processed file export.

### `main.py`

Command-line interface used to run the preprocessing pipeline.

---



## 2. Input files

At minimum, two files are required.

### Interaction file

The interaction table must contain:

```text
user_id
item_id
timestamp
```

The column names can be changed through command-line arguments.

Example:

```csv
user_id,item_id,timestamp
u001,b001,2025-01-01 09:20:00
u001,b003,2025-01-04 18:05:00
u002,b002,2025-01-08 14:10:00
```

### Metadata file

A typical metadata file may contain:

```text
item_id
title
category
keywords
resource_type
author
publisher
abstract
discipline
```

Not every field is mandatory. The pipeline uses fields that are actually available.

---



## 3. Processing procedure

### 3.1 Duplicate removal

Repeated records with identical:

```text
user
item
timestamp
```

are removed.

### 3.2 Metadata normalization

The implementation performs:

- Unicode NFKC normalization;
- lowercase conversion for English text;
- whitespace normalization;
- normalization of common resource type labels;
- splitting of multi-value category, keyword, author, and discipline fields.

### 3.3 Dataset-specific filtering

Three dataset modes are supported.

#### Goodreads

Keeps academic, educational, scientific, professional, and knowledge-oriented books.

#### MIND-Edu

Keeps records related to:

- education;
- science;
- technology;
- career;
- finance;
- health;
- policy.

Entertainment, celebrity, crime, gaming, and sports-oriented records are removed.

#### DouBan-Acad

Keeps books related to:

- academic subjects;
- textbooks;
- science;
- technology;
- medicine;
- economics;
- law;
- education;
- social sciences.

The filtering vocabulary is stored in `config.py` and can be modified without changing the processing code.

---

## 4. 5-core interaction filtering

The manuscript retains users and items with at least five interactions.

The implementation performs iterative 5-core filtering:

```text
user interaction count >= 5
item interaction count >= 5
```

The procedure is repeated until no additional user or item falls below the threshold.

---

## 5. Chronological split

Interactions are sorted by timestamp separately for each user.

The split is:

```text
Train       earliest 70%
Validation  next 10%
Test        latest 20%
```

This prevents later interactions from entering the training set for the same user.

---


## 6. Running the code

### Goodreads

```bash
python main.py \
  --dataset goodreads \
  --interactions data/goodreads_interactions.csv \
  --metadata data/goodreads_metadata.csv \
  --output-dir processed/goodreads
```

### MIND-Edu

```bash
python main.py \
  --dataset mind-edu \
  --interactions data/mind_interactions.tsv \
  --metadata data/mind_metadata.tsv \
  --output-dir processed/mind_edu
```

### DouBan-Acad

```bash
python main.py \
  --dataset douban-acad \
  --interactions data/douban_interactions.csv \
  --metadata data/douban_metadata.csv \
  --output-dir processed/douban_acad
```

---

