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

## 2. Requirements

Python 3.9 or later is recommended.

Install the required packages with:

```bash
pip install pandas numpy pyarrow
```

`pyarrow` is only needed when Parquet files are used.

---

## 3. Input files

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

## 4. Optional files

### Course mapping file

An optional course/curriculum file can be supplied to construct:

- `alignedWithCourse`
- `prerequisiteOf`

relations.

Example:

```csv
course,concept,curriculum,prerequisite
machine learning,deep learning,data science,linear algebra
```

### Co-citation file

An optional co-citation file can be supplied to construct the `coCited` relation.

Example:

```csv
item_a,item_b
b001,b005
b003,b007
```

---

## 5. Processing procedure

### 5.1 Duplicate removal

Repeated records with identical:

```text
user
item
timestamp
```

are removed.

### 5.2 Metadata normalization

The implementation performs:

- Unicode NFKC normalization;
- lowercase conversion for English text;
- whitespace normalization;
- normalization of common resource type labels;
- splitting of multi-value category, keyword, author, and discipline fields.

### 5.3 Dataset-specific filtering

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

## 6. 5-core interaction filtering

The manuscript retains users and items with at least five interactions.

The implementation performs iterative 5-core filtering:

```text
user interaction count >= 5
item interaction count >= 5
```

The procedure is repeated until no additional user or item falls below the threshold.

---

## 7. Chronological split

Interactions are sorted by timestamp separately for each user.

The split is:

```text
Train       earliest 70%
Validation  next 10%
Test        latest 20%
```

This prevents later interactions from entering the training set for the same user.

---

## 8. Knowledge graph construction

The implementation supports the five relation types described in the manuscript:

```text
hasConcept
coCited
sameDiscipline
prerequisiteOf
alignedWithCourse
```

### `hasConcept`

Constructed from item metadata such as:

- categories;
- keywords;
- titles;
- abstracts;
- authors;
- publishers.

### `coCited`

Constructed from an optional item-item co-citation or co-occurrence table.

### `sameDiscipline`

Connects concepts associated with the same discipline.

### `prerequisiteOf`

Constructed from course prerequisite metadata.

### `alignedWithCourse`

Connects courses with concepts or curriculum nodes.

---

## 9. Running the code

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

## 10. Custom column names

If the raw dataset uses different field names, they can be specified directly.

Example:

```bash
python main.py \
  --dataset goodreads \
  --interactions reviews.csv \
  --metadata books.csv \
  --output-dir processed/goodreads \
  --user-col user \
  --item-col book_id \
  --time-col date_added \
  --meta-item-col book_id \
  --title-col name \
  --category-col genres
```

This design avoids assuming that Goodreads, MIND, and Douban use identical raw schemas.

---

## 11. Adding curriculum information

Example:

```bash
python main.py \
  --dataset goodreads \
  --interactions data/interactions.csv \
  --metadata data/books.csv \
  --courses data/course_mapping.csv \
  --output-dir processed/goodreads
```

---

## 12. Adding co-citation information

Example:

```bash
python main.py \
  --dataset goodreads \
  --interactions data/interactions.csv \
  --metadata data/books.csv \
  --cocited data/cocited.csv \
  --output-dir processed/goodreads
```

---

## 13. Output files

Each run produces:

```text
train.csv
valid.csv
test.csv
metadata.csv
kg_triples.csv
user_mapping.csv
item_mapping.csv
stats.json
```

### `train.csv`

Training interactions.

### `valid.csv`

Validation interactions.

### `test.csv`

Test interactions.

### `metadata.csv`

Filtered and normalized item metadata.

### `kg_triples.csv`

Knowledge graph triples with columns:

```text
head
relation
tail
```

### `user_mapping.csv`

Mapping between raw and anonymized user IDs.

### `item_mapping.csv`

Mapping between raw and anonymized item IDs.

### `stats.json`

Summary statistics of the processed dataset.

Example:

```json
{
  "dataset": "goodreads",
  "users": 10000,
  "items": 25000,
  "interactions": 500000,
  "train": 350000,
  "valid": 50000,
  "test": 100000,
  "kg_triples": 120000,
  "relations": [
    "alignedWithCourse",
    "coCited",
    "hasConcept",
    "prerequisiteOf",
    "sameDiscipline"
  ]
}
```

---

## 14. Notes on reproducibility

The manuscript specifies the high-level construction pipeline but does not define one universal raw-file schema or a fixed NLP concept extraction model for all three source datasets.

For this reason:

- raw column names are configurable;
- metadata filters are deterministic;
- concept extraction uses a lightweight deterministic rule;
- train/validation/test splitting is deterministic;
- anonymized IDs are generated using SHA-256 with a fixed salt.

The salt can be changed with:

```bash
--salt your-own-salt
```

---

## 15. Scope

This repository intentionally contains **data preprocessing only**.

It does not include:

- KHALRS neural network implementation;
- KG embedding training;
- hierarchical attention training;
- BPR optimization;
- baseline models;
- evaluation metrics;
- statistical testing.

This separation makes the released preprocessing pipeline easier to audit and reproduce.
