# Netflix Prize Capstone Project

Predicting movie ratings using collaborative filtering on the Netflix Prize dataset.

## Data Sources
- `data.txt` - 27M ratings from ~400k users on 5k movies
- `movieTitles.csv` - Movie metadata (id, year, title)

---

## Phase 1: Data Parsing

### Raw Data Format

**data.txt:**
```
758:
1488844,3,2005-09-06
822109,5,2005-05-13
759:
...
```
Lines ending with `:` indicate movie ID. Subsequent lines are `user_id,rating,date` for that movie.

**movieTitles.csv:**
```
1,2003,Dinosaur Planet,,
827,1994,C'est La Vie, Mon Cheri,
```
Format: `movie_id,year,title` with trailing commas (handled for titles containing commas).

### Parsed Output
- `ratings` DataFrame: `movie_id, user_id, date, rating`
- `movies` DataFrame: `movie_id, year, title`

---

## Phase 2: Exploratory Data Analysis

### Key Findings

| Metric | Value |
|--------|-------|
| Total ratings | 27,010,225 |
| Unique users | 472,542 |
| Unique movies | 5,000 |
| Sparsity | 98.86% |
| Rating mean | 3.59 |
| Rating median | 4.0 |
| Date range | 1999-11-11 to 2005-12-31 |

### Rating Distribution
| Rating | Count |
|--------|-------|
| 1 | 1,263,090 |
| 2 | 2,763,520 |
| 3 | 7,803,220 |
| 4 | 9,073,370 |
| 5 | 6,107,030 |

### User/Movie Statistics
- Ratings per user: min 1, max 4,963, median 27
- Ratings per movie: min 13, max 193,941, median 548

---

## Phase 3: Train/Test Split

Per assignment: Hold out one random rating per movie (5,000 test samples).

---

## Phase 4: Model

SVD with biases (collaborative filtering).

---

## Phase 5: Evaluation

RMSE on held-out test set.