"""
Netflix SVD using Surprise library
Run first: pip install "numpy<2.0" scikit-surprise --break-system-packages
"""
import pandas as pd
import numpy as np
from surprise import SVD, Dataset, Reader
import time

# === Config ===
SEED = 15414494
N_FACTORS = 64
N_EPOCHS = 20
LR = 0.005
REG = 0.02

np.random.seed(SEED)

# === Data Loading ===
def parse_ratings(path):
    movie_ids, user_ids, ratings = [], [], []
    current_movie = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.endswith(':'):
                current_movie = int(line[:-1])
            else:
                uid, r, _ = line.split(',')
                movie_ids.append(current_movie)
                user_ids.append(int(uid))
                ratings.append(int(r))
    return pd.DataFrame({'user_id': user_ids, 'item_id': movie_ids, 'rating': ratings})

print("Loading data...")
df = parse_ratings('data.txt')
print(f"Loaded {len(df):,} ratings")

# === Train/Test Split (per assignment: one rating per movie) ===
np.random.seed(SEED)
test_idx = df.groupby('item_id').apply(lambda x: x.sample(1).index[0]).values
test_df = df.loc[test_idx].reset_index(drop=True)
train_df = df.drop(test_idx).reset_index(drop=True)
print(f"Train: {len(train_df):,} | Test: {len(test_df):,}")

# === Baselines ===
rmse = lambda y, p: np.sqrt(np.mean((y - p) ** 2))
mu = train_df['rating'].mean()
user_bias = (train_df.groupby('user_id')['rating'].mean() - mu).to_dict()
item_bias = (train_df.groupby('item_id')['rating'].mean() - mu).to_dict()

baseline_preds = np.clip([mu + user_bias.get(u, 0) + item_bias.get(i, 0) 
                          for u, i in zip(test_df['user_id'], test_df['item_id'])], 1, 5)
baseline_rmse = rmse(test_df['rating'].values, baseline_preds)
print(f"Bias baseline RMSE: {baseline_rmse:.4f}")

# === Build Surprise Dataset ===
reader = Reader(rating_scale=(1, 5))
train_data = Dataset.load_from_df(train_df[['user_id', 'item_id', 'rating']], reader)
trainset = train_data.build_full_trainset()

# === Train Model ===
print(f"\nTraining SVD (factors={N_FACTORS}, epochs={N_EPOCHS})...")
print("Estimated time: ~20-40 min on CPU\n")

model = SVD(
    n_factors=N_FACTORS,
    n_epochs=N_EPOCHS,
    lr_all=LR,
    reg_all=REG,
    random_state=SEED,
    verbose=True
)

t0 = time.time()
model.fit(trainset)
print(f"\nDone in {(time.time() - t0)/60:.1f} min")

# === Evaluate ===
test_preds = np.array([model.predict(u, i).est for u, i in zip(test_df['user_id'], test_df['item_id'])])
test_rmse = rmse(test_df['rating'].values, test_preds)

print(f"\n{'='*44}")
print(f"Naive (global mean): {rmse(test_df['rating'].values, np.full(len(test_df), mu)):.4f}")
print(f"Bias baseline:       {baseline_rmse:.4f}")
print(f"SVD (Surprise):      {test_rmse:.4f} ({100*(baseline_rmse - test_rmse)/baseline_rmse:+.1f}%)")