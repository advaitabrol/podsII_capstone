# Baseline Model: Global Mean with User and Movie Biases

## Approach

Before implementing our primary matrix factorization model, we establish a baseline to quantify improvement. Our baseline predicts ratings using additive bias terms:

$$\hat{r}_{ui} = \mu + b_u + b_i$$

where μ is the global mean rating, $b_u$ captures how user *u* deviates from the global mean, and $b_i$ captures how movie *i* deviates from the global mean.

## Rationale

This baseline captures two fundamental patterns in rating data:

1. **User tendencies** — Some users are consistently generous (rating everything 4-5 stars), while others are harsh critics (averaging 2-3 stars). The user bias $b_u$ absorbs this individual tendency.

2. **Movie quality** — Some movies are universally acclaimed, others universally disliked. The movie bias $b_i$ captures this intrinsic quality signal.

We chose additive biases over a weighted average of user and movie means because averaging would double-count the global mean. For example, if a user averages 4.0 and a movie averages 4.0 (with global mean 3.59), a 50-50 weighted average predicts 4.0—but both component means already include the global baseline. The additive formulation correctly predicts 3.59 + 0.41 + 0.41 = 4.41.

This baseline is also the foundation upon which SVD-based methods build. The BellKor team that won the Netflix Prize used exactly this bias structure as their starting point, then added latent factor interactions to capture more nuanced user-movie compatibility.

## Results

| Parameter | Value |
|-----------|-------|
| Global mean (μ) | 3.5924 |
| User bias range | -2.59 to +1.41 |
| Movie bias range | -2.32 to +1.08 |
| **Test RMSE** | **1.0001** |

The bias ranges confirm substantial variation in both user generosity and movie quality—user biases span nearly 4 rating points, indicating some users average close to 1 star while others average close to 5.

## Interpretation

An RMSE of 1.0001 means our baseline predictions are off by approximately 1 star on average. For context, a naive model predicting the global mean for every rating achieves RMSE ≈ 1.09 (std of ratings), so capturing user and movie biases alone reduces error by roughly 10%.

However, this baseline ignores the key insight that makes collaborative filtering powerful: the interaction between specific users and specific movies. A user who loves comedies but hates horror will rate those genres differently, even controlling for overall generosity. Our SVD++ model will capture these latent preference patterns by learning low-dimensional representations of users and movies, targeting RMSE in the 0.85-0.92 range.