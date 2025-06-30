# LightFM-recommender

Model created for Amazon product recommendation for [Data Science Club PJATK kaggle challenge](https://www.kaggle.com/competitions/product-recommendation-challenge).

The idea is to create a model that recommends each user a new items as accurately as possible. Dataset contains item mapping, test and train sets and product metadata as well.

# Submission journey

| Model Name | Score |
|------------|-------|
| 1. Top 10 most popular | 0.02212 |
| 2. LightFM first model | 0.04186 |
| 3. LightFM tuned | Score: 0.04649 |
| 4. LFM with features | Score: 0.04711 |
| 5. LFM revised, tuned with more computation time | Score: 0.05113 |

1. `Top 10 most popular` submission based only on top 10 most popular items to benchmark and evaluate basic information about dataset
2. `LightFM first model ` first, most basic usage of lightfm (https://making.lyst.com/lightfm/docs/home.html), basic evaluation to get first submission ready for evaluation.
3. `LightFM tuned ` incresed number of epochs, components and threads to see performance difference.
4. `LFM with features` adds items features into model, information about images, category.
5. `LFM revised, tuned with more computation time` increased model tuning, increased amount of features. Building interaction matrix between products and users. Created function that takes into account nulls in data.
