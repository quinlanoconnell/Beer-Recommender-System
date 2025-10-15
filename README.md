# Beer-Recommender-System
Web scraper to pull customer reviews of the top 150 ranked beers, in order to understand what makes a beer "awesome"

**Summary:** A content-based recommender built from 10k+ beer reviews. We clean reviews, extract attributes (n-grams), compute TF-IDF cosine similarity between beers, layer in sentiment, and compare against custom Word2Vec embeddings. We also sanity-check the corpus with a Zipf’s-law test and compute attribute **lift** to see which flavors truly stand out for a beer.

## Goal

Recommend beers that *match a target flavor profile* (e.g., “dark chocolate, tropical fruit, roasted”) and surface the most similar beers to a given product using only review text. Evaluate whether classic bag-of-words (TF-IDF + cosine) or learned embeddings (Word2Vec) better preserve attribute intent.

## Data Context

* **Source:** Beer review dataset (CSV assembled from scraped BeerAdvocate-style reviews).
* **Unit of analysis:** Individual text reviews aggregated to the **beer level** for modeling.
* **Tools:** Python (pandas, scikit-learn, nltk/spaCy, VADER, gensim), Jupyter.
* **Scope used here:** ~10,000 reviews (subset) for faster experimentation.

## Content Summary (What we did)

* **Data Cleaning / Engineering**

  * Normalize text, remove noise, drop empties, lemmatize/tokenize.
  * Aggregate all reviews per beer into a single “beer document.”
* **Zipf’s Law Check**

  * Verified heavy-tailed token distribution to validate BOW assumptions.
* **Attribute Extraction (N-grams)**

  * Built candidate flavor descriptors from unigrams/bigrams/trigrams.
  * Curated target attributes like **dark chocolate**, **tropical fruit**, **roasted**.
* **Normalize Frequency & Compute Lift**

  * For each beer × attribute: relative frequency vs. corpus baseline.
  * **Lift** highlights attributes unusually associated with a beer.
* **Sentiment Analysis**

  * VADER sentiment on reviews to up-weight positive mentions of attributes.
* **Bag-of-Words (TF-IDF) + Cosine Similarity**

  * Vectorized beer documents (stop-worded, min-df tuned).
  * Ranked nearest neighbors to a target beer and to target attributes.
* **Embeddings Comparison (Word2Vec)**

  * Trained custom word vectors on the same corpus; averaged to beer vectors.
  * Re-ranked similarity and compared to TF-IDF results for **attribute fidelity**.
* **Top-K Recommendations**

  * Produced **top-3** for a user-specified flavor set and a named target beer.
  * Observed how adding sentiment and switching to embeddings **changes** picks.

## Files Attached

* **`recommender_final.ipynb`** — End-to-end code for cleaning, feature engineering, TF-IDF & Word2Vec similarity, sentiment, lift, and recommendations.
* **`beer_reviews.csv`** — Review dataset (beer name + review text, etc.) used by the notebook.

