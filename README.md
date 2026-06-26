# Zomato (Bangalore) Restaurant & Review Data Analysis

**Alfido Tech — Data Internship · Task 1**

Analysis of the Zomato **Bangalore** restaurants dataset to surface insights on **ratings, cuisines,
location (neighbourhood) preferences, and the factors that drive ratings**, with five concrete
recommendations for a restaurant-discovery platform.

Dataset: [Kaggle · bhanupratapbiswas/zomato](https://www.kaggle.com/datasets/bhanupratapbiswas/zomato)
(`zomato.csv`, ~56k rows, 13 columns — single city, Bengaluru).

---

## Deliverables
| File | Description |
|------|-------------|
| `zomato_analysis.ipynb` | Jupyter/Colab notebook — cleaning, EDA, visualizations, findings |
| `Zomato_Analysis_Report.pdf` | 12-page PDF report with key findings & 5 recommendations |
| `pipeline.py` | Reusable cleaning / EDA / visualization functions |
| `build_report.py` | Renders the PDF report from the pipeline outputs |
| `figures/` | Generated charts (heatmap, wordclouds, distributions, …) |

## Key findings
- ~26% of venues are **unrated** (incl. ~2.2k "NEW"); rated venues cluster tightly around **3.63/5**.
- Supply concentrates in **tech-corridor neighbourhoods** — Whitefield, Electronic City, BTM, HSR.
- **Popularity ≠ quality**: North Indian leads by volume; Asian / American / Continental rate highest.
- **Table booking is the strongest rating signal** (~4.12 vs ~3.57); price barely matters until the luxury tier.
- **Votes** are the strongest numeric correlate of rating (r ≈ 0.40) — engagement, not price.

## What's inside
- **Data cleaning** — repairs ~4.5k rows corrupted by un-escaped newlines (integrity filter on the
  strict Yes/No columns), parses `rate` ("4.1/5", "NEW", "-") to numeric, cleans cost, splits
  multi-value cuisines / rest-type, de-duplicates venues. *Single city → all INR, no currency conversion.*
- **EDA** — ratings, neighbourhood hotspots, cuisine vs rating, cost vs rating, online-order &
  table-booking lift, restaurant-type quality, service-type mix.
- **Visualizations** — rating distribution, top neighbourhoods/cuisines, **correlation heatmap**,
  **most-liked-dishes wordcloud**, cuisine wordcloud, cost–rating, votes scatter, service charts.

## Run it
```bash
pip install -r requirements.txt
# Place zomato.csv in this folder, then:
jupyter notebook zomato_analysis.ipynb     # or open in Google Colab
```
To (re)build just the PDF:
```bash
python build_report.py
```

## Getting the data via Kaggle API
```bash
pip install kaggle
kaggle datasets download -d bhanupratapbiswas/zomato --unzip
```

---
*Note: `zomato.csv` is git-ignored by default (don't redistribute the raw dataset). Remove it from
`.gitignore` if your submission requires committing the data.*
