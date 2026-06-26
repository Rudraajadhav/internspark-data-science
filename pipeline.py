"""
Zomato (Bangalore) Data Analysis - core pipeline.
Dataset: Kaggle `bhanupratapbiswas/zomato` -> Zomato Bangalore restaurants
(56,252 raw rows, 13 columns; ~51,717 clean after dropping corrupted export rows).

Single city (Bangalore): "location" = neighbourhood, all prices in INR (no currency
conversion needed). Shared by the notebook and the PDF report builder.
"""
import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

ZRED, INK, ACCENT = "#CB202D", "#1f2933", "#2563eb"
PALETTE = ["#CB202D", "#F25C54", "#F4A259", "#5B8E7D", "#2563eb", "#7C3AED", "#0EA5E9"]

RAW_PATH = "zomato.csv"


# --------------------------------------------------------------------------------------
# 1. LOAD
# --------------------------------------------------------------------------------------
def load_data(path=RAW_PATH):
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            df = pd.read_csv(path, encoding=enc, engine="python")
            print(f"[load] {path}: {df.shape[0]:,} raw rows x {df.shape[1]} cols (encoding={enc})")
            return df
        except Exception:
            continue
    raise FileNotFoundError(f"Could not read {path}")


# --------------------------------------------------------------------------------------
# 2. CLEAN
# --------------------------------------------------------------------------------------
def _parse_rate(x):
    """'4.1/5' or '3.9 /5' -> 4.1 ; 'NEW','-',NaN -> NaN (unrated)."""
    if pd.isna(x):
        return np.nan
    m = re.search(r"(\d\.\d|\d)", str(x))
    if m and "NEW" not in str(x).upper():
        v = float(m.group(1))
        return v if 0 <= v <= 5 else np.nan
    return np.nan


def clean(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # ---- integrity filter: corrupted export rows have garbage in strict-format cols ----
    before = len(df)
    mask = df["online_order"].isin(["Yes", "No"]) & df["book_table"].isin(["Yes", "No"])
    df = df[mask].copy()
    corrupted = before - len(df)

    # ---- rate -> numeric, plus a 'new restaurant' flag ----
    df["rate_raw"] = df["rate"]
    df["rating"] = df["rate"].apply(_parse_rate)
    df["is_new"] = df["rate"].astype(str).str.upper().str.contains("NEW")
    df["is_rated"] = df["rating"].notna()

    # ---- votes ----
    df["votes"] = pd.to_numeric(df["votes"], errors="coerce").fillna(0).astype(int)

    # ---- cost: strip thousands separators, INR (single city -> no FX conversion) ----
    df["cost_for_two"] = pd.to_numeric(
        df["approx_cost(for two people)"].astype(str).str.replace(",", "", regex=False),
        errors="coerce")

    # ---- text fields ----
    df["online_order"] = df["online_order"].str.strip()
    df["book_table"] = df["book_table"].str.strip()
    df["location"] = df["location"].astype("string").str.strip()
    df["rest_type"] = df["rest_type"].astype("string").str.strip()
    df["listed_in(type)"] = df["listed_in(type)"].astype("string").str.strip()

    # cuisines -> primary cuisine + count
    df["cuisines"] = df["cuisines"].astype("string").str.strip()
    df["primary_cuisine"] = df["cuisines"].str.split(",").str[0].str.strip()
    df["num_cuisines"] = df["cuisines"].apply(
        lambda s: 0 if pd.isna(s) else len([x for x in str(s).split(",") if x.strip()]))
    df["primary_rest_type"] = df["rest_type"].str.split(",").str[0].str.strip()

    # ---- unique-restaurant view (dataset lists each venue once per service type) ----
    df["restaurant_key"] = (df["name"].astype(str).str.strip().str.lower() + "|" +
                            df["address"].astype(str).str.strip().str.lower())
    du = df.drop_duplicates(subset="restaurant_key").copy()

    summary = {
        "raw_rows": before,
        "corrupted_dropped": int(corrupted),
        "clean_listings": len(df),
        "unique_restaurants": len(du),
        "rated_pct": round(100 * df["is_rated"].mean(), 1),
        "new_count": int(df["is_new"].sum()),
        "missing_cost": int(df["cost_for_two"].isna().sum()),
        "missing_cuisine": int(df["primary_cuisine"].isna().sum()),
    }
    return df, du, summary


# --------------------------------------------------------------------------------------
# 3. ANALYZE
# --------------------------------------------------------------------------------------
def analyze(df, du):
    """Restaurant-level stats use `du` (unique venues); service-mix uses full `df`."""
    rated = du[du["is_rated"]]
    out = {}
    out["n_listings"] = len(df)
    out["n_unique"] = len(du)
    out["mean_rating"] = round(rated["rating"].mean(), 2)
    out["pct_unrated"] = round(100 * (~du["is_rated"]).mean(), 1)
    out["median_cost"] = int(du["cost_for_two"].median())
    out["pct_online_order"] = round(100 * (df["online_order"] == "Yes").mean(), 1)
    out["pct_book_table"] = round(100 * (df["book_table"] == "Yes").mean(), 1)

    out["top_locations"] = du["location"].value_counts().head(12)
    out["top_cuisines"] = du["primary_cuisine"].value_counts().head(12)

    cz = (rated.groupby("primary_cuisine")
          .agg(avg_rating=("rating", "mean"), count=("rating", "size")))
    out["cuisine_rating_top"] = cz[cz["count"] >= 40].sort_values("avg_rating", ascending=False).head(10)

    rt = (rated.groupby("primary_rest_type")
          .agg(avg_rating=("rating", "mean"), count=("rating", "size")))
    out["resttype_rating_top"] = rt[rt["count"] >= 40].sort_values("avg_rating", ascending=False).head(8)

    out["online_vs_rating"] = rated.groupby("online_order")["rating"].mean().round(3)
    out["booktable_vs_rating"] = rated.groupby("book_table")["rating"].mean().round(3)
    out["service_mix"] = df["listed_in(type)"].value_counts()

    out["corr_rating_votes"] = round(du["rating"].corr(du["votes"]), 3)
    out["corr_rating_cost"] = round(du["rating"].corr(du["cost_for_two"]), 3)
    out["corr_cost_votes"] = round(du["cost_for_two"].corr(du["votes"]), 3)

    # cost band vs rating
    du = du.copy()
    du["_cost_band"] = pd.qcut(du["cost_for_two"].rank(method="first"), 4,
                               labels=["Budget", "Mid", "Premium", "Luxury"])
    out["cost_band_rating"] = du[du["is_rated"]].groupby("_cost_band")["rating"].mean().round(3)
    out["cost_band_edges"] = du.groupby("_cost_band")["cost_for_two"].agg(["min", "max"])

    num = ["rating", "votes", "cost_for_two", "num_cuisines"]
    out["corr"] = du[num].corr()
    out["_du"] = du
    return out


# --------------------------------------------------------------------------------------
# 4. VISUALS
# --------------------------------------------------------------------------------------
def _save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def make_visuals(df, du, A):
    P = {}
    rated = du[du["is_rated"]]

    # 1 rating distribution
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.hist(rated["rating"], bins=np.arange(1.5, 5.05, 0.2), color=ZRED, edgecolor="white", alpha=.9)
    ax.axvline(rated["rating"].mean(), color=INK, ls="--", lw=1.5,
               label=f"Mean = {rated['rating'].mean():.2f}")
    ax.set(title="Distribution of Ratings (rated restaurants)", xlabel="Rating (out of 5)",
           ylabel="Restaurants"); ax.legend()
    P["rating_dist"] = _save(fig, "01_rating_distribution.png")

    # 2 top locations
    fig, ax = plt.subplots(figsize=(8, 5))
    A["top_locations"].sort_values().plot(kind="barh", color=ACCENT, ax=ax)
    ax.set(title="Top Bangalore Neighbourhoods by Restaurant Count", xlabel="Unique restaurants", ylabel="")
    P["top_locations"] = _save(fig, "02_top_locations.png")

    # 3 top cuisines
    fig, ax = plt.subplots(figsize=(8, 5))
    A["top_cuisines"].sort_values().plot(kind="barh", color=PALETTE[3], ax=ax)
    ax.set(title="Most Common Primary Cuisines", xlabel="Unique restaurants", ylabel="")
    P["top_cuisines"] = _save(fig, "03_top_cuisines.png")

    # 4 cuisine vs rating
    fig, ax = plt.subplots(figsize=(8, 4.8))
    cz = A["cuisine_rating_top"].sort_values("avg_rating")
    ax.barh(cz.index, cz["avg_rating"], color=PALETTE[5])
    ax.set_xlim(cz["avg_rating"].min() - 0.15, cz["avg_rating"].max() + 0.08)
    ax.set(title="Highest-Rated Cuisines (min. 40 restaurants)", xlabel="Average rating")
    P["cuisine_rating"] = _save(fig, "04_cuisine_vs_rating.png")

    # 5 cost vs rating (band means + scatter)
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    s = rated.sample(min(3000, len(rated)), random_state=1)
    ax.scatter(s["cost_for_two"], s["rating"], s=8, alpha=.25, color="#cbd5e1")
    cb = A["cost_band_rating"]
    edges = A["cost_band_edges"]
    xs = [(edges.loc[b, "min"] + edges.loc[b, "max"]) / 2 for b in cb.index]
    ax.plot(xs, cb.values, marker="o", color=ZRED, lw=2.2, label="Cost-band mean rating")
    for x, y, b in zip(xs, cb.values, cb.index):
        ax.annotate(f"{b}\n{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=8, fontweight="bold")
    ax.set(title="Cost for Two vs Rating", xlabel="Approx. cost for two (INR)", ylabel="Rating")
    ax.set_xscale("log"); ax.legend(loc="lower right")
    P["cost_rating"] = _save(fig, "05_cost_vs_rating.png")

    # 6 correlation heatmap
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    sns.heatmap(A["corr"], annot=True, fmt=".2f", cmap="RdBu_r", center=0, square=True,
                cbar_kws={"shrink": .8}, linewidths=.5, ax=ax)
    ax.set_title("Correlation Heatmap (numeric features)", fontweight="bold")
    P["heatmap"] = _save(fig, "06_correlation_heatmap.png")

    # 7 votes vs rating
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    s = rated.sample(min(3000, len(rated)), random_state=2)
    ax.scatter(s["votes"], s["rating"], s=9, alpha=.3, color=ACCENT)
    ax.set_xscale("symlog")
    ax.set(title=f"Votes vs Rating (corr = {A['corr_rating_votes']})",
           xlabel="Votes (symlog)", ylabel="Rating")
    P["votes_rating"] = _save(fig, "07_votes_vs_rating.png")

    # 8 dish wordcloud (popular dishes = review-derived highlights)
    dishes = df["dish_liked"].dropna().str.split(",").explode().str.strip()
    dishes = dishes[dishes.str.len() > 1]
    freq = dishes.value_counts().to_dict()
    wc = WordCloud(width=1100, height=540, background_color="white", colormap="Reds",
                   prefer_horizontal=.95, max_words=120).generate_from_frequencies(freq)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
    ax.set_title("Most-Liked Dishes Wordcloud", fontweight="bold")
    P["dish_wordcloud"] = _save(fig, "08_dish_wordcloud.png")

    # 8b cuisine wordcloud
    cf = du["cuisines"].dropna().str.split(",").explode().str.strip()
    cf = cf[cf.str.len() > 1].value_counts().to_dict()
    wc2 = WordCloud(width=1100, height=520, background_color="white", colormap="viridis",
                    prefer_horizontal=.95).generate_from_frequencies(cf)
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.imshow(wc2, interpolation="bilinear"); ax.axis("off")
    ax.set_title("Cuisine Popularity Wordcloud", fontweight="bold")
    P["cuisine_wordcloud"] = _save(fig, "08b_cuisine_wordcloud.png")

    # 9 online order / book table vs rating
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.9))
    for ax, key, title in [(axes[0], "online_vs_rating", "Online Ordering"),
                           (axes[1], "booktable_vs_rating", "Table Booking")]:
        d = A[key]
        ax.bar(d.index.astype(str), d.values, color=[PALETTE[4], "#cbd5e1"])
        for i, v in enumerate(d.values):
            ax.text(i, v + .02, f"{v:.2f}", ha="center", fontweight="bold")
        ax.set(title=f"Avg Rating by {title}", ylabel="Average rating",
               ylim=(0, max(d.values) + .6))
    P["service_rating"] = _save(fig, "09_service_vs_rating.png")

    # 10 rest_type ratings
    fig, ax = plt.subplots(figsize=(8, 4.4))
    rt = A["resttype_rating_top"].sort_values("avg_rating")
    ax.barh(rt.index, rt["avg_rating"], color=PALETTE[6])
    ax.set_xlim(rt["avg_rating"].min() - 0.15, rt["avg_rating"].max() + 0.08)
    ax.set(title="Highest-Rated Restaurant Types (min. 40)", xlabel="Average rating")
    P["resttype_rating"] = _save(fig, "10_resttype_vs_rating.png")

    # 11 service mix
    fig, ax = plt.subplots(figsize=(8, 4.2))
    A["service_mix"].sort_values().plot(kind="barh", color=PALETTE[2], ax=ax)
    ax.set(title="Listings by Service Type", xlabel="Listings", ylabel="")
    P["service_mix"] = _save(fig, "11_service_mix.png")

    return P


if __name__ == "__main__":
    raw = load_data()
    df, du, csum = clean(raw)
    A = analyze(df, du)
    P = make_visuals(df, du, A)
    print("clean:", csum)
    print("mean rating:", A["mean_rating"], "| unrated %:", A["pct_unrated"],
          "| median cost INR:", A["median_cost"])
    print("corr rating~votes:", A["corr_rating_votes"], "| rating~cost:", A["corr_rating_cost"])
    print("top loc:", list(A["top_locations"].index[:3]),
          "| top cuisine:", A["top_cuisines"].index[0],
          "| best cuisine:", A["cuisine_rating_top"].index[0])
    print("figures:", list(P.keys()))
