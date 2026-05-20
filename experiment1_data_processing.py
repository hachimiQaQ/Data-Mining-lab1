import ast
import itertools
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

TMDB_API_KEY = ""
INPUT_FILE = Path("movies.csv")
OUTPUT_FILE = Path("movies_basic_processed.csv")
CACHE_FILE = Path("tmdb_cache.json")
FAILED_TMDB_FILE = Path("tmdb_failed_records.csv")
TMDB_MOVIE_URL = "https://api.themoviedb.org/3/movie/{movie_id}"
MAX_TMDB_WORKERS = 16


DROP_REASON = {
    "homepage": "high missing rate and not useful for numeric mining",
    "tagline": "marketing text with many missing values",
    "overview": "long text, not used in this basic structured preprocessing step",
    "keywords": "nested text field, not required by the experiment's basic processing tasks",
    "original_title": "duplicates title information for this task",
    "production_companies": "nested organization text, high dimensional if expanded",
    "production_countries": "nested country text, not required in this step",
    "spoken_languages": "nested language text; original_language is kept as compact language feature",
    "cast": "large nested actor list, not required in the basic processing tasks",
    "id": "duplicates movie_id in this data set",
    "movie_id": "identifier only, no modeling meaning after TMDB lookup",
    "release_date": "converted into numeric release_year",
}


def parse_json_list(value):
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return value
    text = str(value).strip()
    if not text:
        return []
    try:
        result = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return []
    return result if isinstance(result, list) else []


def extract_names(value):
    names = []
    for item in parse_json_list(value):
        if isinstance(item, dict):
            name = str(item.get("name", "")).strip()
            if name:
                names.append(name)
    return names


def format_genres(value):
    return "|".join(extract_names(value))


def extract_directors(value):
    directors = []
    for item in parse_json_list(value):
        if not isinstance(item, dict):
            continue
        if item.get("job") == "Director":
            name = str(item.get("name", "")).strip()
            if name:
                directors.append(name)
    return "|".join(directors)


def load_cache():
    if not CACHE_FILE.exists():
        return {}
    try:
        with CACHE_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def save_cache(cache):
    with CACHE_FILE.open("w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def fetch_tmdb_movie_from_api(movie_id):
    key = str(movie_id)
    url = TMDB_MOVIE_URL.format(movie_id=urllib.parse.quote(key))
    query = urllib.parse.urlencode({"api_key": TMDB_API_KEY, "language": "en-US"})
    request_url = f"{url}?{query}"
    try:
        with urllib.request.urlopen(request_url, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
        return key, {
            "budget": data.get("budget") or 0,
            "revenue": data.get("revenue") or 0,
            "runtime": data.get("runtime") or 0,
            "genres": "|".join(
                genre.get("name", "").strip()
                for genre in data.get("genres", [])
                if genre.get("name", "").strip()
            ),
        }, None
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError):
        return key, {}, "failed"


def positive_or_missing(series):
    return series.isna() | (series <= 0)


def first_genre(genres_text):
    if pd.isna(genres_text) or not str(genres_text).strip():
        return "Unknown"
    return str(genres_text).split("|")[0] or "Unknown"


def fill_numeric_by_groups(df, column, stats):
    missing = positive_or_missing(df[column])
    if not missing.any():
        return

    positive = df[column] > 0
    group_median = (
        df.loc[positive]
        .groupby(["main_genre", "release_year"])[column]
        .median()
        .to_dict()
    )
    genre_median = df.loc[positive].groupby("main_genre")[column].median().to_dict()
    global_median = df.loc[positive, column].median()
    if pd.isna(global_median) or global_median <= 0:
        global_median = 1

    filled_count = 0
    for index in df.index[missing]:
        genre = df.at[index, "main_genre"]
        year = df.at[index, "release_year"]
        value = group_median.get((genre, year))
        if pd.isna(value) or value is None or value <= 0:
            value = genre_median.get(genre)
        if pd.isna(value) or value is None or value <= 0:
            value = global_median
        df.at[index, column] = value
        filled_count += 1

    stats[f"{column}_stat_filled"] = filled_count


def fill_genres_by_groups(df, stats):
    missing = df["genres_clean"].isna() | (df["genres_clean"].str.strip() == "")
    if not missing.any():
        return

    year_modes = {}
    for year, values in df.loc[~missing].groupby("release_year")["genres_clean"]:
        counter = Counter(values)
        if counter:
            year_modes[year] = counter.most_common(1)[0][0]

    all_counter = Counter(df.loc[~missing, "genres_clean"])
    global_mode = all_counter.most_common(1)[0][0] if all_counter else "Drama"

    filled_count = 0
    for index in df.index[missing]:
        year = df.at[index, "release_year"]
        df.at[index, "genres_clean"] = year_modes.get(year, global_mode)
        filled_count += 1

    stats["genres_stat_filled"] = filled_count


def apply_tmdb_fill(df, stats):
    cache = load_cache()
    need_api = (
        positive_or_missing(df["budget"])
        | positive_or_missing(df["revenue"])
        | positive_or_missing(df["runtime"])
        | (df["genres_clean"].isna())
        | (df["genres_clean"].str.strip() == "")
    )

    needed_ids = [str(movie_id) for movie_id in df.loc[need_api, "id"].tolist()]
    uncached_ids = sorted(set(movie_id for movie_id in needed_ids if movie_id not in cache))
    if uncached_ids:
        print(f"\nFetching {len(uncached_ids)} movies from TMDB API with {MAX_TMDB_WORKERS} workers...")
        failed_ids = []
        with ThreadPoolExecutor(max_workers=MAX_TMDB_WORKERS) as executor:
            futures = {
                executor.submit(fetch_tmdb_movie_from_api, movie_id): movie_id
                for movie_id in uncached_ids
            }
            completed = 0
            for future in as_completed(futures):
                movie_id, data, error = future.result()
                completed += 1
                if data:
                    cache[movie_id] = data
                    stats["api_success"] += 1
                else:
                    stats["api_failed"] += 1
                    failed_ids.append(movie_id)
                if completed % 100 == 0 or completed == len(uncached_ids):
                    save_cache(cache)
                    print(f"TMDB progress: {completed}/{len(uncached_ids)}")
        if failed_ids:
            failed_records = df[df["id"].astype(str).isin(failed_ids)][
                ["id", "title", "release_year", "budget", "revenue", "runtime", "genres_clean"]
            ].copy()
            failed_records.to_csv(FAILED_TMDB_FILE, index=False, encoding="utf-8-sig")
            print(f"TMDB failed records saved to {FAILED_TMDB_FILE}")

    for index in df.index[need_api]:
        movie_id = df.at[index, "id"]
        data = cache.get(str(movie_id), {})
        if data:
            stats["api_cache_hit"] += 1
        if not data:
            continue

        for column in ["budget", "revenue", "runtime"]:
            current_value = df.at[index, column]
            new_value = data.get(column)
            if (pd.isna(current_value) or current_value <= 0) and new_value and new_value > 0:
                df.at[index, column] = new_value
                stats[f"{column}_api_filled"] += 1

        current_genres = df.at[index, "genres_clean"]
        new_genres = data.get("genres", "")
        if (pd.isna(current_genres) or not str(current_genres).strip()) and new_genres:
            df.at[index, "genres_clean"] = new_genres
            stats["genres_api_filled"] += 1

    save_cache(cache)


def add_profit_level(df):
    profit_ratio = (df["revenue"] - df["budget"]) / df["budget"]
    df["profit_level"] = 1
    df.loc[(profit_ratio >= 0) & (profit_ratio < 0.6), "profit_level"] = 2
    df.loc[profit_ratio >= 0.6, "profit_level"] = 3


def add_genre_one_hot(df):
    all_genres = sorted(
        {
            genre
            for genres_text in df["genres_clean"].fillna("")
            for genre in str(genres_text).split("|")
            if genre
        }
    )
    for genre in all_genres:
        df[genre] = df["genres_clean"].apply(
            lambda value, target=genre: int(target in str(value).split("|"))
        )
    return all_genres


def print_director_statistics(df):
    director_counts = Counter()
    director_revenue = defaultdict(float)
    director_combo_counts = Counter()

    for _, row in df.iterrows():
        directors = [name for name in str(row["director"]).split("|") if name]
        revenue = float(row["revenue"])
        for director in directors:
            director_counts[director] += 1
            director_revenue[director] += revenue
        if len(directors) >= 2:
            for combo in itertools.combinations(sorted(set(directors)), 2):
                director_combo_counts[combo] += 1

    print("\nDirector movie counts:")
    for director, count in director_counts.most_common():
        print(f"{director}: {count}")

    print("\nDirector total revenue:")
    for director, revenue in sorted(director_revenue.items(), key=lambda item: item[1], reverse=True):
        print(f"{director}: {revenue:.0f}")

    print("\nTop 5 director collaborations:")
    for combo, count in director_combo_counts.most_common(5):
        print(f"{' | '.join(combo)}: {count}")

    return director_counts


def add_total_directed(df, director_counts):
    def calculate_total_directed(value):
        directors = [name for name in str(value).split("|") if name]
        if not directors:
            return 0
        return sum(director_counts[name] for name in directors) / len(directors)

    df["total_directed"] = df["director"].apply(calculate_total_directed)


def print_missing_summary(df, title):
    print(f"\n{title}")
    for column in ["budget", "revenue", "runtime"]:
        print(
            f"{column}: missing={int(df[column].isna().sum())}, "
            f"zero_or_negative={int((df[column] <= 0).sum())}"
        )
    empty_genres = int((df["genres_clean"].isna() | (df["genres_clean"].str.strip() == "")).sum())
    print(f"genres: empty={empty_genres}")


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Cannot find input file: {INPUT_FILE}")

    stats = Counter()
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {INPUT_FILE}: {df.shape[0]} rows, {df.shape[1]} columns")

    for column in ["budget", "revenue", "runtime"]:
        df[column] = pd.to_numeric(df[column], errors="coerce").astype(float)

    df["release_year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year
    median_year = int(df["release_year"].dropna().median())
    df["release_year"] = df["release_year"].fillna(median_year).astype(int)

    df["genres_clean"] = df["genres"].apply(format_genres)
    df["main_genre"] = df["genres_clean"].apply(first_genre)
    df["director"] = df["crew"].apply(extract_directors)

    print_missing_summary(df, "Before filling")
    apply_tmdb_fill(df, stats)
    df["main_genre"] = df["genres_clean"].apply(first_genre)
    fill_genres_by_groups(df, stats)
    df["main_genre"] = df["genres_clean"].apply(first_genre)
    for column in ["budget", "revenue", "runtime"]:
        fill_numeric_by_groups(df, column, stats)
    print_missing_summary(df, "After filling")

    print("\nFill statistics:")
    for key in sorted(stats):
        print(f"{key}: {stats[key]}")

    add_profit_level(df)
    genre_columns = add_genre_one_hot(df)
    director_counts = print_director_statistics(df)
    add_total_directed(df, director_counts)

    df["genres"] = df["genres_clean"]
    drop_columns = [
        "homepage",
        "tagline",
        "overview",
        "keywords",
        "original_title",
        "production_companies",
        "production_countries",
        "spoken_languages",
        "cast",
        "id",
        "movie_id",
        "release_date",
        "main_genre",
        "genres_clean",
        "director",
        "genres",
        "crew",
        "revenue",
        "title",
    ]

    print("\nDropped columns and reasons:")
    for column in drop_columns:
        if column in DROP_REASON:
            print(f"{column}: {DROP_REASON[column]}")
        elif column in ["director", "genres", "crew", "revenue", "title"]:
            print(f"{column}: required by experiment after derived features are created")
        else:
            print(f"{column}: temporary helper column")

    df = df.drop(columns=[column for column in drop_columns if column in df.columns])
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print(f"\nGenre one-hot columns ({len(genre_columns)}):")
    print(", ".join(genre_columns))
    print(f"\nSaved processed data to {OUTPUT_FILE}: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\nFinal feature columns:")
    for column in df.columns:
        print(column)


if __name__ == "__main__":
    main()
