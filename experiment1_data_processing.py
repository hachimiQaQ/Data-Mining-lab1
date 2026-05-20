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


# Windows 的默认控制台编码可能不是 UTF-8，直接打印外文导演名会报编码错误。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# TMDB API key：本实验按要求直接写入代码。公开仓库中不建议这样做。
TMDB_API_KEY = "626360df446c552d75af6ba81901736b"
INPUT_FILE = Path("movies.csv")
OUTPUT_FILE = Path("movies_basic_processed.csv")

# tmdb_cache.json 保存已经从 TMDB API 成功获取到的信息：
# {
#   "电影 id": {"budget": ..., "revenue": ..., "runtime": ..., "genres": "...|..."}
# }
# 这样再次运行脚本时不用重复请求 API，速度更快，也能减少接口调用次数。
CACHE_FILE = Path("tmdb_cache.json")

# tmdb_failed_records.csv 保存 TMDB API 没有成功获取到的少量记录。
# 这些记录后续会继续使用本脚本的统计方法进行填充，保证最终数据完整。
FAILED_TMDB_FILE = Path("tmdb_failed_records.csv")

# 使用数据集里的 id 访问 TMDB 电影详情接口。
# 例如 id=19995 时，请求 https://api.themoviedb.org/3/movie/19995。
TMDB_MOVIE_URL = "https://api.themoviedb.org/3/movie/{movie_id}"
MAX_TMDB_WORKERS = 16


# 这些字段要么缺失较多，要么是文本/嵌套结构，要么只是编号。
# 本实验后续要做降维、聚类等结构化数据分析，所以先删除它们并在日志中打印理由。
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
    """把 genres/crew 这类 JSON 风格字符串转换成 Python list。"""
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
    """从 [{"id": ..., "name": "..."}] 结构中提取所有 name。"""
    names = []
    for item in parse_json_list(value):
        if isinstance(item, dict):
            name = str(item.get("name", "")).strip()
            if name:
                names.append(name)
    return names


def format_genres(value):
    """把 genres 字段转换为实验要求的 Action|Adventure|... 形式。"""
    return "|".join(extract_names(value))


def extract_directors(value):
    """从 crew 字段中找出 job == Director 的人员姓名。"""
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
    """读取 TMDB 缓存；如果文件不存在或损坏，就返回空字典。"""
    if not CACHE_FILE.exists():
        return {}
    try:
        with CACHE_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def save_cache(cache):
    """把 TMDB 查询结果写入 tmdb_cache.json。"""
    with CACHE_FILE.open("w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def fetch_tmdb_movie_from_api(movie_id):
    """通过 TMDB 电影详情接口获取 budget/revenue/runtime/genres。

    返回格式为 (movie_id, data, error)。data 只保留本实验需要补全的字段：
    budget 电影预算，revenue 电影票房，runtime 电影时长，genres 电影类型。
    如果接口访问失败、id 不存在或返回无法解析，则 data 为空字典。
    """
    key = str(movie_id)
    url = TMDB_MOVIE_URL.format(movie_id=urllib.parse.quote(key))
    query = urllib.parse.urlencode({"api_key": TMDB_API_KEY, "language": "en-US"})
    request_url = f"{url}?{query}"
    try:
        with urllib.request.urlopen(request_url, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
        # TMDB 返回的类型是列表，这里也转换成实验要求的 A|B|C 格式。
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
    """本实验把缺失值和小于等于 0 的值都视为需要补全。"""
    return series.isna() | (series <= 0)


def first_genre(genres_text):
    """取第一个电影类型，作为统计回退填充时的分组依据。"""
    if pd.isna(genres_text) or not str(genres_text).strip():
        return "Unknown"
    return str(genres_text).split("|")[0] or "Unknown"


def fill_numeric_by_groups(df, column, stats):
    """使用统计方法填充 TMDB 仍然无法补全的数值字段。

    填充顺序：
    1. 优先使用「主类型 + 上映年份」分组的中位数；
    2. 如果该组没有可用值，再使用「主类型」分组中位数；
    3. 如果仍然没有可用值，最后使用全局中位数。

    使用中位数而不是平均数，是因为 budget/revenue 这类字段极易受大片影响，
    中位数对极端值更稳健。
    """
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
    """使用统计方法填充 TMDB 仍然无法补全的 genres。

    genres 是分类文本，不能用中位数。这里先使用同上映年份中最常见的 genres，
    如果对应年份没有可用类型，再使用全局最常见的 genres。
    """
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
    """优先使用 TMDB API 补全缺失或为 0 的字段。

    处理流程：
    1. 找出 budget/revenue/runtime 缺失或 <=0、genres 为空的电影；
    2. 读取 tmdb_cache.json，已经缓存过的电影不再请求 API；
    3. 对未缓存电影并发调用 TMDB API，提高运行速度；
    4. 成功获取的数据写入 tmdb_cache.json；
    5. API 失败的电影写入 tmdb_failed_records.csv；
    6. 把 API 返回的有效字段回填到原始 DataFrame。

    注意：TMDB 查询成功不代表每个字段都有值，例如有的电影详情页 budget 仍为 0。
    这种字段后面会继续交给 fill_numeric_by_groups 或 fill_genres_by_groups 处理。
    """
    cache = load_cache()

    # 只对真正需要补全的记录调用 API，避免无意义请求。
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
        # 并发请求 TMDB。每个线程只处理一个 movie_id，返回后统一写入缓存。
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
                    # 定期保存缓存，即使中途停止，下次运行也可以接着已有结果继续。
                    save_cache(cache)
                    print(f"TMDB progress: {completed}/{len(uncached_ids)}")
        if failed_ids:
            # 保存 API 获取失败的原始记录，报告中可用来说明“无法通过 API 获取的少量缺失值”。
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
            # 只有当前值缺失/无效，且 TMDB 返回的是正数时才回填。
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
    """根据盈利比例 profit_ratio 离散化得到 profit_level。"""
    profit_ratio = (df["revenue"] - df["budget"]) / df["budget"]
    df["profit_level"] = 1
    df.loc[(profit_ratio >= 0) & (profit_ratio < 0.6), "profit_level"] = 2
    df.loc[profit_ratio >= 0.6, "profit_level"] = 3


def add_genre_one_hot(df):
    """把 genres 转换成多个 0/1 类型特征列。"""
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
    """打印实验要求的导演统计信息。"""
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
    """新增 total_directed，多导演电影取各导演总执导数的平均值。"""
    def calculate_total_directed(value):
        directors = [name for name in str(value).split("|") if name]
        if not directors:
            return 0
        return sum(director_counts[name] for name in directors) / len(directors)

    df["total_directed"] = df["director"].apply(calculate_total_directed)


def print_missing_summary(df, title):
    """打印关键字段缺失情况，方便写入实验报告。"""
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

    # 统一转成数值类型，便于判断 0 值和计算 profit_level。
    for column in ["budget", "revenue", "runtime"]:
        df[column] = pd.to_numeric(df[column], errors="coerce").astype(float)

    # release_date 转换成 release_year，作为统计回退填充时的分组依据。
    df["release_year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year
    median_year = int(df["release_year"].dropna().median())
    df["release_year"] = df["release_year"].fillna(median_year).astype(int)

    # 先解析嵌套字段，生成后续处理需要的辅助列。
    df["genres_clean"] = df["genres"].apply(format_genres)
    df["main_genre"] = df["genres_clean"].apply(first_genre)
    df["director"] = df["crew"].apply(extract_directors)

    print_missing_summary(df, "Before filling")

    # 第一优先级：使用 TMDB API 补全。
    apply_tmdb_fill(df, stats)

    # 第二优先级：对 TMDB 没有补到的字段使用统计方法补全。
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

    # 为了满足实验要求，派生特征生成后删除 director、genres、crew、revenue、title。
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
