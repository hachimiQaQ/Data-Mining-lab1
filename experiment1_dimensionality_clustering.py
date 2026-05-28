from __future__ import annotations

import math
import os
import warnings
from collections import Counter
from pathlib import Path

# joblib 在某些 Windows 环境下可能无法正确识别物理 CPU 核心数，
# 这里给一个保守默认值，避免运行 Isomap / scikit-learn 时打印无关告警。
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import matplotlib

# 使用 Agg 后端可以在没有图形界面的环境中保存 PNG 图片。
# 本实验只需要输出图表文件，不需要弹出交互式窗口。
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Isomap 在邻接图不完全连通时会自动补全图；这不影响本实验输出，
# 但会产生较长的库告警。这里屏蔽掉这些不影响结果解释的告警。
warnings.filterwarnings(
    "ignore",
    message="Changing the sparsity structure of a csr_matrix is expensive.*",
)
warnings.filterwarnings(
    "ignore",
    message="The number of connected components of the neighbors graph.*",
)

try:
    # 本实验后半部分需要用到 scikit-learn：
    # DBSCAN 用于聚类，PCA/Isomap 用于降维，StandardScaler 用于标准化。
    from sklearn.cluster import DBSCAN
    from sklearn.decomposition import PCA
    from sklearn.manifold import Isomap
    from sklearn.preprocessing import StandardScaler
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: scikit-learn. Install it with "
        "`python -m pip install scikit-learn` and run this script again."
    ) from exc


INPUT_FILE = Path("movies_basic_processed.csv")
OUTPUT_DIR = Path("experiment1_outputs")
SUMMARY_CSV = OUTPUT_DIR / "cluster_profit_level_summary.csv"
SUMMARY_TXT = OUTPUT_DIR / "dimensionality_clustering_summary.txt"

# 三个盈利等级在所有图中保持同一套颜色，便于报告中横向比较。
PROFIT_COLORS = {1: "#d95f02", 2: "#1b9e77", 3: "#7570b3"}
PROFIT_LABELS = {
    1: "profit_level 1: loss",
    2: "profit_level 2: low profit",
    3: "profit_level 3: high profit",
}


def load_data() -> pd.DataFrame:
    """读取基础预处理后的数据，并做最基本的输入合法性检查。

    movies_basic_processed.csv 是上一阶段脚本生成的结果，已经完成缺失值填充、
    genre one-hot、profit_level、total_directed 等基础处理。本脚本只负责
    “降维与聚类分析”，因此不再回头读取原始 movies.csv。
    """
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Cannot find input file: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)
    # profit_level 是实验要求中的盈利水平标签：
    # 后面用它给散点图着色、统计聚类内分布，但不能放进模型特征。
    if "profit_level" not in df.columns:
        raise ValueError("Input data must contain profit_level.")
    if df.empty:
        raise ValueError("Input data is empty.")
    return df


def build_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    """把当前数据集转换为可用于降维/聚类的标准化数值矩阵。

    处理流程：
    1. 删除 profit_level，避免把“答案标签”泄漏进降维和聚类；
    2. 找出字符串类型字段，例如 original_language、status；
    3. 对字符串字段做 one-hot 编码，因为它们是无序类别变量；
    4. 检查编码后是否仍有缺失值；
    5. 用 StandardScaler 标准化，使预算、票数、年份、0/1 特征处在可比较尺度。
    """
    # profit_level 是后续评价聚类效果的标签，不属于输入特征。
    features = df.drop(columns=["profit_level"]).copy()

    # original_language 不能用 1、2、3 这种整数编码，否则会制造
    # “英语 > 法语 > 西班牙语”之类不存在的顺序关系，所以使用 one-hot。
    # status 也是无序类别，并且取值很少，同样适合 one-hot。
    text_columns = features.select_dtypes(include=["object"]).columns.tolist()
    encoded = pd.get_dummies(features, columns=text_columns, prefix=text_columns, dummy_na=False)

    # get_dummies 后理论上应该全是数值；这里再强制转换和检查，方便发现异常输入。
    encoded = encoded.apply(pd.to_numeric, errors="coerce")
    if encoded.isna().any().any():
        missing_columns = encoded.columns[encoded.isna().any()].tolist()
        raise ValueError(f"Encoded features contain missing values: {missing_columns}")

    # 标准化非常关键：如果不做标准化，budget、vote_count 这类数值范围大的特征
    # 会在 PCA / Isomap / DBSCAN 的距离计算中占据过高权重。
    scaler = StandardScaler()
    scaled = scaler.fit_transform(encoded)
    return scaled, encoded, text_columns


def plot_profit_scatter(points: np.ndarray, profit_level: pd.Series, title: str, output_path: Path) -> None:
    """绘制二维降维结果，并用 profit_level 区分颜色。

    points 是 PCA 或 Isomap 生成的二维坐标；profit_level 只用于上色，
    这样可以观察降维空间是否自然呈现出盈利水平分组。
    """
    fig, ax = plt.subplots(figsize=(8, 6), dpi=160)
    for level in sorted(profit_level.unique()):
        mask = profit_level == level
        ax.scatter(
            points[mask, 0],
            points[mask, 1],
            s=14,
            alpha=0.72,
            c=PROFIT_COLORS.get(int(level), "#666666"),
            label=PROFIT_LABELS.get(int(level), f"profit_level {level}"),
            edgecolors="none",
        )
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.grid(True, linewidth=0.4, alpha=0.28)
    ax.legend(frameon=True, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def cluster_count(labels: np.ndarray) -> int:
    """统计 DBSCAN 生成的非噪声簇数量。

    scikit-learn 的 DBSCAN 使用 -1 表示噪声点，实验要求的聚类数不应把 -1 算作簇。
    """
    return len(set(labels) - {-1})


def noise_count(labels: np.ndarray) -> int:
    """统计 DBSCAN 中被标记为噪声的样本数量。"""
    return int(np.sum(labels == -1))


def parameter_score(labels: np.ndarray) -> tuple[int, int, int]:
    """给不满足 3 个簇要求的 DBSCAN 结果打分，作为兜底选择。

    优先级：
    1. 产生的簇越多越好；
    2. 噪声点越少越好；
    3. 最大簇不要过大，避免所有样本几乎挤在一个簇里。
    """
    clusters = cluster_count(labels)
    noise = noise_count(labels)
    largest_cluster = 0
    for label in set(labels) - {-1}:
        largest_cluster = max(largest_cluster, int(np.sum(labels == label)))
    return clusters, -noise, -largest_cluster


def search_dbscan(points: np.ndarray) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    """在二维降维结果上自动搜索 DBSCAN 参数。

    DBSCAN 主要参数：
    - eps：邻域半径，越大越容易把点合并到同一簇；
    - min_samples：形成核心点所需的最少邻居数，越大越严格。

    实验要求“聚类数目不小于 3”。这里不是手写固定参数，而是根据二维点云
    的 90% 分位距离构造 eps 搜索范围，自动选出满足要求且图表更易读的结果。
    """
    mins = [4, 5, 8, 10, 15, 20]
    # 用每个点到二维空间中心的距离估计点云尺度，再按比例生成 eps 候选值。
    # 这样 PCA 和 Isomap 的坐标尺度不同，也可以各自找到合适参数。
    distances = np.linalg.norm(points - np.median(points, axis=0), axis=1)
    spread = float(np.percentile(distances, 90))
    if spread <= 0:
        spread = 1.0

    # 搜索较小到中等的 eps。小 eps 容易产生更多小簇和噪声，
    # 大 eps 容易合并为少数大簇；两段 linspace 覆盖这两种情况。
    eps_values = np.unique(
        np.round(
            np.concatenate(
                [
                    np.linspace(spread * 0.015, spread * 0.20, 34),
                    np.linspace(spread * 0.22, spread * 0.60, 16),
                ]
            ),
            4,
        )
    )

    valid_labels: np.ndarray | None = None
    valid_meta: dict[str, float | int | bool] | None = None
    valid_score: tuple[int, int, int, int] | None = None
    fallback_labels: np.ndarray | None = None
    fallback_meta: dict[str, float | int | bool] | None = None
    fallback_score: tuple[int, int, int] | None = None

    for min_samples in mins:
        for eps in eps_values:
            labels = DBSCAN(eps=float(eps), min_samples=min_samples).fit_predict(points)
            clusters = cluster_count(labels)
            noise = noise_count(labels)
            meta = {
                "eps": float(eps),
                "min_samples": min_samples,
                "cluster_count": clusters,
                "noise_count": noise,
                "meets_requirement": clusters >= 3,
            }
            if clusters >= 3:
                # 已满足实验要求时，优先选择 3-12 个簇之间、接近 6 个簇、
                # 噪声点较少的结果。这样图表不会因为簇太多而难以阅读。
                too_many_clusters_penalty = max(0, clusters - 12)
                readable_target_distance = abs(clusters - 6)
                score = (too_many_clusters_penalty, readable_target_distance, noise, -clusters)
                if valid_score is None or score < valid_score:
                    valid_score = score
                    valid_labels = labels
                    valid_meta = meta
            else:
                # 如果所有参数都不满足 3 个簇，就记录一个最接近要求的结果，
                # 并在 summary 文本中通过 meets_requirement=False 标明。
                score = parameter_score(labels)
                if fallback_score is None or score > fallback_score:
                    fallback_score = score
                    fallback_labels = labels
                    fallback_meta = meta

    if valid_labels is not None and valid_meta is not None:
        return valid_labels, valid_meta

    if fallback_labels is not None and fallback_meta is not None:
        return fallback_labels, fallback_meta

    raise RuntimeError("DBSCAN parameter search did not produce any result.")


def plot_cluster_scatter(points: np.ndarray, labels: np.ndarray, title: str, output_path: Path) -> None:
    """绘制 DBSCAN 聚类结果散点图。

    普通簇使用不同颜色圆点，噪声点使用灰色 x。这样报告中可以清楚区分
    “被归入某个簇的电影”和“DBSCAN 认为不属于任何高密度区域的电影”。
    """
    fig, ax = plt.subplots(figsize=(8, 6), dpi=160)
    unique_labels = sorted(set(labels))
    cmap = plt.get_cmap("tab20")
    for index, label in enumerate(unique_labels):
        mask = labels == label
        if label == -1:
            # DBSCAN 约定 -1 是噪声，不是一个正常簇。
            ax.scatter(
                points[mask, 0],
                points[mask, 1],
                s=18,
                alpha=0.75,
                c="#9e9e9e",
                marker="x",
                label="noise",
            )
        else:
            ax.scatter(
                points[mask, 0],
                points[mask, 1],
                s=14,
                alpha=0.75,
                c=[cmap(index % 20)],
                marker="o",
                label=f"cluster {label}",
                edgecolors="none",
            )
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.grid(True, linewidth=0.4, alpha=0.28)
    ax.legend(frameon=True, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def entropy_from_counts(counts: list[int]) -> float:
    """根据信息熵衡量一个簇内部 profit_level 的混杂程度。

    熵越低，说明该簇越集中于某一个盈利等级；
    熵越高，说明三个盈利等级混在一起，聚类并没有自然对应盈利水平。
    """
    total = sum(counts)
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counts:
        if count == 0:
            continue
        probability = count / total
        entropy -= probability * math.log2(probability)
    return entropy


def summarize_clusters(method: str, labels: np.ndarray, profit_level: pd.Series) -> pd.DataFrame:
    """统计每个聚类内部 profit_level 的数量、比例和信息熵。"""
    rows = []
    for label in sorted(set(labels)):
        mask = labels == label
        total = int(mask.sum())
        # Counter 统计当前簇内 profit_level=1/2/3 各有多少部电影。
        level_counts = Counter(profit_level[mask].astype(int))
        counts = [level_counts.get(level, 0) for level in [1, 2, 3]]
        row = {
            "method": method,
            "cluster": int(label),
            "cluster_name": "noise" if label == -1 else f"cluster_{label}",
            "total": total,
            "profit_level_1_count": counts[0],
            "profit_level_2_count": counts[1],
            "profit_level_3_count": counts[2],
            "profit_level_1_ratio": counts[0] / total if total else 0.0,
            "profit_level_2_ratio": counts[1] / total if total else 0.0,
            "profit_level_3_ratio": counts[2] / total if total else 0.0,
            "profit_level_entropy": entropy_from_counts(counts),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def plot_profit_distribution(summary: pd.DataFrame, method: str, title: str, output_path: Path) -> None:
    """绘制每个 DBSCAN 簇中 profit_level 1/2/3 的分布直方图。"""
    ordered = summary[summary["method"] == method].copy()
    ordered = ordered.sort_values("cluster")
    labels = ordered["cluster_name"].tolist()
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(9, len(labels) * 0.65), 6), dpi=160)
    for offset, level in zip([-width, 0, width], [1, 2, 3]):
        ax.bar(
            x + offset,
            ordered[f"profit_level_{level}_count"],
            width=width,
            label=f"profit_level {level}",
            color=PROFIT_COLORS[level],
        )
    ax.set_title(title)
    ax.set_xlabel("DBSCAN cluster")
    ax.set_ylabel("Movie count")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.28)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def format_dbscan_meta(method: str, meta: dict[str, float | int | bool]) -> str:
    """把 DBSCAN 参数和结果格式化为一行文本，便于控制台和 summary 输出。"""
    requirement = "yes" if meta["meets_requirement"] else "no"
    return (
        f"{method}: eps={meta['eps']}, min_samples={meta['min_samples']}, "
        f"clusters={meta['cluster_count']}, noise={meta['noise_count']}, "
        f"cluster_count>=3={requirement}"
    )


def write_text_summary(
    df: pd.DataFrame,
    encoded: pd.DataFrame,
    text_columns: list[str],
    pca_meta: dict[str, float | int | bool],
    isomap_meta: dict[str, float | int | bool],
    summary: pd.DataFrame,
) -> None:
    """写出文本摘要，报告中可以直接引用这里的编码理由和聚类统计。"""
    lines = [
        "Experiment 1 dimensionality reduction and clustering summary",
        "=" * 62,
        f"Input file: {INPUT_FILE}",
        f"Rows: {len(df)}",
        f"Original columns: {len(df.columns)}",
        f"Encoded feature columns: {len(encoded.columns)}",
        "Excluded from feature matrix: profit_level",
        f"Encoded text columns: {', '.join(text_columns) if text_columns else 'none'}",
        "",
        "Encoding and scaling",
        "-" * 20,
        "original_language uses one-hot encoding because language is a nominal category.",
        "Integer/ordinal encoding would add a false order between languages.",
        "status uses one-hot encoding for the same reason and has very few categories.",
        "All encoded features are standardized with StandardScaler before PCA/Isomap.",
        "",
        "DBSCAN parameters",
        "-" * 20,
        format_dbscan_meta("PCA", pca_meta),
        format_dbscan_meta("Isomap", isomap_meta),
        "",
        "Profit-level distribution by cluster",
        "-" * 36,
        summary.to_string(index=False),
        "",
        "Interpretation note",
        "-" * 19,
        "Low entropy means a cluster is dominated by fewer profit levels.",
        "High entropy means profit levels are mixed, so the cluster does not naturally",
        "match the profit_level label very well.",
    ]
    SUMMARY_TXT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """主流程：读取数据、编码标准化、降维、聚类、画图、保存统计结果。"""
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = load_data()
    # profit_level 不参与训练，只在画图和统计时作为标签使用。
    profit_level = df["profit_level"].astype(int)

    scaled, encoded, text_columns = build_feature_matrix(df)
    print(f"Loaded {INPUT_FILE}: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Encoded feature matrix: {scaled.shape[0]} rows, {scaled.shape[1]} columns")
    print("profit_level excluded from feature matrix")

    # PCA 是线性降维方法，会寻找方差最大的两个方向。
    pca = PCA(n_components=2, random_state=42)
    pca_points = pca.fit_transform(scaled)
    print(f"PCA output shape: {pca_points.shape}")
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")

    # Isomap 是非线性流形学习方法，通过近邻图近似样本间的测地距离。
    isomap = Isomap(n_components=2, n_neighbors=10)
    isomap_points = isomap.fit_transform(scaled)
    print(f"Isomap output shape: {isomap_points.shape}")

    # 第一组图：降维后按真实 profit_level 着色，观察盈利等级在二维空间中的分布。
    plot_profit_scatter(
        pca_points,
        profit_level,
        "PCA projection colored by profit_level",
        OUTPUT_DIR / "pca_profit_level_scatter.png",
    )
    plot_profit_scatter(
        isomap_points,
        profit_level,
        "Isomap projection colored by profit_level",
        OUTPUT_DIR / "isomap_profit_level_scatter.png",
    )

    # 在“上一步生成的二维数据集”上做 DBSCAN，严格对应实验要求。
    pca_labels, pca_meta = search_dbscan(pca_points)
    isomap_labels, isomap_meta = search_dbscan(isomap_points)

    # 第二组图：降维后按 DBSCAN 聚类标签着色，观察算法找到的密度簇。
    plot_cluster_scatter(
        pca_points,
        pca_labels,
        "DBSCAN clusters on PCA projection",
        OUTPUT_DIR / "pca_dbscan_clusters.png",
    )
    plot_cluster_scatter(
        isomap_points,
        isomap_labels,
        "DBSCAN clusters on Isomap projection",
        OUTPUT_DIR / "isomap_dbscan_clusters.png",
    )

    # 统计每个聚类内部的盈利等级分布，用来讨论聚类是否自然发现盈利分组。
    pca_summary = summarize_clusters("PCA", pca_labels, profit_level)
    isomap_summary = summarize_clusters("Isomap", isomap_labels, profit_level)
    summary = pd.concat([pca_summary, isomap_summary], ignore_index=True)
    summary.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")

    # 第三组图：每个簇内 profit_level 1/2/3 的数量直方图。
    plot_profit_distribution(
        summary,
        "PCA",
        "profit_level distribution in PCA-DBSCAN clusters",
        OUTPUT_DIR / "pca_cluster_profit_distribution.png",
    )
    plot_profit_distribution(
        summary,
        "Isomap",
        "profit_level distribution in Isomap-DBSCAN clusters",
        OUTPUT_DIR / "isomap_cluster_profit_distribution.png",
    )

    # 文本摘要集中记录参数、编码说明、信息熵等报告素材。
    write_text_summary(df, encoded, text_columns, pca_meta, isomap_meta, summary)

    print(format_dbscan_meta("PCA", pca_meta))
    print(format_dbscan_meta("Isomap", isomap_meta))
    print(f"Saved outputs to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
