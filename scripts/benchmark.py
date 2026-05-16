"""Benchmark end-to-end RAG latency with fine-grained stage timing.

Measures each stage of the RAG pipeline independently:
    query_rewrite (Chinese only) -> clip_encode -> faiss_search -> sqlite_lookup -> llm_generate

Runs N queries (warmup + measured), collects per-stage latencies,
reports p50 / p95 / p99.
"""

from __future__ import annotations

import argparse
import logging
import statistics
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

# Ensure project root on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.models.schemas import RetrievalResult  # noqa: E402
from app.services.encoder import CLIPEncoder  # noqa: E402
from app.services.generator import LLMGenerator, _format_sources  # noqa: E402
from app.services.indexer import FAISSIndexer  # noqa: E402
from app.services.metadata import MetadataStore  # noqa: E402


logging.basicConfig(
    level=logging.WARNING,  # silence info logs during benchmark
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ---------- Test queries ----------

CHINESE_QUERIES = [
    "一只在草地上奔跑的狗",
    "海边的日落",
    "穿红色衣服的小女孩",
    "街道上的人群",
    "山顶的雪景",
    "两个人在海滩上玩耍",
    "厨房里的一位女士",
    "骑自行车的男子",
    "公园里散步的一家人",
    "儿童在操场上玩耍",
]

ENGLISH_QUERIES = [
    "dog running on grass",
    "sunset at beach",
    "children playing in the park",
    "man riding a bicycle",
    "people walking on a busy street",
]


# ---------- Timing helpers ----------

@dataclass
class StageTiming:
    """Timings (in milliseconds) for a single query."""

    query: str
    is_chinese: bool
    query_rewrite_ms: float = 0.0  # 0 if English (skipped)
    clip_encode_ms: float = 0.0
    faiss_search_ms: float = 0.0
    sqlite_lookup_ms: float = 0.0
    llm_generate_ms: float = 0.0
    total_ms: float = 0.0
    error: str | None = None


@contextmanager
def measure(name: str, target: dict):
    """Context manager: time a block and store in target[name]."""
    t0 = time.perf_counter()
    yield
    target[name] = (time.perf_counter() - t0) * 1000.0


def contains_chinese(text: str) -> bool:
    return any("\u4e00" <= c <= "\u9fff" for c in text)


# ---------- Single query execution ----------

def run_one_query(
    query: str,
    top_k: int,
    encoder: CLIPEncoder,
    indexer: FAISSIndexer,
    metadata: MetadataStore,
    generator: LLMGenerator,
) -> StageTiming:
    """Execute one RAG query, capturing per-stage timings."""

    is_cn = contains_chinese(query)
    t = StageTiming(query=query, is_chinese=is_cn)
    stages: dict[str, float] = {}
    wall_start = time.perf_counter()

    try:
        # Stage 1: Query rewrite (Chinese only)
        search_query = query
        if is_cn:
            with measure("query_rewrite", stages):
                search_query = generator.rewrite_query(query)

        # Stage 2: CLIP text encode
        with measure("clip_encode", stages):
            query_emb = encoder.encode_texts([search_query])

        # Stage 3: FAISS search
        with measure("faiss_search", stages):
            scores, indices = indexer.search_images(query_emb, top_k)

        # Stage 4: SQLite metadata lookup
        idx_list = indices[0].tolist()
        score_list = scores[0].tolist()
        valid_idx = [i for i in idx_list if i >= 0]
        with measure("sqlite_lookup", stages):
            records = metadata.get_images_by_indices(valid_idx)

        # Assemble sources for the LLM
        idx_to_record = {idx: rec for idx, rec in zip(valid_idx, records)}
        sources: list[RetrievalResult] = []
        for faiss_idx, score in zip(idx_list, score_list):
            if faiss_idx < 0 or faiss_idx not in idx_to_record:
                continue
            rec = idx_to_record[faiss_idx]
            sources.append(
                RetrievalResult(
                    image_id=rec.image_id,
                    filename=rec.filename,
                    filepath=rec.filepath,
                    score=float(score),
                    captions=rec.captions,
                )
            )

        # Stage 5: LLM generate
        with measure("llm_generate", stages):
            _ = generator.generate(query, sources)

        t.query_rewrite_ms = stages.get("query_rewrite", 0.0)
        t.clip_encode_ms = stages.get("clip_encode", 0.0)
        t.faiss_search_ms = stages.get("faiss_search", 0.0)
        t.sqlite_lookup_ms = stages.get("sqlite_lookup", 0.0)
        t.llm_generate_ms = stages.get("llm_generate", 0.0)
        t.total_ms = (time.perf_counter() - wall_start) * 1000.0

    except Exception as exc:  # noqa: BLE001
        t.error = f"{type(exc).__name__}: {exc}"
        logger.warning("Query failed: %s | %s", query, t.error)

    return t


# ---------- Stats reporting ----------

def percentile(sorted_values: list[float], p: float) -> float:
    """Nearest-rank percentile (p in [0, 100])."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (p / 100.0) * (len(sorted_values) - 1)
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return sorted_values[f]
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


@dataclass
class StageStats:
    name: str
    n: int
    mean: float
    p50: float
    p95: float
    p99: float
    min_v: float = field(default=0.0)
    max_v: float = field(default=0.0)


def compute_stats(values: list[float], name: str) -> StageStats:
    values = [v for v in values if v > 0]  # filter zeros (skipped rewrite)
    if not values:
        return StageStats(name=name, n=0, mean=0, p50=0, p95=0, p99=0)
    s = sorted(values)
    return StageStats(
        name=name,
        n=len(s),
        mean=statistics.mean(s),
        p50=percentile(s, 50),
        p95=percentile(s, 95),
        p99=percentile(s, 99),
        min_v=s[0],
        max_v=s[-1],
    )


def print_report(timings: list[StageTiming], warmup: int) -> None:
    measured = timings[warmup:]
    ok = [t for t in measured if t.error is None]
    failed = [t for t in measured if t.error is not None]

    print("\n" + "=" * 78)
    print(f"  Benchmark Report  (runs={len(timings)}, warmup={warmup}, "
          f"measured={len(measured)}, ok={len(ok)}, failed={len(failed)})")
    print("=" * 78)

    if not ok:
        print("\n  No successful queries to report on.")
        if failed:
            print("  Sample error:", failed[0].error)
        return

    stages = [
        ("Query Rewrite", [t.query_rewrite_ms for t in ok if t.is_chinese]),
        ("CLIP Encode",   [t.clip_encode_ms   for t in ok]),
        ("FAISS Search",  [t.faiss_search_ms  for t in ok]),
        ("SQLite Lookup", [t.sqlite_lookup_ms for t in ok]),
        ("LLM Generate",  [t.llm_generate_ms  for t in ok]),
        ("TOTAL (E2E)",   [t.total_ms         for t in ok]),
    ]

    print(f"\n  {'Stage':<18} {'N':>4}  {'mean':>9}  {'p50':>9}  {'p95':>9}  {'p99':>9}  {'min':>9}  {'max':>9}")
    print(f"  {'-' * 18} {'-' * 4}  {'-' * 9}  {'-' * 9}  {'-' * 9}  {'-' * 9}  {'-' * 9}  {'-' * 9}")
    for name, vals in stages:
        s = compute_stats(vals, name)
        print(f"  {name:<18} {s.n:>4}  {s.mean:>8.1f}ms  {s.p50:>8.1f}ms  "
              f"{s.p95:>8.1f}ms  {s.p99:>8.1f}ms  {s.min_v:>8.1f}ms  {s.max_v:>8.1f}ms")

    # Breakdown by language
    cn = [t for t in ok if t.is_chinese]
    en = [t for t in ok if not t.is_chinese]
    if cn and en:
        print("\n  Split by language:")
        cn_total = compute_stats([t.total_ms for t in cn], "  Chinese (full)")
        en_total = compute_stats([t.total_ms for t in en], "  English (no rewrite)")
        print(f"    Chinese  total p50: {cn_total.p50:>7.1f}ms  p95: {cn_total.p95:>7.1f}ms  (n={cn_total.n})")
        print(f"    English  total p50: {en_total.p50:>7.1f}ms  p95: {en_total.p95:>7.1f}ms  (n={en_total.n})")
        print(f"    Rewrite alone saves ~{cn_total.p50 - en_total.p50:.0f}ms")

    # Warmup comparison (first query vs later ones)
    if warmup > 0 and timings[0].error is None:
        first = timings[0]
        print(f"\n  Cold-start (first query) vs warm:")
        print(f"    first  clip_encode: {first.clip_encode_ms:>7.1f}ms")
        s = compute_stats([t.clip_encode_ms for t in ok], "warm")
        print(f"    warm   clip_encode p50: {s.p50:>7.1f}ms  (speedup: {first.clip_encode_ms / s.p50:.1f}x)")

    if failed:
        print(f"\n  Failed queries ({len(failed)}):")
        for t in failed[:5]:
            print(f"    - {t.query!r}: {t.error}")


def save_markdown_report(
    timings: list[StageTiming],
    warmup: int,
    output_path: Path,
) -> None:
    measured = timings[warmup:]
    ok = [t for t in measured if t.error is None]
    if not ok:
        logger.warning("No successful runs; skipping markdown report.")
        return

    stages = [
        ("Query Rewrite", [t.query_rewrite_ms for t in ok if t.is_chinese]),
        ("CLIP Encode",   [t.clip_encode_ms   for t in ok]),
        ("FAISS Search",  [t.faiss_search_ms  for t in ok]),
        ("SQLite Lookup", [t.sqlite_lookup_ms for t in ok]),
        ("LLM Generate",  [t.llm_generate_ms  for t in ok]),
        ("TOTAL (E2E)",   [t.total_ms         for t in ok]),
    ]

    lines = [
        "# 性能实测报告",
        "",
        f"- 测试日期: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 总查询数: {len(timings)} (预热 {warmup} + 测量 {len(measured)})",
        f"- 成功: {len(ok)}  失败: {len(measured) - len(ok)}",
        "",
        "## 各阶段耗时 (ms)",
        "",
        "| Stage | N | mean | p50 | p95 | p99 | min | max |",
        "|-------|---|------|-----|-----|-----|-----|-----|",
    ]
    for name, vals in stages:
        s = compute_stats(vals, name)
        lines.append(
            f"| {s.name} | {s.n} | {s.mean:.1f} | {s.p50:.1f} | "
            f"{s.p95:.1f} | {s.p99:.1f} | {s.min_v:.1f} | {s.max_v:.1f} |"
        )

    cn = [t for t in ok if t.is_chinese]
    en = [t for t in ok if not t.is_chinese]
    if cn and en:
        cn_p50 = compute_stats([t.total_ms for t in cn], "cn").p50
        en_p50 = compute_stats([t.total_ms for t in en], "en").p50
        lines += [
            "",
            "## 中英文对比",
            "",
            f"- 中文 query 总耗时 p50: **{cn_p50:.1f} ms**",
            f"- 英文 query 总耗时 p50: **{en_p50:.1f} ms**",
            f"- 跳过 Query Rewrite 可节省约 **{cn_p50 - en_p50:.0f} ms**",
        ]

    if warmup > 0 and timings[0].error is None:
        first = timings[0]
        warm_p50 = compute_stats([t.clip_encode_ms for t in ok], "warm").p50
        lines += [
            "",
            "## 冷启动 vs 稳态 (CLIP 编码)",
            "",
            f"- 首次请求: {first.clip_encode_ms:.1f} ms",
            f"- 稳态 p50: {warm_p50:.1f} ms",
            f"- 加速比: **{first.clip_encode_ms / warm_p50:.1f}x**",
        ]

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  Markdown report saved to: {output_path}")


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=5, help="top-k for retrieval")
    parser.add_argument("--n", type=int, default=15,
                        help="total queries to run (including warmup)")
    parser.add_argument("--warmup", type=int, default=3,
                        help="warmup queries excluded from stats")
    parser.add_argument("--english-only", action="store_true",
                        help="only run English queries (skip rewrite)")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "docs" / "performance-measurements.md",
        help="path to write markdown report",
    )
    args = parser.parse_args()

    print("Loading services...")
    t0 = time.perf_counter()
    encoder = CLIPEncoder()
    indexer = FAISSIndexer()
    indexer.load()
    metadata = MetadataStore()
    generator = LLMGenerator()
    print(f"  Services loaded in {(time.perf_counter() - t0) * 1000:.0f}ms")
    print(f"  Image index: {indexer.image_index_size} vectors")
    print(f"  Text  index: {indexer.text_index_size} vectors")
    print(f"  Metadata:    {metadata.count_images()} images, "
          f"{metadata.count_captions()} captions")
    print(f"  LLM model:   from settings (see .env)")
    print()

    # Build query list
    if args.english_only:
        pool = ENGLISH_QUERIES
    else:
        pool = (CHINESE_QUERIES + ENGLISH_QUERIES)
    queries = (pool * ((args.n // len(pool)) + 1))[: args.n]

    print(f"Running {args.n} queries (warmup={args.warmup}, top_k={args.top_k})...")
    timings: list[StageTiming] = []
    for i, q in enumerate(queries, 1):
        marker = "[warmup]" if i <= args.warmup else "[measure]"
        print(f"  {marker} {i}/{args.n} | {q!r}", end=" ", flush=True)
        t = run_one_query(q, args.top_k, encoder, indexer, metadata, generator)
        if t.error:
            print(f"ERROR ({t.error})")
        else:
            print(f"{t.total_ms:.0f}ms  "
                  f"(rewrite={t.query_rewrite_ms:.0f}, "
                  f"enc={t.clip_encode_ms:.0f}, "
                  f"faiss={t.faiss_search_ms:.1f}, "
                  f"sql={t.sqlite_lookup_ms:.1f}, "
                  f"gen={t.llm_generate_ms:.0f})")
        timings.append(t)

    print_report(timings, args.warmup)
    save_markdown_report(timings, args.warmup, args.output)


if __name__ == "__main__":
    main()
