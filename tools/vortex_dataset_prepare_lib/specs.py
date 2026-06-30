"""Dataset specifications and shared validation errors for Vortex datasets."""

from __future__ import annotations

from dataclasses import dataclass


DTYPE_BYTES = {
    "uint8": 1,
    "int8": 1,
    "float32": 4,
}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    display_name: str
    role: str
    source_family: str
    base_url: str
    query_url: str
    groundtruth_url: str
    base_dtype: str
    query_dtype: str
    dim: int
    base_count: int
    source_count: int
    query_count: int
    groundtruth_topk: int
    metric: str = "L2"
    slice_rule: str = "first_10000000_vectors"
    slice_based: bool = True
    base_split: str = "canonical_prefix_slice_from_official_1b_base"
    query_split: str = "official_public_query_set"
    groundtruth_scope: str = "official_exact_top100_for_evaluated_slice"
    groundtruth_exact: bool = True

    @property
    def dtype_bytes(self) -> int:
        return dtype_size(self.base_dtype)

    @property
    def prefix_bytes(self) -> int:
        return xbin_size(self.base_count, self.dim, self.base_dtype)

    @property
    def byte_range(self) -> str:
        return f"0-{self.prefix_bytes - 1}"


DATASETS: dict[str, DatasetSpec] = {
    "sift10m": DatasetSpec(
        name="sift10m",
        display_name="SIFT10M",
        role="10M descriptor stress test",
        source_family="bigann",
        base_url="https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/base.1B.u8bin",
        query_url="https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/query.public.10K.u8bin",
        groundtruth_url="https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/GT_10M/bigann-10M",
        base_dtype="uint8",
        query_dtype="uint8",
        dim=128,
        base_count=10_000_000,
        source_count=1_000_000_000,
        query_count=10_000,
        groundtruth_topk=100,
    ),
    "deep10m_l2": DatasetSpec(
        name="deep10m_l2",
        display_name="DEEP10M-L2",
        role="neural-embedding stress test",
        source_family="deep1b",
        base_url="https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/base.1B.fbin",
        query_url="https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/query.public.10K.fbin",
        groundtruth_url="https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/GT_10M/deep-10M",
        base_dtype="float32",
        query_dtype="float32",
        dim=96,
        base_count=10_000_000,
        source_count=1_000_000_000,
        query_count=10_000,
        groundtruth_topk=100,
    ),
    "spacev10m": DatasetSpec(
        name="spacev10m",
        display_name="SPACEV10M",
        role="production-style web-search embeddings",
        source_family="spacev1b",
        base_url="https://comp21storage.z5.web.core.windows.net/comp21/spacev1b/spacev1b_base.i8bin",
        query_url="https://comp21storage.z5.web.core.windows.net/comp21/spacev1b/query.i8bin",
        groundtruth_url="https://comp21storage.z5.web.core.windows.net/comp21/spacev1b/msspacev-gt-10M",
        base_dtype="int8",
        query_dtype="int8",
        dim=100,
        base_count=10_000_000,
        source_count=1_000_000_000,
        query_count=29_316,
        groundtruth_topk=100,
    ),
}


@dataclass(frozen=True)
class TexMexFileSpec:
    archive_member: str
    output_name: str
    count: int
    dim: int
    kind: str
    id_bound: int | None = None


@dataclass(frozen=True)
class TexMexSpec:
    name: str
    display_name: str
    role: str
    url: str
    archive_bytes: int
    md5: str
    files: tuple[TexMexFileSpec, ...]
    metric: str = "L2"
    slice_based: bool = False
    base_split: str = "official_texmex_base_file"
    query_split: str = "official_texmex_query_file"
    groundtruth_scope: str = "official_exact_top100_for_official_base"
    groundtruth_exact: bool = True


TEXMEX_DATASETS: dict[str, TexMexSpec] = {
    "siftsmall": TexMexSpec(
        name="siftsmall",
        display_name="SIFT-Small",
        role="test/smoke only",
        url="ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz",
        archive_bytes=5_305_734,
        md5="0b8324a7a82d7f2663d7dcbd57642df7",
        metric="L2",
        files=(
            TexMexFileSpec(
                "siftsmall/siftsmall_base.fvecs",
                "siftsmall_base.fvecs",
                10_000,
                128,
                "fvecs",
            ),
            TexMexFileSpec(
                "siftsmall/siftsmall_query.fvecs",
                "siftsmall_query.fvecs",
                100,
                128,
                "fvecs",
            ),
            TexMexFileSpec(
                "siftsmall/siftsmall_learn.fvecs",
                "siftsmall_learn.fvecs",
                25_000,
                128,
                "fvecs",
            ),
            TexMexFileSpec(
                "siftsmall/siftsmall_groundtruth.ivecs",
                "siftsmall_groundtruth.ivecs",
                100,
                100,
                "ivecs",
                id_bound=10_000,
            ),
        ),
    ),
    "sift": TexMexSpec(
        name="sift",
        display_name="SIFT1M",
        role="benchmark baseline",
        url="ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz",
        archive_bytes=168_280_445,
        md5="b23d1b3b2ee8469d819b61ca900ef0ed",
        metric="L2",
        files=(
            TexMexFileSpec(
                "sift/sift_base.fvecs", "sift_base.fvecs", 1_000_000, 128, "fvecs"
            ),
            TexMexFileSpec(
                "sift/sift_query.fvecs", "sift_query.fvecs", 10_000, 128, "fvecs"
            ),
            TexMexFileSpec(
                "sift/sift_learn.fvecs", "sift_learn.fvecs", 100_000, 128, "fvecs"
            ),
            TexMexFileSpec(
                "sift/sift_groundtruth.ivecs",
                "sift_groundtruth.ivecs",
                10_000,
                100,
                "ivecs",
                id_bound=1_000_000,
            ),
        ),
    ),
    "gist1m": TexMexSpec(
        name="gist1m",
        display_name="GIST1M",
        role="high-dimensional benchmark baseline",
        url="ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz",
        archive_bytes=2_740_172_684,
        md5="31185e0f00854f74d27e8ad8d52628a9",
        metric="L2",
        files=(
            TexMexFileSpec(
                "gist/gist_base.fvecs", "gist_base.fvecs", 1_000_000, 960, "fvecs"
            ),
            TexMexFileSpec(
                "gist/gist_query.fvecs", "gist_query.fvecs", 1_000, 960, "fvecs"
            ),
            TexMexFileSpec(
                "gist/gist_learn.fvecs", "gist_learn.fvecs", 500_000, 960, "fvecs"
            ),
            TexMexFileSpec(
                "gist/gist_groundtruth.ivecs",
                "gist_groundtruth.ivecs",
                1_000,
                100,
                "ivecs",
                id_bound=1_000_000,
            ),
        ),
    ),
}


class ValidationError(RuntimeError):
    """Raised when a dataset artifact is not benchmark-safe."""


def dtype_size(dtype: str) -> int:
    try:
        return DTYPE_BYTES[dtype]
    except KeyError as exc:
        raise ValidationError(f"unsupported dtype: {dtype}") from exc


def xbin_size(count: int, dim: int, dtype: str) -> int:
    return 8 + count * dim * dtype_size(dtype)
