"""Access to the source Parquet files, on the Hugging Face Hub or in a local folder."""
from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq


class Source:
    def __init__(self, repo_id: str | None = None, revision: str = "main", local_dir: str | None = None):
        self.local_dir = Path(local_dir) if local_dir else None
        self.repo_id = repo_id
        self.revision = revision
        self.sha = None
        if self.local_dir is None:
            from huggingface_hub import HfApi, HfFileSystem

            self.sha = HfApi().dataset_info(repo_id, revision=revision).sha
            self.fs = HfFileSystem()

    def files(self) -> list[str]:
        if self.local_dir is not None:
            return sorted(str(p.relative_to(self.local_dir)) for p in self.local_dir.rglob("*.parquet"))
        base = f"datasets/{self.repo_id}@{self.sha}"
        found = self.fs.glob(f"{base}/**/*.parquet")
        return sorted(p[len(base) + 1:] for p in found)

    def parquet(self, rel_path: str) -> pq.ParquetFile:
        if self.local_dir is not None:
            return pq.ParquetFile(self.local_dir / rel_path)
        return pq.ParquetFile(self.fs.open(f"datasets/{self.repo_id}@{self.sha}/{rel_path}", "rb", block_size=8 * 2**20))

    def describe(self) -> dict:
        if self.local_dir is not None:
            return {"local_dir": str(self.local_dir)}
        return {"repo_id": self.repo_id, "requested_revision": self.revision, "resolved_sha": self.sha,
                "url": f"https://huggingface.co/datasets/{self.repo_id}/tree/{self.sha}"}
