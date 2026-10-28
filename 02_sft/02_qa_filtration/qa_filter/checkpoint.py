import json
import os
import time


class CheckpointStore:
    def __init__(self, run_state_dir: str):
        self._path = os.path.join(run_state_dir, "checkpoint.json")
        os.makedirs(run_state_dir, exist_ok=True)
        if os.path.exists(self._path):
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._completed = set(data["completed_shards"])
            self._timestamps = {int(k): v for k, v in data.get("shard_timestamps", {}).items()}
            self._total_shards = data.get("total_shards")
        else:
            self._completed = set()
            self._timestamps = {}
            self._total_shards = None

    def _persist(self) -> None:
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump({
                "completed_shards": sorted(self._completed),
                "shard_timestamps": {str(k): v for k, v in self._timestamps.items()},
                "total_shards": self._total_shards,
            }, f)

    def is_complete(self, shard_id: int) -> bool:
        return shard_id in self._completed

    def mark_complete(self, shard_id: int) -> None:
        self._completed.add(shard_id)
        self._timestamps[shard_id] = time.time()
        self._persist()

    def completed_shard_ids(self) -> list[int]:
        return sorted(self._completed)

    def shard_timestamps(self) -> dict[int, float]:
        return dict(self._timestamps)

    def set_total_shards(self, n: int) -> None:
        self._total_shards = n
        self._persist()

    def get_total_shards(self):
        return self._total_shards
