from concurrent.futures import ThreadPoolExecutor, as_completed

from qa_filter.config import Settings


def run_batches_concurrently(
    batches: list[list[dict]],
    settings: Settings,
    judge_fn,
) -> tuple[list[dict], list[dict]]:
    all_verdicts: list[dict] = []
    all_failed: list[dict] = []

    with ThreadPoolExecutor(max_workers=settings.max_workers) as executor:
        futures = [executor.submit(judge_fn, batch, settings) for batch in batches]
        for future in as_completed(futures):
            verdicts, failed = future.result()
            all_verdicts.extend(verdicts)
            all_failed.extend(failed)

    return all_verdicts, all_failed
