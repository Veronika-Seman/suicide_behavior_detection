from pathlib import Path
import json
from typing import List, Dict, Any, Iterable, Optional

# --- Project paths ---
# Define project root (two levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# Input files to merge
IN_FILES = [
    RAW_DIR / "subreddits_combined.json",
    RAW_DIR / "suicidewatch_new_posts_filtered.json",
]

# Output file where merged and deduplicated posts will be saved
OUT_FILE = RAW_DIR / "subreddits_combined2.json"


# --- Helpers ---
def load_as_list(p: Path) -> List[Dict[str, Any]]:
    """
    Load a JSON file and normalize it into a list of dictionaries.

    - Supports files saved as list[dict].
    - Supports files saved as dict[str, dict] (ID-to-object maps).
    - Filters out non-dict entries.
    """
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        # Some JSON dumps use a mapping: {id: object}
        return [v for v in data.values() if isinstance(v, dict)]
    raise ValueError(f"Unsupported JSON structure in {p}")


def choose_unique_key(items: Iterable[Dict[str, Any]]) -> Optional[str]:
    """
    Try to pick a reliable unique key (ID field) from the items.
    Checks common candidates such as: id, post_id, name, url.
    Returns the key name if found, otherwise None.
    """
    candidates = ("id", "post_id", "name", "url")
    for k in candidates:
        if any(k in it for it in items):
            return k
    return None


def dedup(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deduplicate a list of post dictionaries.
    - If a stable unique key exists (like 'id'), use it.
    - Otherwise, fall back to a lightweight identity hash
      built from (author, title, text, timestamp).
    """
    key = choose_unique_key(items)
    seen = set()
    result: List[Dict[str, Any]] = []

    def fallback_key(d: Dict[str, Any]) -> str:
        """Build a fallback identity when no single ID field is present."""
        author = str(d.get("author", "")).strip().lower()
        title = str(d.get("title", "")).strip()
        text = str(d.get("selftext", "") or d.get("text", "")).strip()
        ts = str(d.get("created_utc", "") or d.get("created", "")).strip()
        return "|".join([author, title, text, ts])

    for d in items:
        if key and d.get(key) not in (None, ""):
            k = f"{key}:{str(d.get(key))}"  # Use the chosen unique key
        else:
            k = f"fb:{fallback_key(d)}"  # Use the fallback identity
        if k in seen:
            continue  # Skip duplicates
        seen.add(k)
        result.append(d)
    return result


# --- Merge ---
def main():
    # Validate that all input files exist
    for p in IN_FILES:
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {p}")

    # Load all items from input files
    all_items: List[Dict[str, Any]] = []
    for p in IN_FILES:
        items = load_as_list(p)
        all_items.extend(items)

    before = len(all_items)  # Count before deduplication
    all_items = dedup(all_items)
    after = len(all_items)  # Count after deduplication

    # Ensure output directory exists and save the merged file
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(all_items, f, ensure_ascii=False, indent=2)

    # Print summary
    print(" Merge completed.")
    print(f"Inputs:  {len(IN_FILES)} files")
    print(f"Records: {before} → {after} after deduplication")
    print(f"Saved:   {OUT_FILE}")


# Run the script
if __name__ == "__main__":
    main()
