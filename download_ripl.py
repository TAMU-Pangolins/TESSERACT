"""
Download RIPL-3 HFB+combinatorial level-density tables (zXXX.tab and zXXX.cor) into ./data/densities/level-densities-hfb.

Options (edit the "USER SETTINGS" block below):
  - ONLY_Z_RANGE: set to (zmin, zmax) to restrict to a Z range, or None for all.
  - INCLUDE_COR:  True to also fetch zXXX.cor correction files (recommended).
"""

from pathlib import Path
import re
import sys
import time
import urllib.request
import urllib.error

from nucres.config import store_data_root

# --------------------------- USER SETTINGS ---------------------------
ONLY_Z_RANGE = None          # e.g. (8, 28) to fetch Z=8..28 only; or None for all Z
INCLUDE_COR  = True          # also download zXXX.cor if present
DEST_DIR     = Path.cwd() / "data" / "densities" / "level-densities-hfb"
# ---------------------------------------------------------------------

BASE = "https://www-nds.iaea.org/RIPL-3/densities/level-densities-hfb/"
UA   = "Mozilla/5.0 (compatible; RIPL-HFB-fetch/1.0)"
TIMEOUT = 60
RETRY   = 3
WAIT_BETWEEN = 0.25  # be nice to the server

MANUAL_DOWNLOAD_INSTRUCTIONS = f"""
The RIPL-3 server returned HTTP 403 Forbidden. This can happen if the server or
its front-end protection blocks scripted downloads.

Manual fallback:
  1. Open this URL in a browser:
     {BASE}
  2. Download the needed zXXX.tab files and, if available, matching zXXX.cor
     correction files.
  3. Place them in:
     {DEST_DIR}
  4. Point nucres at that directory:
     export NUCRES_DATA_ROOT="{DEST_DIR}"

After the files are present, rerun the THICC workflow that needed HFB data.
"""

def http_get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return r.read()

def scrape_hfb_listing() -> list[str]:
    try:
        html = http_get(BASE).decode("utf-8", errors="ignore")
    except urllib.error.HTTPError as exc:
        if exc.code == 403:
            raise RuntimeError(MANUAL_DOWNLOAD_INSTRUCTIONS.strip()) from exc
        raise
    # Pick out zNNN.tab and zNNN.cor names
    hits = re.findall(r"z(\d{3})\.(tab|cor)", html)
    names = {f"z{z}.{ext}" for z, ext in hits}
    # Filter .cor if not requested
    if not INCLUDE_COR:
        names = {n for n in names if n.endswith(".tab")}
    # Filter by Z range if requested
    if ONLY_Z_RANGE:
        zmin, zmax = ONLY_Z_RANGE
        keep = set()
        for n in names:
            z = int(n[1:4])
            if zmin <= z <= zmax:
                keep.add(n)
        names = keep
    return sorted(names)

def download_file(name: str, dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    url  = BASE + name
    dest = dest_dir / name
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  exists: {name}")
        return
    last_err = None
    for attempt in range(1, RETRY + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp, open(dest, "wb") as f:
                total = resp.length if hasattr(resp, "length") else None
                got = 0
                while True:
                    chunk = resp.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
                    got += len(chunk)
                    if total:
                        pct = 100.0 * got / total
                        sys.stdout.write(f"\r  downloading {name}: {got}/{total} bytes ({pct:5.1f}%)")
                    else:
                        sys.stdout.write(f"\r  downloading {name}: {got} bytes")
                    sys.stdout.flush()
            sys.stdout.write("\n")
            return
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            if isinstance(e, urllib.error.HTTPError) and e.code == 403:
                raise RuntimeError(MANUAL_DOWNLOAD_INSTRUCTIONS.strip()) from e
            last_err = e
            print(f"  attempt {attempt} failed for {name}: {e}")
            time.sleep(1.0 * attempt)
    raise RuntimeError(f"Failed: {url} -> {dest} after {RETRY} attempts ({last_err})")

def main():
    print("Destination:", DEST_DIR)
    try:
        files = scrape_hfb_listing()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    if not files:
        print("No files found — check connectivity or the directory URL.")
        return
    print(f"Found {len(files)} files to fetch.")
    for i, name in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {name}")
        try:
            download_file(name, DEST_DIR)
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
        time.sleep(WAIT_BETWEEN)
    store_data_root(DEST_DIR)
    print("Done.")
    print(f"nucres data root stored at {DEST_DIR}")

if __name__ == "__main__":
    main()
