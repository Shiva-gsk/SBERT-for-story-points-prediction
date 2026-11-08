"""Stage 1: download TAWOS data into data/raw/ (or accept local copies).

The TAWOS dataset is hosted at the Brunel Data Repository (DOI
10.5522/04/21308124). The publisher's access can change; this module is
schema- and source-flexible:

  * If `configs/ps6.yaml → data.download_url` is non-empty, it is fetched
    (HTTP) or copied (local path) into `data/raw/` and unzipped if needed.
  * Otherwise, the script inspects `data/raw/` for already-placed CSV files
    (single combined CSV with a project column, OR one CSV per project).

Nothing in this module assumes a specific upstream layout. The downstream
preprocess.py unifies the schema.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

from .config import load_config
from .utils import append_experiment_log, get_logger

log = get_logger("download_data")

# Canonical project list from guidelines §3.3 (used for reporting/manifest).
CANONICAL_PROJECTS = [
    "XD", "MESOS", "Usergrid", "Titanium Mobile Platform", "The Titanium SDK",
    "Alloy Framework", "Software Cloud", "Appcelerator Daemon",
    "Appcelerator Studio", "Sawtooth", "Nexus", "Command-Line Interface",
    "MXNet", "Mule", "Mule APIkit", "Mule C++ driver (Appcelerator)", "Moodle",
    "Atlassian Software Server", "Bamboo", "Clover", "Lyrasis Dura Cloud",
    "Lsstcorp Data management", "Confluence Cloud", "Confluence Server",
    "Jira Server", "Jira Cloud", "Java driver", "Indy SDK", "Indy Node",
    "FishEye", "DotNetNuke Platform", "Fabric", "Evergreen", "DataCass",
    "Crowd", "Compass", "Blockchain Explorer", "MongoDB Core Server",
    "C++ driver (Appcelerator)", "C++ driver (MongoDB)",
]


def _safe_download(url: str, dst: Path) -> None:
    log.info("Downloading %s -> %s", url, dst)
    req = urllib.request.Request(url, headers={"User-Agent": "ps6-repro/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r, dst.open("wb") as f:
        shutil.copyfileobj(r, f)


def _maybe_unzip(archive: Path, out_dir: Path) -> None:
    if archive.suffix.lower() == ".zip":
        log.info("Unpacking %s", archive)
        with zipfile.ZipFile(archive) as z:
            z.extractall(out_dir)
        archive.unlink()


def discover_raw_csvs(raw_dir: Path) -> list[Path]:
    return sorted(raw_dir.glob("*.csv")) + sorted(raw_dir.glob("*.CSV"))


def main() -> int:
    ap = argparse.ArgumentParser(parents=[_placeholder_parser()])
    args = ap.parse_args()
    cfg = load_config(args.config)
    raw_dir = Path(cfg["data"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    url = (cfg["data"].get("download_url") or "").strip()
    if url:
        if url.startswith("http://") or url.startswith("https://"):
            archive = raw_dir / "tawos_download.zip"
            try:
                _safe_download(url, archive)
                _maybe_unzip(archive, raw_dir)
            except Exception as e:  # pragma: no cover
                log.warning("Download failed (%s). Falling back to local files.", e)
        else:
            local = Path(url)
            if local.is_file() and local.suffix.lower() == ".zip":
                shutil.copy2(local, raw_dir / local.name)
                _maybe_unzip(raw_dir / local.name, raw_dir)
            elif local.is_dir():
                for p in local.glob("*.csv"):
                    shutil.copy2(p, raw_dir / p.name)
            else:
                log.error("download_url %s is neither an http URL, a .zip file, nor a directory.", url)
                return 2

    csvs = discover_raw_csvs(raw_dir)
    if not csvs:
        log.warning(
            "No CSVs found in %s. Place the TAWOS CSVs there manually, or set "
            "`data.download_url` in configs/ps6.yaml to the latest archive URL. "
            "The DOI is https://doi.org/10.5522/04/21308124.",
            raw_dir,
        )
    else:
        log.info("Found %d raw CSV file(s) in %s:", len(csvs), raw_dir)
        for p in csvs:
            log.info("  - %s (%.1f KB)", p.name, p.stat().st_size / 1024)

    append_experiment_log(
        cfg["results"]["experiment_log"], config=cfg, stage="download_data",
        notes=f"{len(csvs)} raw CSV file(s) discovered",
    )
    return 0


def _placeholder_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--config", default=None)
    return ap


if __name__ == "__main__":
    sys.exit(main())
