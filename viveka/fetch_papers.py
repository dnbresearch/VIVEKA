#!/usr/bin/env python3
"""
VIVEKA — Paper Repository Collection
======================================
Collects ML paper repositories from PapersWithCode and OpenReview
across 28 venues for experimental feature mining.

Usage:
  python viveka/fetch_papers.py --output data/repo_list.json
  python viveka/fetch_papers.py --output data/repo_list.json --venues CVPR,NeurIPS --years 2023,2024
"""
import argparse, json, os, sys, time, requests
from pathlib import Path


def fetch_pwc_repos(venue, year, max_pages=50):
    """Fetch repositories from PapersWithCode API."""
    repos = []
    base_url = "https://paperswithcode.com/api/v1/papers/"

    for page in range(1, max_pages + 1):
        try:
            params = {"page": page, "items_per_page": 50}
            resp = requests.get(base_url, params=params, timeout=30)
            if resp.status_code != 200:
                break
            data = resp.json()
            results = data.get("results", [])
            if not results:
                break

            for paper in results:
                # Filter by venue/year
                proc = paper.get("proceeding", "") or ""
                if venue.lower() in proc.lower() and str(year) in proc:
                    repos.append({
                        "title": paper.get("title", ""),
                        "venue": f"{venue} {year}",
                        "repo_url": paper.get("github", ""),
                        "paper_url": paper.get("url_abs", ""),
                        "source": "pwc",
                    })
            time.sleep(0.5)
        except Exception as e:
            print(f"    Error page {page}: {e}")
            break

    return repos


def fetch_openreview_repos(venue, year):
    """Fetch repositories from OpenReview API."""
    repos = []
    venue_map = {
        "NeurIPS": "NeurIPS.cc",
        "ICLR": "ICLR.cc",
        "ICML": "ICML.cc",
    }

    venue_id = venue_map.get(venue)
    if not venue_id:
        return repos

    try:
        url = f"https://api2.openreview.net/notes/search"
        params = {
            "query": f"venue:{venue_id}/{year}/Conference",
            "limit": 1000,
        }
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            for note in data.get("notes", []):
                content = note.get("content", {})
                title = content.get("title", {})
                if isinstance(title, dict):
                    title = title.get("value", "")

                # Look for code links
                code_url = ""
                for field in ["code", "code_url", "github", "supplementary_material"]:
                    val = content.get(field, {})
                    if isinstance(val, dict):
                        val = val.get("value", "")
                    if val and "github.com" in str(val):
                        code_url = str(val)
                        break

                if code_url:
                    repos.append({
                        "title": title,
                        "venue": f"{venue} {year}",
                        "repo_url": code_url,
                        "source": "openreview",
                    })
    except Exception as e:
        print(f"    OpenReview error: {e}")

    return repos


def main():
    parser = argparse.ArgumentParser(description="Collect ML paper repositories")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--venues", default=None,
                        help="Comma-separated venues (default: all 28)")
    parser.add_argument("--years", default=None,
                        help="Comma-separated years (default: 2021-2024)")
    parser.add_argument("--venue-file", default="data/venue_list.json",
                        help="Venue definition file")
    args = parser.parse_args()

    # Load venue definitions
    if os.path.exists(args.venue_file):
        with open(args.venue_file) as f:
            venue_data = json.load(f)
        all_venues = venue_data.get("venues", [])
    else:
        all_venues = [
            {"name": "NeurIPS", "years": [2021, 2022, 2023, 2024]},
            {"name": "ICML", "years": [2021, 2022, 2023, 2024]},
            {"name": "ICLR", "years": [2021, 2022, 2023, 2024]},
            {"name": "CVPR", "years": [2021, 2022, 2023, 2024]},
        ]

    # Filter by user selection
    if args.venues:
        selected = set(args.venues.split(","))
        all_venues = [v for v in all_venues if v["name"] in selected]

    if args.years:
        selected_years = set(int(y) for y in args.years.split(","))
    else:
        selected_years = None

    all_repos = []
    for venue_info in all_venues:
        venue = venue_info["name"]
        years = venue_info.get("years", [2023, 2024])
        if selected_years:
            years = [y for y in years if y in selected_years]

        for year in years:
            print(f"  Fetching {venue} {year}...", end=" ", flush=True)

            repos = fetch_pwc_repos(venue, year)
            repos += fetch_openreview_repos(venue, year)

            # Deduplicate by repo URL
            seen = set()
            unique = []
            for r in repos:
                url = r.get("repo_url", "")
                if url and url not in seen:
                    seen.add(url)
                    unique.append(r)

            all_repos.extend(unique)
            print(f"{len(unique)} repos")

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_repos, f, indent=2)

    print(f"\nTotal: {len(all_repos)} repos saved to {args.output}")


if __name__ == "__main__":
    main()
