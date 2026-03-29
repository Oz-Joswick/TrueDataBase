"""
integration/ig_scraper.py
──────────────────────────
Download images from a public Instagram profile.
Extracts @mention tags from captions. No Claude, no AI analysis.

Usage:
    python ig_scraper.py <username> --count 100 [--output ./ig_output]
    python ig_scraper.py <username> --count 100 --mentions-only  # only posts with @tags

Output:
    <output>/<YYYY-MM-DD>_<username>_<N>posts/images/   — downloaded images
    <output>/<YYYY-MM-DD>_<username>_<N>posts/catalog.json — post metadata
"""

import argparse
import datetime
import json
import re
import sys
import time
from pathlib import Path

import requests
import instaloader


def extract_mentions(caption: str | None) -> list[str]:
    if not caption:
        return []
    return re.findall(r"@(\w+)", caption)


def get_image_urls(post: instaloader.Post) -> list[str]:
    """Return all image URLs for a post (handles single + sidecar/carousel)."""
    if post.typename == "GraphSidecar":
        return [node.display_url for node in post.get_sidecar_nodes()
                if not node.is_video]
    elif not post.is_video:
        return [post.url]
    else:
        return []  # video post with no thumbnail


def download_image(url: str, dest: Path, retries: int = 2) -> bool:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, headers=headers, timeout=30)
            r.raise_for_status()
            dest.write_bytes(r.content)
            return True
        except Exception as e:
            if attempt < retries:
                time.sleep(1)
            else:
                print(f"    ! download failed: {e}")
    return False


def scrape(username: str, count: int, output_dir: Path, mentions_only: bool) -> list[dict]:
    """Scrape posts into output_dir. Returns catalog list (does not write catalog.json)."""
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    loader = instaloader.Instaloader(quiet=True, download_pictures=False)

    print(f"Connecting to Instagram @{username}...")
    try:
        profile = instaloader.Profile.from_username(loader.context, username)
    except instaloader.exceptions.ProfileNotExistsException:
        print(f"Error: profile '{username}' not found or is private.")
        sys.exit(1)

    print(f"Fetching up to {count} posts...")
    catalog = []
    fetched = 0

    for post in profile.get_posts():
        if fetched >= count:
            break

        mentions = extract_mentions(post.caption)
        urls = get_image_urls(post)

        if mentions_only and not mentions:
            continue  # skip posts without @tags

        if not urls:
            continue  # skip video-only posts

        fetched += 1
        print(f"  [{fetched}] {post.shortcode}  @mentions={mentions}", end="", flush=True)

        # Download images
        image_paths = []
        for idx, url in enumerate(urls):
            suffix = f"_{idx + 1}" if len(urls) > 1 else ""
            dest = images_dir / f"{post.shortcode}{suffix}.jpg"
            if dest.exists():
                image_paths.append(str(dest))
            elif download_image(url, dest):
                image_paths.append(str(dest))

        print(f"  {len(image_paths)}/{len(urls)} image(s)")

        catalog.append({
            "shortcode": post.shortcode,
            "url": f"https://www.instagram.com/p/{post.shortcode}/",
            "caption": post.caption or "",
            "mentions": mentions,
            "timestamp": post.date_utc.isoformat(),
            "image_paths": image_paths,   # list — one per carousel slide
        })

    return catalog


def main():
    p = argparse.ArgumentParser(description="Download Instagram images for TrueDataBase import")
    p.add_argument("username", help="Public Instagram username (no @)")
    p.add_argument("--count", type=int, default=100, help="Max posts to fetch (default 100)")
    p.add_argument("--output", default="./ig_output", help="Base output directory (default ./ig_output)")
    p.add_argument("--mentions-only", action="store_true",
                   help="Only include posts where people are @mentioned in caption")
    args = p.parse_args()

    username = args.username.lstrip("@")
    base_dir = Path(args.output)
    date_str = datetime.date.today().strftime("%Y-%m-%d")

    # Scrape into a temp directory first (post count unknown until done)
    temp_dir = base_dir / f"{date_str}_{username}_scraping"
    catalog = scrape(username, args.count, temp_dir, args.mentions_only)

    # Final directory name includes actual post count
    n = len(catalog)
    final_name = f"{date_str}_{username}_{n}posts"
    final_dir = base_dir / final_name

    # Avoid collisions with existing runs from the same day
    if final_dir.exists():
        suffix = int(time.time())
        final_dir = base_dir / f"{final_name}_{suffix}"

    # Rename temp dir → final dir and fix absolute image paths in catalog
    temp_dir.rename(final_dir)
    for post in catalog:
        post["image_paths"] = [
            str(final_dir / Path(p).relative_to(temp_dir))
            for p in post["image_paths"]
        ]

    catalog_path = final_dir / "catalog.json"
    catalog_path.write_text(json.dumps(catalog, indent=2, ensure_ascii=False))

    print(f"\nDone. {n} posts saved to:")
    print(f"  {final_dir}/")
    print(f"  catalog: {catalog_path}")


if __name__ == "__main__":
    main()
