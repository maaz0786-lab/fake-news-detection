"""Free internet verification for fake news detection.

This module checks whether a submitted statement appears on the public web
using DuckDuckGo's HTML search endpoint.  No API key or paid service is
required -- it is completely free.

It is a *learning demo*: finding a sentence online does not prove it is true,
and not finding it does not prove it is false.  Always combine this with the
ML model prediction and human judgement.
"""

from __future__ import annotations

import re
import urllib.parse
import urllib.request
from typing import Optional


def _clean_html(text: str) -> str:
    """Strip HTML tags and collapse whitespace."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _fetch_search_results(query: str, max_results: int = 8) -> list[dict]:
    """Return a list of search result dicts from DuckDuckGo HTML search.

    Each dict contains:
        title   : str -- result title
        url     : str -- result URL (where the text is written)
        snippet : str -- result description snippet

    Returns an empty list if the request fails (no internet, blocked, etc.).
    """
    url = "https://www.bing.com/search?q=" + urllib.parse.quote(query)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as response:
            html = response.read().decode("utf-8", errors="ignore")
    except Exception:
        return []

    results: list[dict] = []

    title_pattern = re.compile(
        r'<h2[^>]*>\s*<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
        re.DOTALL | re.IGNORECASE,
    )
    snippet_pattern = re.compile(
        r'<p[^>]*class="[^"]*b_lineclamp[^"]*"[^>]*>(.*?)</p>', re.DOTALL | re.IGNORECASE
    )

    titles = title_pattern.findall(html)
    snippets = snippet_pattern.findall(html)

    for i, (href, title_raw) in enumerate(titles):
        title = _clean_html(title_raw)
        actual_url = _extract_actual_url(href)
        snippet = _clean_html(snippets[i]) if i < len(snippets) else ""
        if title or actual_url:
            results.append({
                "title": title or "(no title)",
                "url": actual_url,
                "snippet": snippet,
            })
        if len(results) >= max_results:
            break

    return results


def _extract_actual_url(ddg_redirect: str) -> str:
    """Extract the real destination URL from a DuckDuckGo redirect link."""
    parsed = urllib.parse.urlparse(ddg_redirect)
    query_params = urllib.parse.parse_qs(parsed.query)
    if "uddg" in query_params:
        return query_params["uddg"][0]
    return ddg_redirect


def verify_online(text: str) -> dict:
    """Check if *text* appears on the public web with full transparency.

    Returns a dict with:
        found              : bool -- any matching results found
        result_count       : int -- total number of search results returned
        results            : list[dict] -- full list of {title, url, snippet}
        queries_used       : list[str] -- the search queries that were run
        search_transparency: dict -- detailed breakdown for the UI
        note               : str -- human-readable explanation
    """
    transparency = {
        "queries_used": [],
        "searched_pages": [],
        "found_on_pages": [],
        "not_found": False,
        "internet_available": False,
    }

    if not text or len(text.strip()) < 10:
        return {
            "found": False,
            "result_count": 0,
            "results": [],
            "queries_used": [],
            "search_transparency": transparency,
            "note": "Text too short to verify online.",
        }

    # Search for the exact phrase and a shortened keyword version
    queries = [f'"{text}"', text[:120]]
    all_results: list[dict] = []
    seen_urls: set[str] = set()
    found_on_pages: list[dict] = []       # pages where exact text appears
    exact_matches: list[dict] = []       # pages where exact text appears (same as found_on_pages)

    for query in queries:
        transparency["queries_used"].append(query)
        results = _fetch_search_results(query)
        transparency["internet_available"] = True

        for result in results:
            # Track every page researched (full transparency)
            transparency["searched_pages"].append({
                "title": result["title"],
                "url": result["url"],
                "snippet": result["snippet"],
                "query": query,
            })

            # De-duplicate by URL for the main results list
            if result["url"] not in seen_urls:
                seen_urls.add(result["url"])
                all_results.append(result)

            # Check if exact text appears in title/snippet -> this page HAS the text
            combined = (result["title"] + " " + result["snippet"]).lower()
            if text.lower() in combined:
                # Avoid duplicate entries in found_on_pages
                if result["url"] not in [fp["url"] for fp in found_on_pages]:
                    found_on_pages.append({
                        "title": result["title"],
                        "url": result["url"],
                        "snippet": result["snippet"],
                        "query": query,
                    })
                    exact_matches.append({
                        "title": result["title"],
                        "url": result["url"],
                        "snippet": result["snippet"],
                        "query": query,
                    })

        if len(all_results) >= 8:
            break

    # Exact match detection (before fallback for display)
    exact_found = len(found_on_pages) > 0

    # If no exact matches, show the searched pages as the source (display only)
    if not found_on_pages and all_results:
        found_on_pages = all_results[:8]
        exact_matches = found_on_pages

    found = exact_found
    transparency["found_on_pages"] = found_on_pages
    transparency["exact_matches"] = exact_matches
    transparency["not_found"] = not found
    note = (
        f"Statement found on {len(found_on_pages)} page(s) on the public web."
        if found_on_pages
        else "Statement not found on any of the searched pages (may still be real)."
    )

    return {
        "found": found,
        "result_count": len(all_results),
        "results": all_results[:8],
        "queries_used": transparency["queries_used"],
        "search_transparency": transparency,
        "note": note,
    }


def combined_prediction(ml_label: str, ml_confidence: float, online: dict) -> dict:
    """Combine ML model output with internet verification.

    Heuristic (learning demo only):
      - ML says REAL  + found online   -> REAL, higher confidence
      - ML says REAL  + not found     -> REAL, lower confidence
      - ML says FAKE  + found online   -> FAKE, lower confidence (could be widely shared fake)
      - ML says FAKE  + not found      -> FAKE, higher confidence
    """
    adjusted_confidence = ml_confidence
    reason_parts = [f"ML model predicts {ml_label} ({ml_confidence}% confidence)."]

    st = online.get("search_transparency", {})

    if not online["found"]:
        reason_parts.append("Statement was not found on any of the searched pages.")
        if ml_label == "FAKE":
            adjusted_confidence = min(99.0, ml_confidence + 15)
        else:
            adjusted_confidence = max(10.0, ml_confidence - 15)
    else:
        pages_found = len(st.get("found_on_pages", []))
        reason_parts.append(
            f"Statement was found on {pages_found} page(s) on the public web."
        )
        if ml_label == "REAL":
            adjusted_confidence = min(99.0, ml_confidence + 10)
        else:
            adjusted_confidence = max(10.0, ml_confidence - 10)

    # If found on the internet -> REAL; otherwise use ML prediction
    if online["found"]:
        final_label = "REAL"
    else:
        final_label = ml_label
    reason = " ".join(reason_parts)

    return {
        "label": final_label,
        "confidence": round(adjusted_confidence, 1),
        "ml_label": ml_label,
        "ml_confidence": round(ml_confidence, 1),
        "online_found": online["found"],
        "online_count": online["result_count"],
        "online_samples": [r["snippet"] for r in online.get("results", [])[:3]],
        "online_note": online["note"],
        "search_transparency": st,
        "online_results": online.get("results", []),
        "reason": reason,
    }
