"""Fetch recent neuroscience articles and export them to an Excel workbook.

This script queries the Crossref API for articles published in Nature Neuroscience,
Neuron, and Cell over a configurable date range (default: the past two years).
Results include article metadata, author and affiliation information, and a pair of
heuristic categorizations derived from article text.
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import pandas as pd
import requests


JOURNALS: Dict[str, Dict[str, str]] = {
    "Nature Neuroscience": {"issn": "1097-6256"},
    "Neuron": {"issn": "0896-6273"},
    "Cell": {"issn": "0092-8674"},
}

MODEL_KEYWORDS: Dict[str, Sequence[str]] = {
    "artificial neuron / computational": (
        "artificial neural network",
        "convolutional neural network",
        "deep learning",
        "recurrent neural network",
        "computational model",
        "in silico",
        "artificial agent",
    ),
    "human": (
        "human",
        "participant",
        "patient",
        "healthy adult",
        "volunteer",
        "clinical trial",
    ),
    "non-human primate": (
        "macaque",
        "monkey",
        "marmoset",
        "baboon",
    ),
    "mouse / rat": (
        "mouse",
        "mice",
        "murine",
        "rat",
        "rodent",
    ),
    "zebrafish": (
        "zebrafish",
        "danio rerio",
    ),
    "drosophila": (
        "drosophila",
        "fruit fly",
    ),
    "in vitro": (
        "in vitro",
        "cell culture",
        "organoid",
        "slice culture",
        "brain organoid",
    ),
    "post-mortem / ex vivo": (
        "post-mortem",
        "postmortem",
        "ex vivo",
    ),
}

TECHNIQUE_KEYWORDS: Dict[str, Sequence[str]] = {
    "electrophysiology": (
        "electrophysiology",
        "patch clamp",
        "multi-electrode",
        "eeg",
        "meg",
        "local field potential",
        "intracellular recording",
        "extracellular recording",
    ),
    "calcium imaging / two-photon": (
        "two-photon",
        "2-photon",
        "calcium imaging",
        "fiber photometry",
        "gcamp",
    ),
    "optogenetics": (
        "optogenetic",
        "channelrhodopsin",
        "halorhodopsin",
        "optical stimulation",
    ),
    "electron microscopy": (
        "electron microscopy",
        "em dataset",
        "connectomics",
        "serial block-face",
    ),
    "viral tracing / gene delivery": (
        "viral",
        "virus",
        "adeno-associated",
        "lentiviral",
        "retrograde tracing",
        "anterograde tracing",
    ),
    "imaging (MRI / fMRI / PET)": (
        "fmri",
        "bold signal",
        "mri",
        "diffusion imaging",
        "pet imaging",
        "positron emission tomography",
    ),
    "behavioral": (
        "behavioral",
        "psychophysics",
        "task",
        "maze",
        "behavior",
    ),
    "molecular / genomic": (
        "single-cell rna",
        "transcriptomic",
        "genomic",
        "crispr",
        "proteomic",
        "mass spectrometry",
    ),
}

DEFAULT_MODEL = "unspecified"
DEFAULT_TECHNIQUE = "unspecified"


@dataclass
class ArticleRecord:
    journal: str
    published: str
    title: str
    authors: str
    author_affiliations: str
    abstract: str
    category_model: str
    category_technique: str
    doi: str
    url: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "Journal": self.journal,
            "Published": self.published,
            "Title": self.title,
            "Authors": self.authors,
            "Author Affiliations": self.author_affiliations,
            "Abstract": self.abstract,
            "Category - Model": self.category_model,
            "Category - Technique": self.category_technique,
            "DOI": self.doi,
            "URL": self.url,
        }


def fetch_crossref_items(issn: str, start_date: str, end_date: str, max_records: int | None = None) -> List[Dict[str, Any]]:
    """Retrieve Crossref records for a journal within the date range."""
    rows = 200
    offset = 0
    items: List[Dict[str, Any]] = []

    while True:
        params = {
            "filter": f"from-pub-date={start_date},until={end_date},type=journal-article",
            "rows": rows,
            "offset": offset,
            "mailto": "research-bot@example.com",
        }
        response = requests.get(f"https://api.crossref.org/journals/{issn}/works", params=params, timeout=60)
        response.raise_for_status()
        message = response.json().get("message", {})
        batch = message.get("items", [])
        if not batch:
            break
        items.extend(batch)

        if max_records is not None and len(items) >= max_records:
            return items[:max_records]

        if len(batch) < rows:
            break
        offset += rows
        time.sleep(0.2)

    return items


def normalize_text(value: str | None) -> str:
    if not value:
        return ""
    collapsed = re.sub(r"\s+", " ", value)
    return collapsed.strip()


def clean_abstract(raw_abstract: str | None) -> str:
    if not raw_abstract:
        return ""
    text = re.sub(r"<[^>]+>", " ", raw_abstract)
    text = html.unescape(text)
    return normalize_text(text)


def join_nonempty(values: Iterable[str], delimiter: str = "; ") -> str:
    cleaned = [normalize_text(v) for v in values if v]
    return delimiter.join(v for v in cleaned if v)


def parse_authors(author_data: Sequence[Dict[str, Any]] | None) -> tuple[str, str]:
    if not author_data:
        return "", ""

    names: List[str] = []
    affiliations: List[str] = []

    for author in author_data:
        given = normalize_text(author.get("given"))
        family = normalize_text(author.get("family"))
        name = join_nonempty([given, family], delimiter=" ")
        if name:
            names.append(name)
        aff_list = [normalize_text(aff.get("name")) for aff in author.get("affiliation", []) if normalize_text(aff.get("name"))]
        if aff_list:
            affiliations.append(f"{name}: {', '.join(aff_list)}")

    return join_nonempty(names), join_nonempty(affiliations)


def detect_category(text: str, keyword_map: Dict[str, Sequence[str]], default: str) -> str:
    lowered = text.lower()
    for category, keywords in keyword_map.items():
        if any(keyword in lowered for keyword in keywords):
            return category
    return default


def derive_categories(title: str, abstract: str) -> tuple[str, str]:
    combined = f"{title} {abstract}".lower()
    model = detect_category(combined, MODEL_KEYWORDS, DEFAULT_MODEL)
    technique = detect_category(combined, TECHNIQUE_KEYWORDS, DEFAULT_TECHNIQUE)
    return model, technique


def parse_publication_date(item: Dict[str, Any]) -> str:
    date_keys = ("published-print", "published-online", "issued")
    for key in date_keys:
        date_parts = item.get(key, {}).get("date-parts")
        if date_parts:
            parts = date_parts[0]
            return "-".join(str(part) for part in parts)
    return ""


def build_article_records(journal: str, items: Sequence[Dict[str, Any]]) -> List[ArticleRecord]:
    records: List[ArticleRecord] = []
    for item in items:
        title = join_nonempty(item.get("title", []))
        authors, affiliations = parse_authors(item.get("author"))
        abstract = clean_abstract(item.get("abstract"))
        model_category, technique_category = derive_categories(title, abstract)
        record = ArticleRecord(
            journal=journal,
            published=parse_publication_date(item),
            title=title,
            authors=authors,
            author_affiliations=affiliations,
            abstract=abstract,
            category_model=model_category,
            category_technique=technique_category,
            doi=normalize_text(item.get("DOI")),
            url=normalize_text(item.get("URL")),
        )
        records.append(record)
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch recent neuroscience journal articles and export to Excel.")
    parser.add_argument(
        "--output",
        default="recent_neuroscience_articles.xlsx",
        help="Destination Excel file (default: recent_neuroscience_articles.xlsx)",
    )
    parser.add_argument(
        "--start-date",
        help="Start date (YYYY-MM-DD). Defaults to two years before today.",
    )
    parser.add_argument(
        "--end-date",
        help="End date (YYYY-MM-DD). Defaults to today.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Optional limit for the number of records per journal (useful for testing).",
    )
    return parser.parse_args()


def determine_date_range(start_date: str | None, end_date: str | None) -> tuple[str, str]:
    today = dt.date.today()
    default_end = today
    default_start = today - dt.timedelta(days=730)

    start = dt.date.fromisoformat(start_date) if start_date else default_start
    end = dt.date.fromisoformat(end_date) if end_date else default_end

    if start > end:
        raise ValueError("Start date must not be later than end date.")

    return start.isoformat(), end.isoformat()


def export_to_excel(records: Sequence[ArticleRecord], output_path: str) -> None:
    frame = pd.DataFrame([record.to_dict() for record in records])
    frame.sort_values(by=["Published", "Journal", "Title"], inplace=True)
    frame.to_excel(output_path, index=False)


def main() -> None:
    args = parse_args()
    start_date, end_date = determine_date_range(args.start_date, args.end_date)

    all_records: List[ArticleRecord] = []
    for journal, metadata in JOURNALS.items():
        print(f"Fetching {journal} articles from {start_date} to {end_date}...")
        items = fetch_crossref_items(metadata["issn"], start_date, end_date, max_records=args.max_records)
        print(f"  Retrieved {len(items)} records")
        all_records.extend(build_article_records(journal, items))

    if not all_records:
        print("No records found for the specified range.")
        return

    export_to_excel(all_records, args.output)
    print(f"Exported {len(all_records)} articles to {args.output}")


if __name__ == "__main__":
    main()
