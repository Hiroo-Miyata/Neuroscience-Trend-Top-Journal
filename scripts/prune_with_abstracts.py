#!/usr/bin/env python3
"""Prune neurotechnology papers using abstracts with LLM or rule-based fallback."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from openai import OpenAI
    from openai import APIError, APITimeoutError, RateLimitError
except ImportError:  # pragma: no cover - handled by fallback mode
    OpenAI = None  # type: ignore[assignment]
    APIError = APITimeoutError = RateLimitError = Exception  # type: ignore


TARGET_JOURNALS = {
    "nature",
    "nature neuroscience",
    "science",
    "cell",
    "new england journal of medicine",
    "nejm",
    "the lancet",
    "lancet",
}

DEFAULT_MODEL = "gpt-4o-mini"

CACHE_FILENAME = "cache_llm_labels.jsonl"

NEUROTECH_KEYWORDS = [
    "brain-computer interface",
    "bci",
    "bmi",
    "neural interface",
    "intracortical",
    "electrode array",
    "utah array",
    "ecog",
    "electrocorticography",
    "deep brain stimulation",
    "dbs",
    "neurostimulation",
    "cortical stimulation",
    "neural recording",
    "implant",
    "decoder",
    "neuroprosthesis",
]

SPECIES_KEYWORDS = {
    "human": ["human", "patient", "participant", "volunteer"],
    "macaque": ["rhesus", "macaque", "monkey", "nonhuman primate", "nhp"],
}

MODALITY_KEYWORDS = {
    "intracortical": ["intracortical", "utah array", "electrode array"],
    "ECoG": ["ecog", "electrocorticography"],
    "DBS": ["deep brain stimulation", "dbs"],
    "stimulation_non_DBS": ["neurostimulation", "cortical stimulation", "tms", "tdcs"],
    "imaging_only": ["fmri", "meg", "eeg", "pet", "mri", "imaging"],
    "software_only": ["decoder", "algorithm", "software", "modeling"],
}

REVIEW_HINTS = ["review", "meta-analysis", "systematic review", "commentary", "perspective", "editorial"]


@dataclass
class Paper:
    title: str
    journal: str
    publication_date: str
    abstract: str
    doi: str
    url: str
    species: str
    neurotech_modality: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prune papers using abstract-aware LLM filtering.")
    parser.add_argument("--input", default="out/papers.csv", help="Path to input CSV.")
    parser.add_argument("--output-dir", default="out", help="Directory for outputs.")
    parser.add_argument("--batch", action="store_true", help="Use OpenAI Batch API.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model name.")
    parser.add_argument("--max-papers", type=int, default=None, help="Limit number of papers processed.")
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def parse_date(value: str) -> Optional[dt.date]:
    if not value:
        return None
    value = value.strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m", "%Y/%m", "%Y"):
        try:
            parsed = dt.datetime.strptime(value, fmt).date()
            if fmt == "%Y":
                return parsed.replace(month=1, day=1)
            if fmt in ("%Y-%m", "%Y/%m"):
                return parsed.replace(day=1)
            return parsed
        except ValueError:
            continue
    return None


def cutoff_date() -> dt.date:
    today = dt.date.today()
    try:
        return today.replace(year=today.year - 5)
    except ValueError:
        return today.replace(month=2, day=28, year=today.year - 5)


def load_papers(path: Path, max_papers: Optional[int]) -> List[Paper]:
    papers: List[Paper] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            paper = Paper(
                title=normalize_text(row.get("title", "")),
                journal=normalize_text(row.get("journal", "")),
                publication_date=normalize_text(row.get("publication_date", "")),
                abstract=normalize_text(row.get("abstract", "")),
                doi=normalize_text(row.get("doi", "")),
                url=normalize_text(row.get("url", "")),
                species=normalize_text(row.get("species", "")),
                neurotech_modality=normalize_text(row.get("neurotech_modality", "")),
            )
            papers.append(paper)
            if max_papers and len(papers) >= max_papers:
                break
    return papers


def journal_allowed(journal: str) -> bool:
    return journal.lower() in TARGET_JOURNALS


def filter_by_time(paper: Paper, min_date: dt.date) -> Tuple[bool, str]:
    parsed = parse_date(paper.publication_date)
    if not parsed:
        return False, "missing publication_date"
    if parsed < min_date:
        return False, f"publication_date {parsed.isoformat()} older than cutoff"
    return True, ""


def cache_key(paper: Paper) -> str:
    if paper.doi:
        return f"doi:{paper.doi.lower()}"
    identity = f"{paper.title}|{paper.journal}|{paper.publication_date}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return f"hash:{digest}"


def load_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    cache: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return cache
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = payload.get("cache_key")
            label = payload.get("label")
            if key and label:
                cache[key] = label
    return cache


def append_cache(path: Path, key: str, label: Dict[str, Any], model: str) -> None:
    payload = {"cache_key": key, "label": label, "model": model, "timestamp": dt.datetime.utcnow().isoformat()}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def build_prompt(paper: Paper) -> List[Dict[str, str]]:
    system = (
        "You are a careful scientific abstract classifier. "
        "Use only the title and abstract provided. "
        "Be conservative: when uncertain, use \"uncertain\" instead of false."
    )
    user = (
        "Classify the paper using the following JSON schema strictly:\n"
        "{\n"
        '  "include": true/false/"uncertain",\n'
        '  "species": "human"|"macaque"|"other"|"unknown",\n'
        '  "neurotech": true/false/"uncertain",\n'
        '  "modality": one of ["intracortical","ECoG","DBS","stimulation_non_DBS","imaging_only","software_only","other"],\n'
        '  "study_type": "primary"|"review"|"commentary"|"unknown",\n'
        '  "relevance_score": integer 0-10,\n'
        '  "one_sentence_rationale": string,\n'
        '  "keywords": [..]\n'
        "}\n\n"
        f"Title: {paper.title}\n"
        f"Abstract: {paper.abstract or '[NO ABSTRACT]'}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def normalize_label(label: Dict[str, Any]) -> Dict[str, Any]:
    defaults = {
        "include": "uncertain",
        "species": "unknown",
        "neurotech": "uncertain",
        "modality": "other",
        "study_type": "unknown",
        "relevance_score": 0,
        "one_sentence_rationale": "No rationale provided.",
        "keywords": [],
    }
    normalized = {**defaults, **label}
    if normalized["include"] not in (True, False, "uncertain"):
        normalized["include"] = "uncertain"
    if normalized["species"] not in ("human", "macaque", "other", "unknown"):
        normalized["species"] = "unknown"
    if normalized["neurotech"] not in (True, False, "uncertain"):
        normalized["neurotech"] = "uncertain"
    if normalized["modality"] not in (
        "intracortical",
        "ECoG",
        "DBS",
        "stimulation_non_DBS",
        "imaging_only",
        "software_only",
        "other",
    ):
        normalized["modality"] = "other"
    if normalized["study_type"] not in ("primary", "review", "commentary", "unknown"):
        normalized["study_type"] = "unknown"
    try:
        normalized["relevance_score"] = int(normalized["relevance_score"])
    except (TypeError, ValueError):
        normalized["relevance_score"] = 0
    normalized["keywords"] = list(normalized.get("keywords") or [])
    return normalized


def call_llm(client: OpenAI, model: str, paper: Paper) -> Dict[str, Any]:
    messages = build_prompt(paper)
    backoff = 1.0
    for attempt in range(6):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or "{}"
            label = json.loads(content)
            return normalize_label(label)
        except (RateLimitError, APIError, APITimeoutError, json.JSONDecodeError) as exc:
            if attempt == 5:
                logging.warning("LLM failed for %s: %s", paper.title[:80], exc)
                break
            time.sleep(backoff)
            backoff *= 2
    return normalize_label({})


def batch_llm(
    client: OpenAI,
    model: str,
    papers: List[Paper],
    batch_dir: Path,
) -> Dict[str, Dict[str, Any]]:
    batch_dir.mkdir(parents=True, exist_ok=True)
    input_path = batch_dir / "batch_input.jsonl"
    with input_path.open("w", encoding="utf-8") as handle:
        for paper in papers:
            key = cache_key(paper)
            payload = {
                "custom_id": key,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": build_prompt(paper),
                    "temperature": 0,
                    "response_format": {"type": "json_object"},
                },
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    with input_path.open("rb") as handle:
        file_obj = client.files.create(file=handle, purpose="batch")
    batch = client.batches.create(input_file_id=file_obj.id, endpoint="/v1/chat/completions", completion_window="24h")
    logging.info("Batch submitted: %s", batch.id)

    while batch.status not in {"completed", "failed", "cancelled"}:
        time.sleep(10)
        batch = client.batches.retrieve(batch.id)
        logging.info("Batch status: %s", batch.status)

    results: Dict[str, Dict[str, Any]] = {}
    if batch.status != "completed":
        logging.warning("Batch did not complete: %s", batch.status)
        return results
    output_file_id = batch.output_file_id
    if not output_file_id:
        logging.warning("Batch missing output file.")
        return results
    output_path = batch_dir / "batch_output.jsonl"
    content = client.files.content(output_file_id)
    output_path.write_bytes(content.read())
    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            data = json.loads(line)
            key = data.get("custom_id")
            response = data.get("response", {})
            choices = response.get("body", {}).get("choices", [])
            if not choices:
                continue
            message = choices[0].get("message", {}).get("content", "{}")
            try:
                label = normalize_label(json.loads(message))
            except json.JSONDecodeError:
                label = normalize_label({})
            if key:
                results[key] = label
    return results


def rule_based_label(paper: Paper) -> Dict[str, Any]:
    text = f"{paper.title} {paper.abstract}".lower()
    species = "unknown"
    for candidate, keywords in SPECIES_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            species = candidate
            break

    neurotech_hits = [kw for kw in NEUROTECH_KEYWORDS if kw in text]
    neurotech = True if neurotech_hits else "uncertain"

    study_type = "primary"
    if any(hint in text for hint in REVIEW_HINTS):
        study_type = "review"

    modality = "other"
    for candidate, keywords in MODALITY_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            modality = candidate
            break

    include: Any = "uncertain"
    if neurotech is True and species in {"human", "macaque"} and study_type == "primary":
        include = True
    elif neurotech is True and species in {"unknown", "other"}:
        include = "uncertain"
    elif neurotech == "uncertain":
        include = "uncertain"
    else:
        include = False

    relevance = min(10, len(neurotech_hits)) if neurotech_hits else 0
    rationale = "Rule-based heuristic; manual review recommended."
    return normalize_label(
        {
            "include": include,
            "species": species,
            "neurotech": neurotech,
            "modality": modality,
            "study_type": study_type,
            "relevance_score": relevance,
            "one_sentence_rationale": rationale,
            "keywords": neurotech_hits,
        }
    )


def decision_bucket(label: Dict[str, Any]) -> str:
    include = label.get("include")
    species = label.get("species")
    neurotech = label.get("neurotech")
    study_type = label.get("study_type")
    if include is True and species in {"human", "macaque"} and neurotech is True and study_type == "primary":
        return "accept"
    if include == "uncertain" or species == "unknown" or neurotech == "uncertain" or study_type == "unknown":
        return "review"
    return "reject"


def rejection_reason(label: Dict[str, Any]) -> str:
    parts = []
    if label.get("include") is False:
        parts.append("include=false")
    if label.get("species") not in {"human", "macaque"}:
        parts.append(f"species={label.get('species')}")
    if label.get("neurotech") is False:
        parts.append("neurotech=false")
    if label.get("study_type") != "primary":
        parts.append(f"study_type={label.get('study_type')}")
    return "; ".join(parts) or "criteria not met"


def deduplicate(papers: List[Tuple[Paper, Dict[str, Any]]]) -> List[Tuple[Paper, Dict[str, Any]]]:
    grouped: Dict[str, Tuple[Paper, Dict[str, Any]]] = {}
    for paper, label in papers:
        key = cache_key(paper)
        current = grouped.get(key)
        if not current or label.get("relevance_score", 0) > current[1].get("relevance_score", 0):
            grouped[key] = (paper, label)
    return list(grouped.values())


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    papers = load_papers(input_path, args.max_papers)
    min_date = cutoff_date()
    logging.info("Loaded %s papers. Cutoff date: %s", len(papers), min_date.isoformat())

    remaining: List[Paper] = []
    rejected_rows: List[Dict[str, Any]] = []
    for paper in papers:
        if not journal_allowed(paper.journal):
            rejected_rows.append({**paper.__dict__, "rejection_reason": "journal not in target list"})
            continue
        allowed, reason = filter_by_time(paper, min_date)
        if not allowed:
            rejected_rows.append({**paper.__dict__, "rejection_reason": reason})
            continue
        remaining.append(paper)

    cache_path = output_dir / CACHE_FILENAME
    cache = load_cache(cache_path)
    labels: Dict[str, Dict[str, Any]] = {}

    api_key = os.getenv("OPENAI_API_KEY")
    use_llm = bool(api_key and OpenAI)
    client = OpenAI(api_key=api_key) if use_llm else None

    to_label = [paper for paper in remaining if cache_key(paper) not in cache]
    logging.info("Need labels for %s papers (cached: %s)", len(to_label), len(remaining) - len(to_label))

    if use_llm and to_label:
        if args.batch:
            batch_labels = batch_llm(client, args.model, to_label, output_dir / "batch_runs")  # type: ignore[arg-type]
            for paper in to_label:
                key = cache_key(paper)
                label = batch_labels.get(key, normalize_label({}))
                cache[key] = label
                append_cache(cache_path, key, label, args.model)
        else:
            for paper in to_label:
                label = call_llm(client, args.model, paper)  # type: ignore[arg-type]
                key = cache_key(paper)
                cache[key] = label
                append_cache(cache_path, key, label, args.model)
                time.sleep(0.2)
    else:
        if not use_llm:
            logging.warning("OPENAI_API_KEY missing or OpenAI SDK unavailable. Using rule-based fallback.")
        for paper in to_label:
            label = rule_based_label(paper)
            key = cache_key(paper)
            cache[key] = label
            append_cache(cache_path, key, label, "rule-based")

    for paper in remaining:
        key = cache_key(paper)
        labels[key] = cache.get(key, normalize_label({}))

    labeled_pairs = [(paper, labels[cache_key(paper)]) for paper in remaining]
    deduped = deduplicate(labeled_pairs)

    accepted_rows: List[Dict[str, Any]] = []
    review_rows: List[Dict[str, Any]] = []
    for paper, label in deduped:
        bucket = decision_bucket(label)
        row = {**paper.__dict__, **label}
        if bucket == "accept":
            accepted_rows.append(row)
        elif bucket == "review":
            review_rows.append(row)
        else:
            rejected_rows.append({**row, "rejection_reason": rejection_reason(label)})

    labels_output = [
        {
            "cache_key": cache_key(paper),
            "title": paper.title,
            "doi": paper.doi,
            "journal": paper.journal,
            "publication_date": paper.publication_date,
            "label": label,
        }
        for paper, label in deduped
    ]

    write_csv(
        output_dir / "papers_pruned.csv",
        accepted_rows,
        fieldnames=list(Paper.__dataclass_fields__.keys())
        + [
            "include",
            "species",
            "neurotech",
            "modality",
            "study_type",
            "relevance_score",
            "one_sentence_rationale",
            "keywords",
        ],
    )
    write_csv(
        output_dir / "papers_review_queue.csv",
        review_rows,
        fieldnames=list(Paper.__dataclass_fields__.keys())
        + [
            "include",
            "species",
            "neurotech",
            "modality",
            "study_type",
            "relevance_score",
            "one_sentence_rationale",
            "keywords",
        ],
    )
    write_csv(
        output_dir / "papers_rejected.csv",
        rejected_rows,
        fieldnames=list(Paper.__dataclass_fields__.keys())
        + [
            "include",
            "species",
            "neurotech",
            "modality",
            "study_type",
            "relevance_score",
            "one_sentence_rationale",
            "keywords",
            "rejection_reason",
        ],
    )
    with (output_dir / "papers_llm_labels.json").open("w", encoding="utf-8") as handle:
        json.dump(labels_output, handle, ensure_ascii=False, indent=2)

    logging.info(
        "Wrote %s accepted, %s review, %s rejected.",
        len(accepted_rows),
        len(review_rows),
        len(rejected_rows),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
