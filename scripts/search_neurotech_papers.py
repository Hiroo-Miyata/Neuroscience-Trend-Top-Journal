#!/usr/bin/env python3
"""Search neurotechnology papers across PubMed and Europe PMC."""

from __future__ import annotations

import csv
import json
import logging
import os
import re
import time
from dataclasses import dataclass, asdict
from typing import Iterable, List, Optional
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET


DATE_START = "2021-02-09"
DATE_END = "2026-02-09"

JOURNALS = [
    "Nature",
    "Nature Neuroscience",
    "Science",
    "Cell",
    "New England Journal of Medicine",
    "The Lancet",
    "Lancet",
]

NEUROTECH_TERMS = [
    "brain-computer interface",
    "BCI",
    "BMI",
    "neural interface",
    "intracortical",
    "electrode array",
    "Utah array",
    "ECoG",
    "neuroprosthesis",
    "deep brain stimulation",
    "DBS",
    "neurostimulation",
    "cortical stimulation",
    "neural recording",
    "implant",
    "decoder",
]

SPECIES_KEYWORDS = {
    "human": ["human", "patient", "participant"],
    "macaque": ["rhesus", "macaque", "monkey", "nonhuman primate", "nhp"],
}

MODALITY_KEYWORDS = {
    "intracortical": ["intracortical", "utah array", "electrode array"],
    "ECoG": ["ecog"],
    "DBS": ["deep brain stimulation", "dbs"],
    "neurostimulation": ["neurostimulation", "cortical stimulation"],
    "BMI/BCI": ["brain-computer interface", "bci", "bmi", "neural interface"],
    "neural recording": ["neural recording", "electrode array", "intracortical"],
    "neuroprosthesis": ["neuroprosthesis", "implant"],
}

USER_AGENT = "NeurotechLiteratureSearch/1.0 (mailto:example@example.com)"


@dataclass
class Paper:
    title: str
    journal: str
    publication_date: str
    authors: str
    species: str
    neurotech_modality: str
    keywords: str
    abstract: str
    doi: str
    url: str
    source_db: str
    query_used: str


def normalize_space(text: Optional[str]) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def request_with_retries(url: str, params: dict[str, str] | None = None, timeout: int = 30) -> str:
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    backoff = 1
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            if exc.code in {429, 500, 502, 503, 504} and attempt < 4:
                logging.warning("Request failed with %s. Retrying...", exc.code)
                time.sleep(backoff)
                backoff *= 2
                continue
            raise
        except urllib.error.URLError as exc:
            if attempt < 4:
                logging.warning("Network error: %s. Retrying...", exc)
                time.sleep(backoff)
                backoff *= 2
                continue
            raise
    raise RuntimeError("Request failed after retries")


def detect_species(text: str) -> str:
    text_lower = text.lower()
    for species, keywords in SPECIES_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text_lower:
                return species
    return "unknown"


def detect_modalities(text: str) -> str:
    text_lower = text.lower()
    modalities = []
    for modality, keywords in MODALITY_KEYWORDS.items():
        if any(keyword in text_lower for keyword in keywords):
            modalities.append(modality)
    return "; ".join(sorted(set(modalities)))


def detect_keywords(text: str) -> str:
    text_lower = text.lower()
    matches = []
    for term in NEUROTECH_TERMS:
        if term.lower() in text_lower:
            matches.append(term)
    return "; ".join(sorted(set(matches)))


def format_pub_date(year: str, month: str | None, day: str | None) -> str:
    if not year:
        return ""
    month_map = {
        "jan": "01",
        "feb": "02",
        "mar": "03",
        "apr": "04",
        "may": "05",
        "jun": "06",
        "jul": "07",
        "aug": "08",
        "sep": "09",
        "oct": "10",
        "nov": "11",
        "dec": "12",
    }
    month_val = month or "01"
    if month_val.isalpha():
        month_val = month_map.get(month_val[:3].lower(), "01")
    day_val = day or "01"
    return f"{year}-{month_val.zfill(2)}-{day_val.zfill(2)}"


def query_pubmed() -> List[Paper]:
    logging.info("Querying PubMed")
    journal_query = " OR ".join(f'"{journal}"[Journal]' for journal in JOURNALS)
    term_query = " OR ".join(f'"{term}"' for term in NEUROTECH_TERMS)
    full_query = (
        f"({journal_query}) AND ({term_query}) AND (\"{DATE_START}\"[dp] : \"{DATE_END}\"[dp])"
    )
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    search_params = {
        "db": "pubmed",
        "term": full_query,
        "retmax": "500",
        "retstart": "0",
        "retmode": "json",
    }
    pmids: List[str] = []
    while True:
        response_text = request_with_retries(f"{base_url}/esearch.fcgi", params=search_params)
        data = json.loads(response_text)
        id_list = data.get("esearchresult", {}).get("idlist", [])
        pmids.extend(id_list)
        retstart = int(data.get("esearchresult", {}).get("retstart", 0))
        retmax = int(data.get("esearchresult", {}).get("retmax", 0))
        count = int(data.get("esearchresult", {}).get("count", 0))
        logging.info("PubMed search fetched %s/%s IDs", retstart + len(id_list), count)
        if retstart + retmax >= count:
            break
        search_params["retstart"] = str(retstart + retmax)
        time.sleep(0.2)

    papers: List[Paper] = []
    for i in range(0, len(pmids), 200):
        batch = pmids[i : i + 200]
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(batch),
            "retmode": "xml",
        }
        response_text = request_with_retries(f"{base_url}/efetch.fcgi", params=fetch_params)
        root = ET.fromstring(response_text)
        for article in root.findall(".//PubmedArticle"):
            pmid_node = article.find(".//PMID")
            pmid = pmid_node.text if pmid_node is not None else ""
            title = normalize_space(article.findtext(".//ArticleTitle"))
            abstract = normalize_space(" ".join(
                part.text or "" for part in article.findall(".//AbstractText")
            ))
            journal = normalize_space(article.findtext(".//Journal/Title"))
            pub_date_node = article.find(".//JournalIssue/PubDate")
            year = pub_date_node.findtext("Year") if pub_date_node is not None else ""
            month = pub_date_node.findtext("Month") if pub_date_node is not None else None
            day = pub_date_node.findtext("Day") if pub_date_node is not None else None
            publication_date = format_pub_date(year, month, day)
            authors_list = []
            for author in article.findall(".//AuthorList/Author"):
                last = author.findtext("LastName")
                fore = author.findtext("ForeName")
                if last:
                    name = f"{fore} {last}" if fore else last
                    authors_list.append(name)
            authors = "; ".join(authors_list)
            doi = ""
            for article_id in article.findall(".//ArticleId"):
                if article_id.attrib.get("IdType") == "doi":
                    doi = article_id.text or ""
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""
            text_blob = f"{title} {abstract}"
            paper = Paper(
                title=title,
                journal=journal,
                publication_date=publication_date,
                authors=authors,
                species=detect_species(text_blob),
                neurotech_modality=detect_modalities(text_blob),
                keywords=detect_keywords(text_blob),
                abstract=abstract,
                doi=doi,
                url=url,
                source_db="PubMed",
                query_used=full_query,
            )
            papers.append(paper)
        time.sleep(0.2)
    return papers


def query_europe_pmc() -> List[Paper]:
    logging.info("Querying Europe PMC")
    journal_query = " OR ".join(f'JOURNAL:"{journal}"' for journal in JOURNALS)
    term_query = " OR ".join(f'"{term}"' for term in NEUROTECH_TERMS)
    full_query = (
        f"({journal_query}) AND (TITLE_ABS:({term_query})) "
        f"AND FIRST_PDATE:[{DATE_START} TO {DATE_END}]"
    )
    base_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    cursor = "*"
    page_size = 100
    papers: List[Paper] = []
    while True:
        params = {
            "query": full_query,
            "format": "json",
            "pageSize": str(page_size),
            "cursorMark": cursor,
        }
        response_text = request_with_retries(base_url, params=params)
        data = json.loads(response_text)
        result_list = data.get("resultList", {}).get("result", [])
        for entry in result_list:
            title = normalize_space(entry.get("title"))
            abstract = normalize_space(entry.get("abstractText"))
            journal = normalize_space(entry.get("journalTitle"))
            publication_date = normalize_space(entry.get("firstPublicationDate"))
            authors = normalize_space(entry.get("authorString"))
            doi = normalize_space(entry.get("doi"))
            url_candidates = entry.get("fullTextUrlList", {}).get("fullTextUrl", [])
            url = normalize_space(url_candidates[0].get("url") if url_candidates else "")
            text_blob = f"{title} {abstract}"
            paper = Paper(
                title=title,
                journal=journal,
                publication_date=publication_date,
                authors=authors,
                species=detect_species(text_blob),
                neurotech_modality=detect_modalities(text_blob),
                keywords=detect_keywords(text_blob),
                abstract=abstract,
                doi=doi,
                url=url,
                source_db="Europe PMC",
                query_used=full_query,
            )
            papers.append(paper)
        cursor_next = data.get("nextCursorMark")
        logging.info("Europe PMC fetched %s records", len(papers))
        if not cursor_next or cursor_next == cursor:
            break
        cursor = cursor_next
        time.sleep(0.2)
    return papers


def enrich_with_crossref(papers: List[Paper]) -> None:
    logging.info("Enriching missing DOIs with Crossref")
    for paper in papers:
        if paper.doi or not paper.title:
            continue
        params = {
            "query.title": paper.title,
            "filter": f"from-pub-date:{DATE_START},until-pub-date:{DATE_END}",
            "rows": "1",
        }
        response_text = request_with_retries("https://api.crossref.org/works", params=params)
        data = json.loads(response_text)
        items = data.get("message", {}).get("items", [])
        if items:
            paper.doi = items[0].get("DOI", "")
        time.sleep(0.2)


def dedupe_papers(papers: Iterable[Paper]) -> List[Paper]:
    deduped: dict[str, Paper] = {}
    for paper in papers:
        key = paper.doi.lower() if paper.doi else f"{paper.title.lower()}-{paper.publication_date}"
        existing = deduped.get(key)
        if not existing:
            deduped[key] = paper
            continue
        for field in Paper.__dataclass_fields__:
            value = getattr(existing, field)
            new_value = getattr(paper, field)
            if not value and new_value:
                setattr(existing, field, new_value)
    return list(deduped.values())


def write_outputs(papers: List[Paper], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "papers.json")
    csv_path = os.path.join(output_dir, "papers.csv")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump([asdict(paper) for paper in papers], handle, indent=2, ensure_ascii=False)
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(Paper.__dataclass_fields__.keys()))
        writer.writeheader()
        for paper in papers:
            writer.writerow(asdict(paper))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    pubmed_results: List[Paper] = []
    europe_pmc_results: List[Paper] = []
    try:
        pubmed_results = query_pubmed()
    except (urllib.error.URLError, urllib.error.HTTPError) as exc:
        logging.error("PubMed query failed: %s", exc)
    try:
        europe_pmc_results = query_europe_pmc()
    except (urllib.error.URLError, urllib.error.HTTPError) as exc:
        logging.error("Europe PMC query failed: %s", exc)
    papers = dedupe_papers(pubmed_results + europe_pmc_results)
    if papers:
        try:
            enrich_with_crossref(papers)
        except (urllib.error.URLError, urllib.error.HTTPError) as exc:
            logging.error("Crossref enrichment failed: %s", exc)
        papers = dedupe_papers(papers)
    write_outputs(papers, "out")
    logging.info("Wrote %s papers", len(papers))


if __name__ == "__main__":
    main()
