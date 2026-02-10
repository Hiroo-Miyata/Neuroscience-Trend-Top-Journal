# Neurotechnology Literature Search Pipeline

This repository provides a reproducible literature search pipeline for neurotechnology papers (human or macaque) from **2021-02-09 to 2026-02-09** limited to:

- Nature
- Nature Neuroscience
- Science
- Cell
- New England Journal of Medicine
- The Lancet

Primary sources: PubMed/Entrez and Europe PMC, with Crossref used to enrich missing DOI metadata.

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/search_neurotech_papers.py
```

Outputs are written to `./out/papers.csv` and `./out/papers.json`.

## Output Schema

Each record contains:

- `title`
- `journal`
- `publication_date`
- `authors`
- `species` (human/macaque/unknown)
- `neurotech_modality` (intracortical, ECoG, DBS, neurostimulation, BMI/BCI, neural recording, neuroprosthesis)
- `keywords` (matched neurotech terms)
- `abstract`
- `doi`
- `url`
- `source_db`
- `query_used`

## Adjusting Keywords, Journals, or Dates

Edit the constants at the top of `scripts/search_neurotech_papers.py`:

- `JOURNALS` for the journal list
- `NEUROTECH_TERMS` for keyword filtering
- `DATE_START` / `DATE_END` for the date range

Species detection and modality classification are rule-based and can be adjusted via
`SPECIES_KEYWORDS` and `MODALITY_KEYWORDS`.

## Notes

- The script uses public APIs (PubMed E-utilities, Europe PMC REST, Crossref).
- It includes pagination, retries, and basic rate limiting.
- If you need to constrain API usage further, increase the `time.sleep()` delays.
