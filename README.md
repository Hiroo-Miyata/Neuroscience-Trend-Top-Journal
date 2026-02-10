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

## Abstract-Aware Pruning (LLM + Fallback)

After `out/papers.csv` is generated, you can prune with LLM-assisted abstract classification:

```bash
python scripts/prune_with_abstracts.py
```

### What it does
- Filters to the target journals and the last 5 years relative to today.
- Uses title + abstract to classify: species, neurotechnology, modality, study type, and relevance.
- Writes:
  - `out/papers_pruned.csv` (accepted)
  - `out/papers_review_queue.csv` (uncertain)
  - `out/papers_rejected.csv` (rejected with reasons)
  - `out/papers_llm_labels.json` (structured labels per paper)

### OpenAI configuration
- Set your API key: `export OPENAI_API_KEY="..."`
- Model defaults to `gpt-4o-mini`. Override with `--model`.
- Deterministic settings: temperature=0 and strict JSON output.

### Batch mode (scaling)
Use the Batch API for larger runs:

```bash
python scripts/prune_with_abstracts.py --batch
```

Batch runs are stored under `out/batch_runs/`.

### Rule-based fallback (no API key)
If `OPENAI_API_KEY` is not set, the script uses a conservative keyword-based
classifier and marks ambiguous cases as `uncertain`.

### Cost notes
LLM cost depends on the number of papers and abstract length. As a rough guide,
the default model typically costs cents per paper. Use `--max-papers` to estimate
costs on a subset before scaling up.

### Tuning thresholds
Edit `scripts/prune_with_abstracts.py`:
- `NEUROTECH_KEYWORDS`, `SPECIES_KEYWORDS`, `MODALITY_KEYWORDS` for the fallback.
- `TARGET_JOURNALS` or the acceptance rules in `decision_bucket` to tighten/relax filtering.

### Privacy notes
Abstracts are sent to OpenAI only when `OPENAI_API_KEY` is set. If your data
cannot leave your environment, use the rule-based fallback mode.

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
