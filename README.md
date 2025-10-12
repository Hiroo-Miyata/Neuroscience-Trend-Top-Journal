# Neuroscience Trend Top Journal

This repository contains a utility script for downloading recent research articles
from three leading neuroscience journals (Nature Neuroscience, Neuron, and Cell)
and exporting the results to an Excel workbook.

## Prerequisites

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Usage

Run the fetcher to download all journal articles from the past two years and save
the results to `recent_neuroscience_articles.xlsx`:

```bash
python scripts/fetch_articles.py
```

Customise the output file or date range if needed:

```bash
python scripts/fetch_articles.py \
  --output nature_neuroscience_recent.xlsx \
  --start-date 2023-01-01 \
  --end-date 2024-12-31
```

The script contacts the Crossref API and collects, for each article:

- Title
- Authors
- Author affiliations
- Abstract (when available)
- DOI and landing-page URL
- Two heuristic category labels summarising the biological model and the primary
  experimental technique (derived from keyword searches in the title and abstract)

Use `--max-records` to limit the number of records per journal when testing without
retrieving the entire dataset.
