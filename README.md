# Wander

*A travel muse for people who don't know where to go.*

Describe a mood, however vague. Wander proposes places that are genuinely different from each other and paints each
one as a postcard in a single hand-painted style, powered by Meta's Muse models through the Meta Model API.

Built for the **Meta Global AI Developer Hackathon** (track: *Build for Your World*).
See [HACKATHON_LOG.md](HACKATHON_LOG.md) for what was built during the event and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
for the design.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env                                   # then paste MODEL_API_KEY into .env

pytest -q                                              # offline tests, no key or network needed
python scripts/smoke_test.py                           # ~3 cents: proves your key, text and image calls work
python scripts/first_slice.py --vibe "I want to feel small near something enormous, and eat well"
```

Open the `out/run-.../index.html` path that the script prints, or start the web app:

```bash
python scripts/run_ui.py          # then open http://127.0.0.1:8000
```

## What works today

* Vague vibe to four distinct destinations (Muse Spark, schema-validated JSON)
* One style-anchor image, then postcards that reuse it as a reference (Muse Image), painted card by card in the browser
* Narrowing loop: love / not this / a free-text note steers the next round, and places already shown never repeat
* Live spend meter and a hard local cap; all model text is rendered as plain text (no HTML injection)

## Repo map

```
src/wander/config.py     settings, prices, .env loader
src/wander/ledger.py     persistent spend ledger with a hard cap
src/wander/cache.py      disk cache, so identical calls are never paid twice
src/wander/client.py     Meta Model API wrapper (Spark JSON, Muse Image + reference images, retries)
src/wander/pipeline.py   vibe -> destinations -> postcards
src/wander/render.py     static HTML postcard sheet (command-line slice)
src/wander/web/          FastAPI app + postcard-deck frontend (static/index.html, style.css, app.js)
scripts/                 smoke_test.py, first_slice.py, run_ui.py
tests/                   offline tests using a stub SDK (API, pipeline, client, ledger)
```

## Cost control

Local cap `WANDER_BUDGET_USD` (default 120 of the 150 credit), a persistent ledger and a cache. The numbers are estimates;
the Billing page on dev.meta.ai is the source of truth.

## Rules compliance

Public repository, MIT license, all code written during the hackathon, synthetic demo data only, no medical advice,
no political content, no real-person imagery.
