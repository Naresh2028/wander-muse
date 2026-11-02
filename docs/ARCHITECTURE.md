# Wander: architecture

**One-line pitch:** you arrive not knowing where to go. Wander shows you places as painted postcards, learns from
how you react, then paints your whole trip as an illustrated storybook.

**Target prize:** Best Creative Build (judged on originality, execution, and how effectively Muse Image / Muse Wonder
is used in the final experience). Track: *Build for Your World*.

## Flow

```
 vibe (text, later voice)
        |
        v
 [1 INTAKE]  Spark -> TripBrief (structured JSON)
        |
        v
 [2 DISCOVER] Spark -> 4 distinct Destinations (JSON, each with a paintable "visual_scene")
        |        Muse Image: 1 style-anchor card, then 4 postcards that use the anchor as a reference
        v
 [3 NARROW]  user reacts ("more mountains, fewer crowds", love / not this)
        |        Spark folds reactions into the brief -> new round of postcards  (loop until one is chosen)
        v
 [4 REVEAL]  chosen postcard comes alive
        |        SAM 3.1 cuts sky / mountain / boat layers -> parallax loop (ffmpeg) -> short mp4
        v
 [5 PLAN]    Spark (+ web search grounding) -> day-by-day itinerary (JSON, with sources)
        |        Muse Image: one panel per day, anchored on the chosen postcard -> storybook page
        v
 [6 SHARE]   static HTML storybook + mp4, every factual claim marked "verify before booking"
```

Phases 1 to 3 are implemented (`src/wander/pipeline.py`, `src/wander/web/`). Everything else builds on the same client,
cache and ledger.

Web flow: the browser calls `POST /api/discover` (Spark picks places while the style anchor is ensured in parallel),
then fires one `POST /api/postcard` per card concurrently, so postcards appear as each one finishes. Sessions are kept
in server memory, which is enough for a demo; the browser only ever receives public card fields (the painting prompt
stays on the server).

## Where each model is used

| Model | Role in Wander | When |
|---|---|---|
| **Muse Image 1.0** ($0.01/image) | The creative core: style anchor, postcards, day panels, "make it night / emptier / cheaper" edits. Built-in search grounding gives real places accurate looks. | Slice 1 (done), 2, 5 |
| **Muse Spark 1.3** | The brain: structured JSON for briefs, destinations and itineraries; web search grounding; vision check that a postcard matches its scene. | Slice 1 (done), 3, 5 |
| **SAM 3.1** ($2.50 / 1k images) | Cuts foreground, midground and sky from a postcard so it can move (parallax). This is how the postcard "comes alive". | Day 3 |
| **Muse Voice Transcribe** ($0.18/h) | Speak your vibe instead of typing it. | Day 2, optional |
| **Muse Wonder** | A coding / long-running agent model, **not** an image or video model. Optional as the itinerary planner; mainly useful through **Muse Code** for writing this repo faster. | Optional |
| **Muse Glimmer / Muse Spark Open** | Optional open-weight step (for example preference updates) served on Baseten or Modal. Would also target the *Best Build on Meta Open Weight Models* prize; the infra partner must verify it. On Baseten use `meta-models/muse-spark-open-1026`. | Stretch |

## Why this should read as creative, not generic

* **Originality:** the premise is *not knowing* where to go, and the interface is a painted deck of places that you
  react to, not a form with filters.
* **Execution:** one locked art direction across every asset (anchor image + same style words in every prompt),
  motion from real segmentation masks, no text rendered inside images (typography is added in HTML).
* **Model use:** Muse Image is on screen in every step, and the cookbook's *anchored image series* technique keeps the
  look consistent from postcard to storybook.

## Cost control (verified prices, 2026-10-29)

| Item | Price | Planned use | Est. cost |
|---|---|---|---|
| Muse Image | $0.01 / image | ~1,500 draft + ~300 final images | ~$18 |
| Muse Spark 1.3 | $1.25 in / $4.25 out per Mtok (reasoning counts as output) | ~600 calls | ~$10 to $25 |
| SAM 3.1 | $2.50 / 1k images | ~300 images | ~$1 |
| Voice | $0.18 / hour | a few minutes | ~$0 |
| Reserve | | | the rest of the $150 |

The real constraint is time, not money. Safeguards that stay on anyway: local hard cap (`WANDER_BUDGET_USD`, default 120),
a persistent ledger (`.wander/ledger.jsonl`), and a disk cache so identical requests are never paid twice.
`muse-spark-1.3-contributor` is far cheaper but its data is used to improve Meta's products, so it is not the default.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Style drifts between images | Anchor reference on every call, same style words, compare outputs, regenerate outliers |
| Spark states wrong facts | Prompt forbids live prices/hours, search grounding for itineraries, visible "verify" footer |
| Reads as a generic trip planner | Visual-first flow, narrowing loop, animated reveal, storybook output; keep logistics minimal |
| API surprise on first live call | `scripts/smoke_test.py` isolates text, plain image and tool-controlled image calls |
| Running out of time | Feature freeze Nov 2, demo video Nov 3, submit early on Nov 4 |
| Rules | Public repo, provenance log, synthetic data only, no banned categories |

## Schedule (today is Oct 29; submission closes Nov 4, 9:00 PM Eastern, which is 7:30 AM IST on Nov 5)

| Date | Goal |
|---|---|
| Oct 29 | Smoke test passes, slice 1 produces a real postcard sheet, repo public, first commits |
| Oct 30 | Small web UI (Gradio or Streamlit) with the react-and-narrow loop; optional voice input |
| Oct 31 | Reveal: SAM masks to parallax layers to a looping mp4 |
| Nov 1 | Itinerary with grounding and sources, plus anchored day-panel storybook |
| Nov 2 | Polish, three showcase runs, README screenshots. **Feature freeze at the end of the day** |
| Nov 3 | Record the 3-minute demo, test from a clean clone, ask any last questions |
| Nov 4 | Submit by midday Eastern (about 10:30 PM IST on Nov 4), not at the deadline |

## Demo video outline (3 minutes)

1. 0:00 The problem: "I want to go somewhere but I don't know where."
2. 0:20 Type or speak a vague vibe.
3. 0:45 Four postcards appear in one painted style.
4. 1:10 React ("less crowded, more water") and watch the next round change.
5. 1:35 Pick one: the postcard comes alive.
6. 2:00 The trip appears as an illustrated storybook.
7. 2:30 What was built this week, which Meta models do what.
8. 2:50 Close.
