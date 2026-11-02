# Hackathon provenance log

The rules require judges to see clearly what was built **during** the hackathon (Oct 27 - Nov 4, 2026).

**Everything in this repository was written during the hackathon, from an empty directory.**
No code, data, images or assets from earlier projects or employers are used. All demo cases are synthetic.

Models and services: Muse Spark 1.3, Muse Image 1.0 (and later SAM 3.1, Muse Voice Transcribe) via Meta Model API.
Code assistance: Claude (Anthropic) was used as a pair-programming assistant for planning and scaffolding.

## Build log (update every day, and keep commits small and frequent)

| Date (2026) | What was built |
|---|---|
| Oct 29 (later) | Web UI: FastAPI backend (discover, per-card postcard painting, choose, spend meter), react-and-narrow loop (love / not this / free-text note feeds the next Spark round, already-shown places are never repeated), postcard-deck frontend in vanilla JS, browser-verified end to end against a stub API. First successful live calls to Muse Spark and Muse Image. |
| Oct 29 | Repo created. Config, spend ledger with hard cap, disk cache, Meta client (Spark structured output + Muse Image with reference images), prompts, first pipeline slice, HTML postcard sheet, smoke test, offline tests. |
