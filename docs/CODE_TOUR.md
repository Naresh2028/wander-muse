# Code tour: how the files fit together, and why

Read this top to bottom once, then keep it open while you read the code.

## 1. The big picture in one sentence

The **browser** (HTML/CSS/JS) never talks to Meta. It talks to **our FastAPI server** over small JSON
routes; the server talks to Meta through **one client class** that counts every cent.

```
 Browser (index.html + app.js)
        |  fetch('/api/...')  JSON in, JSON out
        v
 FastAPI (web/app.py)  ---- keeps per-visitor state in memory (Session)
        |
        v
 pipeline.py / reveal/pipeline.py   <- the "what to do, in what order" layer
        |
        v
 client.py (MetaClient)  -> ledger.py (cost cap) + cache.py (never pay twice) -> api.meta.ai
```

Why this shape?
- **The API key must stay on the server.** If the browser held it, anyone could read it in DevTools.
- **One door to Meta (`client.py`)** means the budget cap, retries and caching are written once.
- **Pipelines are separate from routes**, so the same code runs from a script (`scripts/*.py`), from
  tests, and from the web server.

## 2. Backend <-> frontend connection (the part you asked about)

The connection is just **HTTP requests with JSON**. Four routes matter:

| Browser does (app.js) | Route (web/app.py) | Server does |
|---|---|---|
| "Show me places" button | `POST /api/discover` | Spark proposes destinations; session remembers them |
| each card appears | `POST /api/postcard` | Muse Image paints one postcard; file saved under `/media/...` |
| "Take me here" | `POST /api/choose` | marks the choice, returns the details |
| right after choosing | `POST /api/reveal` | postcard -> layers -> mp4; returns `{video_url, mode, layers}` |

Three ideas make it work:

1. **Session**: the server keeps a `Session` object per visitor (a dict keyed by a session id the
   browser sends back). It remembers destinations, painted image paths, finished reveals. This is why
   `/api/reveal` only needs `{session, dest_id}`, not the whole image.
2. **Media URLs**: the server saves files to disk and mounts that folder at `/media`. JSON carries only
   the URL (`/media/abc/lofoten.webp`); the browser then loads it with a normal `<img src>` or `<video src>`.
   JSON never carries image bytes, which keeps responses tiny.
3. **Slow work is not awaited by the page**: `choose()` in app.js shows the postcard immediately and
   calls `startReveal(...)` *without waiting*. The status line says "bringing it alive", a shimmer
   plays, and when the server answers, the `<img>` is swapped for a looping `<video>`.

Safety habit in app.js: elements are built with the `h()` helper, never `innerHTML`. Text from the model
is therefore always treated as text, never as HTML (blocks injection).

## 3. File by file

| File | Job | Why it exists separately |
|---|---|---|
| `config.py` | Settings from `.env` (key, model names, budget, sam_max_side) | change behaviour without touching code |
| `ledger.py` | Append-only spend log + hard cap -> `BudgetExceeded` | you can never overspend the $150 by accident |
| `cache.py` | Disk cache keyed by hash of the request | re-running costs $0; tests are free |
| `schemas.py` | Pydantic shapes for Spark's JSON | the model's output is *validated*, not trusted |
| `prompts.py` | All prompt text in one place | prompts are product design; easy to tune |
| `imaging.py` | Read/resize/encode helpers | one place for pixel plumbing |
| `client.py` | `chat_json`, `generate_image`, `segment_text` | the only code that calls Meta |
| `pipeline.py` | discover -> style anchor -> postcards | order of operations for step 1 |
| `web/app.py` | The routes above + Session | the server side of the connection |
| `web/static/*` | The page | what the user sees |
| `reveal/*` | The Reveal feature (next section) | self-contained, can fail without breaking the app |

## 4. Reveal: the flow, step by step

Trigger: user clicks "Take me here".

```
app.js  choose() -> startReveal()
   |  POST /api/reveal {session, dest_id}
   v
web/app.py  reveal()                      checks the postcard exists (409 if not), takes a lock so
   |                                      a double click doesn't pay twice, reuses a cached result
   v
reveal/pipeline.py  bring_to_life()       the conductor
   |  1. read image, resize to 1152 wide
   |  2. cut_into_layers()
   |       a. shrink copy to 768px (SAM decoding in pure Python is slow on big images)
   |       b. layers.plan_layers()   -> Muse Spark LOOKS at the picture and names 3-5 things
   |                                    with depth (sky 0.0 ... boat 0.9)
   |       c. client.segment_text()  -> SAM 3.1, one request per name, in parallel threads
   |       d. sam.parse_sam_text()   -> Meta's official parser turns SAM's text into masks
   |       e. layers.build_layers()  -> far-to-near layers; each nearer object "claims" its pixels;
   |                                    the holes behind it are inpainted so nothing tears
   |  3. motion.render_video()       -> shift each layer by (its depth) x (a looping sine), composite,
   |                                    pipe frames to ffmpeg -> reveal.mp4
   |  if anything in step 2 fails: fall back to a gentle camera move on the whole painting
   v
returns {video_url, mode, layers}  ->  app.js swaps <img> for <video loop muted>
```

### Why each decision

- **Spark names the things, SAM cuts them.** SAM 3.1 segments by a *text concept*. A model that sees
  the painting knows what is actually in it ("lighthouse", not "building"), so cuts match the image.
- **Depth is what makes it feel 3D.** Near things move more than far things (parallax). Depth comes
  from Spark; movement factor = `0.1 + 0.9 * depth`.
- **Fill behind, or it looks broken.** When the boat slides right, the pixels it covered must exist.
  We inpaint them. Without this you see a ghost boat or a hole.
- **Seamless loop.** All motion is `sin/cos` of one phase that runs 0 -> 2pi over the clip, so the last
  frame flows into the first. `base_scale` 1.07 zooms slightly so edges are never exposed.
- **Fallback instead of failure.** A moving painting is still a good result. `RevealResult.notes`
  records why, so we can fix the cause. The one exception: `BudgetExceeded` always stops.
- **Cost:** about 1 Spark call + 3-5 SAM images per postcard, roughly $0.02-0.03. Cached after that.

### Reading the files in this order

1. `reveal/pipeline.py` (the story)  2. `reveal/layers.py` (plan + build)  3. `reveal/sam.py`
4. `reveal/motion.py`  5. `reveal/debug.py` (layers contact sheet)  6. `tests/test_reveal.py`
(the tests are executable documentation of every rule above).

## 5. First live run checklist

```
pip install -e ".[dev]"
python scripts/reveal_probe.py .wander/smoke/<any postcard>.webp
```

Look at: (1) what Spark planned, (2) per-concept "SAM ... N match(es), X%" lines, (3)
`out/probe/layers_debug.jpg`, are the cut-outs clean?, (4) `out/probe/reveal.mp4`.
If it says "Fell back", the notes line tells you why. Paste me that output and we tune.
