"use strict";

const $ = (sel) => document.querySelector(sel);

/** Build DOM safely: strings become text nodes, so model-written text can never inject HTML. */
function h(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === false || value == null) continue;
    node.setAttribute(key, value === true ? "" : String(value));
  }
  for (const child of children.flat()) {
    if (child == null || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

async function api(path, body) {
  const res = await fetch(path, {
    method: body ? "POST" : "GET",
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  let data = {};
  try {
    data = await res.json();
  } catch (_) {
    /* non-JSON error body */
  }
  if (!res.ok) {
    const detail = Array.isArray(data.detail) ? "Please check what you entered." : data.detail;
    throw new Error(detail || `Request failed (${res.status})`);
  }
  return data;
}

const state = {
  sessionId: null,
  round: 0,
  dests: [],
  reactions: new Map(), // dest id -> "love" | "nope"
  images: new Map(), // dest id -> image url
  busy: false,
};

function setBusy(on) {
  state.busy = on;
  $("#go").disabled = on;
  $("#again").disabled = on;
}

function showError(message) {
  const el = state.sessionId ? $("#deck-error") : $("#ask-error");
  el.textContent = message;
  el.hidden = false;
}

function clearErrors() {
  $("#ask-error").hidden = true;
  $("#deck-error").hidden = true;
}

async function refreshMeter() {
  try {
    const { total, cap } = await api("/api/spend");
    $("#meter").textContent = `$${total.toFixed(2)} of $${cap.toFixed(0)} spent`;
  } catch (_) {
    /* the meter is a nicety, never block on it */
  }
}

function paintingFrame() {
  return h("div", { class: "frame painting" }, h("span", { class: "painting-label" }, "Painting…"));
}

async function loadPostcard(d, frame) {
  frame.className = "frame painting";
  frame.replaceChildren(h("span", { class: "painting-label" }, "Painting…"));
  try {
    const { url } = await api("/api/postcard", {
      session_id: state.sessionId,
      dest_id: d.id,
      final: $("#final").checked,
    });
    state.images.set(d.id, url);
    const img = h("img", { alt: `Painted postcard of ${d.name}, ${d.country}`, src: url });
    img.addEventListener("error", () => failFrame(d, frame, "The painting didn't load."));
    frame.replaceChildren(img);
    frame.classList.remove("painting");
  } catch (err) {
    failFrame(d, frame, err.message);
  } finally {
    refreshMeter();
  }
}

function failFrame(d, frame, message) {
  frame.className = "frame";
  const retry = h("button", { type: "button", class: "chip retry" }, "Try again");
  retry.addEventListener("click", () => loadPostcard(d, frame));
  frame.replaceChildren(h("span", { class: "painting-label" }, message), retry);
}

function renderCard(d) {
  const frame = paintingFrame();
  const love = h("button", { type: "button", class: "chip love", "aria-pressed": "false" }, "♥ Love this");
  const nope = h("button", { type: "button", class: "chip no", "aria-pressed": "false" }, "Not this");
  const go = h("button", { type: "button", class: "chip go" }, "Take me here →");

  const card = h(
    "article",
    { class: "card", "data-id": d.id },
    frame,
    h(
      "div",
      { class: "card-body" },
      h("h3", {}, d.name),
      h("div", { class: "country" }, d.country),
      h("p", { class: "tag" }, d.tagline),
      h("p", {}, d.why_it_fits),
      h("ul", {}, d.highlights.map((item) => h("li", {}, item))),
      h(
        "div",
        { class: "meta" },
        `Best months: ${d.best_months} · roughly $${d.daily_budget_usd_low}–$${d.daily_budget_usd_high} a day`,
      ),
      h("div", { class: "actions" }, love, nope, go),
    ),
  );

  function react(kind) {
    const next = state.reactions.get(d.id) === kind ? null : kind;
    if (next) state.reactions.set(d.id, next);
    else state.reactions.delete(d.id);
    card.classList.toggle("loved", next === "love");
    card.classList.toggle("nope", next === "nope");
    love.setAttribute("aria-pressed", String(next === "love"));
    nope.setAttribute("aria-pressed", String(next === "nope"));
  }
  love.addEventListener("click", () => react("love"));
  nope.addEventListener("click", () => react("nope"));
  go.addEventListener("click", () => choose(d));

  return { card, frame };
}

function renderDeck(data) {
  $("#deck-title").textContent =
    data.round === 0 ? "Four places that might be yours" : `Round ${data.round + 1}: closer to what you meant`;
  const cards = $("#cards");
  cards.replaceChildren();
  const jobs = data.destinations.map((d) => {
    const { card, frame } = renderCard(d);
    cards.append(card);
    return loadPostcard(d, frame);
  });
  $("#deck-status").textContent = "Painting your postcards. They arrive one by one.";
  Promise.allSettled(jobs).then(() => {
    $("#deck-status").textContent = "Mark the ones you love, say what to change, then wander again. Or pick one.";
  });
}

async function discover(body) {
  clearErrors();
  setBusy(true);
  const firstRound = !state.sessionId;
  $("#deck").hidden = false;
  $("#deck-status").textContent = firstRound ? "Listening to what you said…" : "Thinking about what you liked…";
  try {
    const data = await api("/api/discover", { session_id: state.sessionId, ...body });
    state.sessionId = data.session_id;
    state.round = data.round;
    state.dests = data.destinations;
    state.reactions.clear();
    state.images.clear();
    renderDeck(data);
    $("#ask").hidden = true;
    $("#deck").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (err) {
    if (firstRound) $("#deck").hidden = true;
    showError(err.message);
    $("#deck-status").textContent = "";
  } finally {
    setBusy(false);
    refreshMeter();
  }
}

async function choose(d) {
  try {
    const chosen = await api("/api/choose", { session_id: state.sessionId, dest_id: d.id });
    const url = state.images.get(d.id);
    $("#chosen-body").replaceChildren(
      url ? h("img", { class: "stage", src: url, alt: `Painted postcard of ${chosen.name}, ${chosen.country}` }) : null,
      h("h2", {}, chosen.name),
      h("div", { class: "country" }, chosen.country),
      h("p", { class: "tag" }, chosen.tagline),
      h("p", {}, chosen.why_it_fits),
      h("ul", {}, chosen.highlights.map((item) => h("li", {}, item))),
      h(
        "div",
        { class: "meta" },
        `Best months: ${chosen.best_months} · roughly $${chosen.daily_budget_usd_low}–$${chosen.daily_budget_usd_high} a day`,
      ),
      h("div", { class: "next" }, "Next, in progress: your trip, painted day by day as an illustrated storybook."),
    );
    $("#deck").hidden = true;
    $("#chosen").hidden = false;
    window.scrollTo({ top: 0, behavior: "smooth" });
  } catch (err) {
    showError(err.message);
  }
}

function restart() {
  state.sessionId = null;
  state.round = 0;
  state.dests = [];
  state.reactions.clear();
  state.images.clear();
  clearErrors();
  $("#cards").replaceChildren();
  $("#deck").hidden = true;
  $("#chosen").hidden = true;
  $("#ask").hidden = false;
  $("#vibe").focus();
}

$("#ask-form").addEventListener("submit", (event) => {
  event.preventDefault();
  if (state.busy) return;
  const avoid = $("#avoid")
    .value.split(",")
    .map((s) => s.trim())
    .filter(Boolean)
    .slice(0, 10);
  discover({
    vibe: $("#vibe").value.trim(),
    days: parseInt($("#days").value, 10) || 5,
    budget: $("#budget").value,
    avoid,
  });
});

$("#refine-form").addEventListener("submit", (event) => {
  event.preventDefault();
  if (state.busy) return;
  const named = (kind) => state.dests.filter((d) => state.reactions.get(d.id) === kind).map((d) => d.name);
  const note = $("#note").value.trim();
  $("#note").value = "";
  discover({ liked: named("love"), disliked: named("nope"), note: note || null });
});

$("#restart").addEventListener("click", restart);
$("#back").addEventListener("click", () => {
  $("#chosen").hidden = true;
  $("#deck").hidden = false;
});

refreshMeter();
