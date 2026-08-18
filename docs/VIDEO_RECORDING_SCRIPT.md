# Demo Video: Recording Script

*Satisfies the submission requirement: "Clear Video recording of working
bot." Target length: 6-9 minutes.*

**How to use this doc:** read the whole thing once before you record, so
the flow is familiar. While actually recording, don't read this —
glance at the **Quick reference table** (Part 2) between clicks instead;
it's built to be skimmed in a second, not read aloud. Part 3 below it has
the fuller narration if you want more to say, but paraphrasing the
table's "Say" column in your own words is enough.

---

## Part 1 — Before you hit record

### Which URL to record against

**Recommended: the public URL** — `https://suitor-kerchief-rut.ngrok-free.dev`
(or whatever your current ngrok/tunnel URL is). Recording against a link
that's actually reachable from outside your machine is a stronger opening
than a `localhost` screen — it shows this is a real, hosted product, not
just code running on your laptop. Say once, plainly, near the start: *"the
original AWS deployment is temporarily down due to a training-lab account
issue unrelated to the app itself, so I'm showing it through a tunnel to
my machine right now"* — then move on, don't dwell on it.

**Fallback: `http://localhost:8501`** — if the tunnel isn't live when you
sit down to record, use this instead (see
[deployment/offline_production.md](deployment/offline_production.md)).
Functionally identical either way.

### Checklist, in order

1. Confirm the stack is up: `curl http://localhost:8000/health` — should
   show all 5 models loaded (`text_embedding`, `reranker`, `clip`,
   `captioner`, `llm`). If not: `docker compose up -d --build api ui
   ollama mlflow` and wait ~30s for models to warm.
2. **If recording the public URL**, confirm the tunnel is actually up:
   `curl -o /dev/null -w "%{http_code}\n" https://<your-url>/_stcore/health`
   should print `200`. If it's not running: `ngrok http 8501 --url
   https://<your-static-domain>`.
3. Open the URL you're recording once in a browser tab *now*, so the
   first real page-load lag doesn't happen on camera. Leave the tab open.
4. Have 2-3 real product photos ready on your Desktop for the photo-search,
   verification, and quantity demos — anything under `data/raw/images/`
   works; pick ones you can identify by category (a chair, a bottle) since
   those double as quantity-check demo material later.
5. Close Slack/email/other notification sources before recording.
6. Zoom your browser to ~110-125% before recording — small UI text reads
   poorly in a screen recording; check it looks right in your first test
   clip.

**Recording tool (Mac, no install needed):** press **Cmd+Shift+5**, choose
"Record Selected Portion," frame just the browser window, click Record.
Stop with the menu-bar icon or Cmd+Shift+5 again. Do a 10-second test clip
first and check both audio levels and text legibility before the real
take.

---

## Part 2 — Quick reference table (glance at this while recording)

| Time | Do | Say (short — paraphrase, don't read verbatim) |
|---|---|---|
| 0:00 | Show landing page, sidebar "API online" | "ShopTalk-X — search in plain English or by photo, plus order and quantity verification." |
| 0:15 | *(if on the public URL)* Point at the address bar | "This is a real hosted URL — AWS is temporarily down for an unrelated lab-account reason, so I'm tunneling to my machine right now." |
| 0:30 | Type: **"red sneakers for men under 150 dollars"** | "Keyword search can't handle constraints like this — watch." |
| 1:00 | Point at the answer, then the product cards | "Grounded answer citing a real product ID and price — not a hallucination." |
| 1:20 | Point at a product card's category/price/score row, then the response-time line under the answer | "Two-stage retrieval — fast approximate search, then a more accurate re-rank — and every response now reports its own timing." |
| 2:00 | Type follow-up: **"what about black sneakers for men instead?"** | "It remembers context — this carries over the $150 limit without me repeating it." |
| 2:40 | Show sidebar past-conversations list | "Real conversation history, not just prompt stuffing — I can resume this later by name." |
| 3:00 | Upload a product photo in the chat tab | "You can search by photo instead of words." |
| 3:40 | Point at the pseudo-query/caption in the answer | "CLIP embeds the photo, BLIP captions it — same grounded-answer treatment as text search." |
| 4:00 | Switch to **Verify order** tab; pick an item + its real catalog photo | "Confirm a delivered item matches what was ordered." |
| 4:30 | Show the green **MATCH** badge + confidence + response time | "Every verification call reports its own latency too." |
| 4:50 | Repeat with a mismatched photo | Show the red **MISMATCH** badge. |
| 5:10 | *(mention, no need to demo live)* | "There's a third outcome, 'suspect,' for borderline cases — routed to a human, never an automatic accusation." |
| 5:30 | Switch to **Verify quantity** tab; pick a chair/bottle/book, claim a quantity, upload photo | "Checks not just *what* arrived, but *how many*." |
| 6:00 | Show the verdict badge + detected count + response time | "Pretrained object detector — honest about only recognizing ~80 general object types." |
| 6:20 | *(optional)* Try a phone case → **UNSUPPORTED** | "For anything outside those 80 classes, it says so instead of guessing." |
| 6:45 | Click 👍/👎 on an earlier answer | "Feedback is logged — feeds a future retraining loop." |
| 7:00 | *(no need to show code)* | "The embedding model was fine-tuned on this catalog — measured +12 points Recall@10, not estimated. There's also an automated Airflow retraining pipeline that only promotes a new model if it actually proves better, and it's genuinely run end to end, not just been designed." |
| 8:00 | Close on the landing page | "That's ShopTalk-X — search, verification, quantity checks, and a retraining loop that's actually been exercised for real. Thanks for watching." |

---

## Part 3 — Fuller narration (optional, if you want more to say than the table)

### 0:00 – 0:30 — Cold open

> "This is ShopTalk-X — a conversational shopping assistant. You can
> search it in plain English, search it with a photo, and it can verify
> whether a delivered package actually matches what you ordered — both
> which item, and how many."

### 0:30 – 2:00 — Natural-language search (the core deliverable)

> "The whole point of this is that keyword search fails on a query like
> this—"

Type: **"red sneakers for men under 150 dollars"** — verified live against
this actual catalog build (see
[QUALITATIVE_EVALUATION.md](QUALITATIVE_EVALUATION.md)'s example 2) — it
correctly grounds on real matching products and respects the price
constraint. **Don't use the project problem statement's literal "red shirt
for men under 50 dollars" example** — this catalog's random 10k-product
sample happens to contain zero men's shirts of any color, so that exact
query returns an honest "no match" instead of a product: correct system
behavior, but a flat demo moment. Any color+category+price query works as
a stand-in — just confirm the category exists first (`grep -i <category>`
over `data/processed/products.parquet`), or reuse the verified example
above.

> "Notice it's not just a list — it's a grounded answer citing a real
> product, with its ID, price, and a reason. Every hit below has a
> product ID, a category, a retrieval score, and a rerank score — that's
> the two-stage retrieval pipeline: a fast approximate search over the
> whole catalog, then a slower, more accurate re-ranking of just the top
> candidates. And now every response — search, verification, quantity
> checks — reports exactly how long it took."

### 2:00 – 3:00 — Follow-up / conversational history

> "It also remembers the conversation. Let me ask a follow-up without
> repeating context."

Type: **"what about black sneakers for men instead?"** — verified live
(example 4 in [QUALITATIVE_EVALUATION.md](QUALITATIVE_EVALUATION.md)).
**Avoid pure-pronoun follow-ups like "anything cheaper than that?"** with
nothing else restated — tested live and it genuinely fails: retrieval only
ever embeds the current turn's raw text, so a follow-up needs to restate
its subject for retrieval to find the right candidates, even though the
LLM's *answer* does correctly use full conversation history (see that same
doc's example 3 for the documented failure, if asked about it directly —
it's a known, honestly-reported limitation, not something to hide).

> "This is backed by real conversation history, not just prompt stuffing
> — I can start a new conversation, or come back later and resume this
> exact one by name."

### 3:00 – 4:00 — Photo search

> "You can also search with a photo instead of words — say you saw
> something you liked and want to find it in the catalog. Under the hood
> this uses CLIP for the image embedding and BLIP to caption it into a
> pseudo-query, so it gets the same grounded, conversational answer
> treatment as a text search."

### 4:00 – 5:30 — Order verification (match / mismatch / suspect)

> "Now the trust side of the app. Say a package arrives and you want to
> confirm it's actually what you ordered." Pick a real item + its actual
> catalog photo → green **MATCH**. Repeat with an unrelated photo → red
> **MISMATCH**. "There's a third outcome too — 'suspect' — for anything
> close to the decision boundary. That's routed to a human, on purpose:
> this system never makes a confident fraud accusation on a borderline
> case."

### 5:30 – 6:45 — Quantity check

> "The newest feature: checking not just *what* arrived, but *how many*."
> Pick a chair/bottle/book, claim a quantity, upload the photo. "This uses
> a pretrained object detector — it only recognizes about 80 general
> object types, not this catalog's specific categories. So for most
> products it'll honestly say 'unsupported' instead of guessing." Optional:
> demo the `unsupported` case with a phone case, to show the honest
> failure mode on camera, not just the happy path.

### 6:45 – 7:30 — Feedback loop

> "Every answer has thumbs up/down feedback, which is logged and, in a
> full production loop, would feed the automated retraining pipeline."

### 7:30 – 8:30 — What's under the hood

> "A few things worth knowing: the embedding model powering search was
> fine-tuned specifically on this catalog — measured +12 percentage
> points Recall@10 over the pretrained baseline, not estimated. There's
> an automated retraining pipeline, built on Airflow, that re-trains,
> re-evaluates against the current production model, and only promotes a
> new version if it actually proves better — and that pipeline has
> genuinely run end-to-end. The whole stack is containerized with Docker,
> the API has real interactive documentation at `/docs`, and this was
> deployed and verified live on AWS EC2 — that instance is temporarily
> down due to a training-lab account issue unrelated to the code, which
> is why I'm showing it through a tunnel today."

### 8:30 – 9:00 — Close

> "That's ShopTalk-X — conversational search, photo search, order and
> quantity verification, and a retraining loop that's actually been
> exercised for real. Thanks for watching."

---

## After recording

- Trim dead air at the start/end (QuickTime Player: File → Export, or
  just leave it — a couple seconds of silence doesn't hurt).
- Save as `demo_video.mp4` (or similar) and reference it from your
  submission — this repo doesn't check video files into git (large
  binary, no reason to version it), so keep it alongside the repo, not
  inside it, unless your submission process specifically wants it
  committed.
- If you want a second, more technical cut for a panel/interview
  audience rather than a pure feature demo, see
  [PRESENTATION_OUTLINE.md](PRESENTATION_OUTLINE.md) — same app, deeper
  architecture talking points, slide-by-slide.
