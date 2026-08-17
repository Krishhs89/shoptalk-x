# Demo Video: Recording Script

*Satisfies the submission requirement: "Clear Video recording of working
bot." Target length: 6-9 minutes. Written to be read/followed live while
recording, not memorized — say it in your own words.*

## Before you hit record

**Which deployment to record against:** the AWS EC2 deployment is
currently unreachable (lab-account access loss — see
[deployment/aws_ec2.md](deployment/aws_ec2.md)'s status note). Record
against the **local Docker deployment** instead
([deployment/offline_production.md](deployment/offline_production.md)) —
it's the same code, same models, same behavior, just running on your Mac
instead of AWS. Mention this once, plainly, at the top of the recording
(script line included below) rather than pretending it's the cloud
deployment.

**Checklist, in order:**
1. `docker compose up -d ollama && docker compose exec ollama ollama pull llama3.1:8b-instruct-q4_0` (skip the pull if already done)
2. `docker compose up -d --build api ui mlflow`
3. Confirm health before recording anything: `curl http://localhost:8000/health` — should show all 5 models loaded (`text_embedding`, `reranker`, `clip`, `captioner`, `llm`).
4. Open `http://localhost:8501` once in a browser tab now, so the first real page-load lag doesn't happen on camera. Leave the tab open.
5. Have 2-3 real product photos ready on your Desktop for the photo-search and verification demos — anything under `data/raw/images/` works (pick ones you recognize, e.g. a chair or a bottle, since those also work for the quantity-check demo later).
6. Close Slack/email/other notification sources before recording.

**Recording tool (Mac, no install needed):** press **Cmd+Shift+5**, choose
"Record Selected Portion," frame just the browser window, click Record.
Stop with the menu-bar icon or Cmd+Shift+5 again. Do a 10-second test
clip first and check the audio levels before the real take.

---

## The script

Timestamps are approximate targets, not hard cuts — pace to what feels
natural. Bold lines are what to *say*; indented lines are what to *do* on
screen.

### 0:00 – 0:30 — Cold open: what this is

> "This is ShopTalk-X — a conversational shopping assistant. You can
> search it in plain English, search it with a photo, and it can verify
> whether a delivered package actually matches what you ordered — both
> which item, and how many. I'm running it locally on Docker right now;
> it's the identical code and models that were also verified running live
> on AWS — I'll mention that setup at the end."

  - Show the UI's landing page, sidebar with "API online" status visible.

### 0:30 – 2:00 — Natural-language search (the core deliverable)

> "The whole point of this is that keyword search fails on a query like
> this—"

  - Type: **"red sneakers for men under 150 dollars"** — verified live
    against this actual catalog build (see
    [QUALITATIVE_EVALUATION.md](QUALITATIVE_EVALUATION.md)'s example 1) —
    it correctly grounds on real matching products and respects the price
    constraint. **Don't use the project problem statement's literal "red
    shirt for men under 50 dollars" example** — this catalog's random
    10k-product sample happens to contain zero men's shirts of any color,
    so that exact query returns an honest "I don't see a match" instead
    of a product, which is correct system behavior but a flat demo
    moment. Any color+category+price query works as a stand-in — just
    confirm the category exists in your build first (`grep -i <category>`
    over `data/processed/products.parquet`, or reuse the verified example
    here).
  - Let it fully answer. Point at the answer text, then the product
    cards below it.

> "Notice it's not just a list — it's a grounded answer citing a real
> product, with its ID, price, and a reason. Every hit below has a
> product ID, a category, and both a retrieval score and a rerank score —
> that's the two-stage retrieval pipeline: a fast approximate search over
> the whole catalog, then a slower, more accurate re-ranking of just the
> top candidates."

  - Point out the latency caption line (stage1/rerank/llm/total ms) if
    visible.

### 2:00 – 3:00 — Follow-up / conversational history

> "It also remembers the conversation. Let me ask a follow-up without
> repeating context."

  - Type: **"what about black sneakers for men instead?"** — verified
    live (example 3 in [QUALITATIVE_EVALUATION.md](QUALITATIVE_EVALUATION.md)).
    **Avoid pure-pronoun follow-ups like "anything cheaper than that?"**
    with nothing else restated — tested live and it genuinely fails:
    retrieval only ever embeds the current turn's raw text (confirmed in
    `api/main.py`), so a follow-up needs to restate the subject
    ("sneakers," "that jacket," etc.) for retrieval to find the right
    candidates, even though the LLM's *answer* does correctly use full
    conversation history. Keep the follow-up self-contained in what it's
    asking for, and it demonstrates the history feature working
    correctly — carrying over the earlier "under $150" constraint without
    restating it.
  - Let it answer, pointing out it understood "instead" / carried over the
    prior turn.

> "This is backed by real conversation history, not just prompt stuffing
> — I can start a new conversation, or come back later and resume this
> exact one by name."

  - Briefly show the sidebar's past-conversations list / "New
    conversation" button.

### 3:00 – 4:00 — Photo search

> "You can also search with a photo instead of words — say you saw
> something you liked and want to find it in the catalog."

  - Upload one of the prepared product photos via the chat tab's photo
    uploader.
  - Let it caption the photo and return results.

> "Under the hood this uses CLIP for the image embedding and BLIP to
> caption it into a pseudo-query, so it gets the same grounded,
> conversational answer treatment as a text search."

### 4:00 – 5:30 — Order verification (match / mismatch / suspect)

> "Now the trust side of the app. Say a package arrives and you want to
> confirm it's actually what you ordered."

  - Switch to the "Verify order" tab.
  - Pick a real item from the dropdown, upload its actual catalog photo
    (a genuine match) — show the green **match** result with its
    confidence score.
  - Then repeat with a **mismatched** photo (upload a photo of a
    different, unrelated item against the same order) — show the red
    **mismatch** result.

> "There's a third outcome too — 'suspect' — for anything close to the
> decision boundary. That's routed to a human, on purpose: this system
> never makes a confident fraud accusation on a borderline case."

### 5:30 – 6:45 — Quantity check

> "The newest feature: checking not just *what* arrived, but *how many*."

  - Switch to the "Verify quantity" tab.
  - Pick a chair (or another COCO-recognizable category — bottle, book,
    cup) from the dropdown, claim a quantity, upload the photo.
  - Show a real result — match or a mismatch/suspect if the claimed count
    doesn't match what's visible.

> "This uses a pretrained object detector — it only recognizes about 80
> general object types, not this catalog's specific categories. So for
> most products it'll honestly say 'unsupported' instead of guessing."

  - Optionally: quickly demo the `unsupported` case with a product
    outside COCO's classes (e.g. a phone case), to show the honest
    failure mode on camera, not just the happy path.

### 6:45 – 7:30 — Feedback loop

> "Every answer has thumbs up/down feedback, which is logged and, in a
> full production loop, would feed the automated retraining pipeline."

  - Click thumbs up/down on an earlier answer.

### 7:30 – 8:30 — What's under the hood (brief, no need to show code live)

> "A few things worth knowing about what's behind this screen: the
> embedding model powering search was fine-tuned specifically on this
> catalog — that alone lifted Recall@10 by about 12 percentage points
> over the pretrained baseline, measured, not estimated. There's also an
> automated retraining pipeline, built on Airflow, that re-trains,
> re-evaluates against the current production model, and only promotes a
> new version if it actually proves better — and that pipeline has
> genuinely run end-to-end, not just been designed. The whole stack is
> containerized with Docker and was deployed and verified live on AWS
> EC2; that instance is temporarily down due to a training-lab account
> issue unrelated to the code, which is why I'm demoing locally today."

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
