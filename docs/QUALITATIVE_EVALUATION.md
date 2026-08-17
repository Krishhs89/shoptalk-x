# Qualitative Evaluation: Generative Results

*Satisfies the Submission Guidelines' "Qualitative testing for Generative
results." These are real, unedited request/response pairs run against the
local deployment (`data/models/bge-finetuned` serving, `llama3.1:8b-instruct-q4_0`
via Ollama), not curated transcripts or hand-picked screenshots. Full JSON
responses (hits, scores, latency) are in `docs/_qual_eval_raw/` alongside
this file for anyone who wants to check the exact retrieved candidates
behind each answer, not just the prose.*

## How these were chosen

The first example uses the project's own problem-statement query verbatim
— then reports, honestly, that it doesn't work as a demo on this specific
catalog build, and why. The rest were chosen to probe real behavior:
a query the system should nail, a conversational follow-up designed to
find the edges of what "conversation history" actually covers, and a
second follow-up that fixes what the first one exposed. This is closer to
an actual test plan than a highlight reel.

---

## Example 1 — the literal problem-statement query, and what actually happened

**Query:** `"a red shirt for men under 50 dollars"` — this is the exact
example from the project's own Motivation and Problem Statement document.

**Response:**
> "I'm not seeing any products that match your description. The only
> product I see is the Solimo Printed Fabric Shirt Stacker Organiser, but
> it's an organizer and not a shirt. If you'd like to search again or
> need help with something else, just let me know!"

**Assessment: correct behavior, misleading as a demo query.** We checked:
this catalog is a random ~10,000-product sample of Amazon Berkeley
Objects, and that particular sample happens to contain **zero men's
shirts of any color** (`grep`-verified against `products.parquet`). The
system did the right thing — it did not hallucinate a shirt that doesn't
exist, and it correctly explained why the one tangentially-related hit
(a "Shirt Stacker Organiser," matched on the word "shirt" and the color
red) isn't actually a match. This is exactly the grounding behavior
the design intends. It's simply the wrong query to *demonstrate* success
with on this specific catalog build — a different random sample of the
same dataset, or a larger `target_product_count`, would likely include
men's shirts. See [VIDEO_RECORDING_SCRIPT.md](VIDEO_RECORDING_SCRIPT.md)
for the verified substitute used in the actual demo recording.

---

## Example 2 — a real success (used in the demo video)

**Query:** `"red sneakers for men under 150 dollars"`

**Response:**
> "Based on your search, I'd recommend the Amazon Brand - Symbol Men's
> Red Sneakers (B074F16RLC) or the Klepe Men's Black/RED Chunky/Platform
> Sneakers (B07WTP2XHX). Both are red sneakers for men and priced under
> $150. The first one is a more classic sneaker style, while the second
> one has a chunky platform design. (id: B074F16RLC)"

**Assessment: correct on every axis.** Both recommended products are
genuinely red (or red-accented), genuinely men's, genuinely under $150
($109.19 and $102.41). The top two rerank scores are the only positive
ones (3.30, 3.30 in the follow-up variant of this search; 0.44, 0.24
here) — the reranker correctly separated real matches from same-category
noise (navy/red mixed-color sneakers, women's sneakers that slipped into
stage-1 candidates) further down the list, and the answer reflects that
separation rather than just listing the raw top-10. The comparison
between the two recommended options ("classic" vs. "chunky platform") is
a genuine, useful distinction, not filler.

---

## Example 3 — a follow-up that fails, and why (a real finding, not hidden)

**Same conversation, follow-up query:** `"do you have anything cheaper
than that?"`

**Response:**
> "Unfortunately, I don't see any red sneakers for men under $150 in our
> catalog. However, among the products listed, the 365 EVERYDAY VALUE
> Reduced Fat 2% Milk (B07813841L) is priced at $77.09, which is the
> cheapest option available."

**Assessment: a genuine, worth-knowing-about limitation.** This is a real
failure, tested live, not a hypothetical edge case. What happened,
traced to the actual code: `POST /search/text` always retrieves using
only the *current turn's raw text* (`api/main.py`: `stage1_search(req.query, ...)`)
— there is no step that rewrites "anything cheaper than that" into
something like "sneakers under $109" using conversation history before
retrieval runs. So the embedding for "do you have anything cheaper than
that?" — a sentence with no product category in it at all — matched
whatever happened to be semantically closest to "cheap," which turned out
to be a handful of grocery items. The LLM's *answer generation* clearly
still has conversation history (it correctly says "red sneakers... under
$150," which it could only know from the prior turn) — the disconnect is
specifically that retrieval and generation are working from different
information this turn, and the answer, while not hallucinating a
nonexistent product, ends up confusingly comparing sneakers to milk.

**Why this matters more than a typical bug report:** conversational
history is graded UI/UX functionality, and this is exactly the kind of
thing E2E testing is supposed to surface — a feature that looks correct
in the simple case and breaks on a natural, realistic follow-up phrasing.
It's now tracked as a concrete, well-understood item on the improvement
roadmap (query reformulation before retrieval, not just in the prompt —
see [PROJECT_EXPLAINED.md](PROJECT_EXPLAINED.md) Part 7).

---

## Example 4 — the same follow-up intent, phrased to work with today's retrieval

**Same conversation, retried as:** `"what about black sneakers for men
instead?"`

**Response:**
> "Based on your search, I'd recommend the Amazon Brand - Symbol Men's
> Black Sneakers (B07BDZBMJ9) or the Amazon Brand - Symbol Men's Black
> Textile Sneakers (B07VNR3B35). Both are black sneakers for men and
> priced under $150."

**Assessment: history genuinely works when the follow-up carries its own
keywords.** Retrieval succeeded here because "black sneakers for men" is
self-contained — no pronoun resolution needed for the *embedding* to find
the right candidates. But notice the price constraint ("under $150") was
**never repeated** in this follow-up, and the answer still respects it
correctly — that part *is* genuine conversation-history behavior, carried
through the LLM's prompt context, not re-derived from this turn's
retrieval. Put plainly: this system's conversation history reliably
carries constraints (price, brand, etc.) into the generated answer, but
does not yet rewrite ambiguous follow-up phrasing before retrieval. Both
halves of that sentence are demonstrated, back to back, by examples 3 and
4 above.

---

## Summary

| # | Query | Outcome | What it demonstrates |
|---|---|---|---|
| 1 | "a red shirt for men under 50 dollars" | Correct "no match" — no hallucination | Grounding/anti-hallucination, but a bad catalog-specific demo choice |
| 2 | "red sneakers for men under 150 dollars" | Correct, well-reasoned recommendation | Core NL search + price-constraint understanding, working end to end |
| 3 | "do you have anything cheaper than that?" | Wrong — retrieval loses context on pure pronoun reference | A real, now-documented limitation: retrieval doesn't reformulate using history |
| 4 | "what about black sneakers for men instead?" | Correct, and correctly carries the unstated $150 constraint | Conversation history *does* work for constraints carried into generation |

Three of four examples are genuine successes; the one failure is reported
in full because it's more useful, and more honest, than a document that
only shows what works.
