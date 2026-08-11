# Model Card: Quantity/Count Validation

## Intended use
Given a photo of received items and a customer's claimed quantity (e.g.
"I ordered 3 and only got 2"), checks whether that many instances of the
product are actually visible in the photo — a count-dispute analog to the
existing [visual verification head](verification.md)'s wrong-item check.
Like that feature, a close-but-not-exact count routes to `suspect` (human
review), never an automatic accusation or rejection.

## Architecture
**Pretrained only — no custom-trained counting model.** There is no
labeled counting dataset for this catalog (counting how many of a given
product appear in a photo isn't something ABO's listings data captures),
so building one was explicitly out of scope. Instead: Ultralytics
**YOLOv8n**, pretrained on COCO (80 general object classes), run directly
via `shoptalk.counting.count.count_detections` — the number of detected
boxes of the matched class, above a confidence threshold
(`counting.conf_threshold`, default 0.35), is the count.

`shoptalk.counting.coco_classes.resolve_coco_class` maps a catalog
product's category or name to one of COCO's 80 classes (e.g.
`CHAIR → "chair"`, an item named "... Water Bottle ..." → `"bottle"`).
When no mapping exists, `verify_quantity` returns `verdict="unsupported"`
rather than guessing.

## Training data / pretrained source
None trained here. Weights are the official Ultralytics YOLOv8n COCO
checkpoint (`yolov8n.pt`), auto-downloaded on first use to
`data/counting/` (see `configs/config.yaml`'s `counting:` section) — same
"resolve on first use" pattern as the other pretrained models in this repo
(open_clip, sentence-transformers base checkpoints).

## Eval results
No catalog-specific eval run (would require ground-truth counts per photo,
which don't exist for this dataset — the same gap that ruled out training a
custom model). Confidence in this feature rests on YOLOv8's published COCO
benchmarks (mAP), not a ShopTalk-X-specific number. Treat this as a
**best-effort assist for the ~20% of the catalog whose category or name
maps to a COCO class**, not a calibrated production classifier.

## Limitations / failure modes
- **Coverage is narrow by construction.** This catalog spans ~50
  fine-grained categories (`CELLULAR_PHONE_CASE`, `FINERING`,
  `HARDWARE_HANDLE`, ...); COCO's 80 classes cover generic
  household/street objects. Most catalog items — including the single
  largest category, `CELLULAR_PHONE_CASE` (over half the catalog) — have
  no COCO equivalent and return `"unsupported"`. See
  `coco_classes.CATEGORY_TO_COCO_CLASS` / `KEYWORD_TO_COCO_CLASS` for the
  exact covered set (seating, bags, books, bottles/cups/bowls, a handful of
  COCO food classes, and a few miscellaneous accessories).
- **No custom fine-tuning.** A category-specific or catalog-specific
  fine-tune of YOLO (or a purpose-built counting head, e.g. density-map
  regression) would meaningfully widen coverage and accuracy — not
  attempted here per the stated scope (no labeled counting data).
- **Occlusion/stacking undercounts.** Overlapping items (e.g. a tightly
  packed multipack) can merge into fewer detected boxes than are actually
  present. `counting.suspect_margin` (default 1) exists specifically to
  route small discrepancies to human review instead of an automatic
  `"mismatch"`, but a margin of 1 won't catch heavier occlusion in denser
  packaging photos.
- **Single photo, single angle.** Like the verification head, this
  compares against one uploaded photo — a partially visible pack (e.g.
  items behind the camera's frame) will undercount regardless of model
  accuracy.
