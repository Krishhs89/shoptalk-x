# ShopTalk-X — Presentation Outline

*For the expert-panel demo. Slide-by-slide outline with speaker notes and
pointers to the exact artifact/number to show for each — not a finished
deck, but everything needed to build one quickly.*

## 1. Title
ShopTalk-X: Production Multimodal Shopping Assistant with Visual
Verification. One line: conversational search over a real catalog, plus
photo-based fraud/wrong-item detection.

## 2. Problem & motivation
- Keyword search fails on natural queries ("red shirt under $50, not
  striped"). Show the problem statement's own example.
- Business cost: poor discoverability hurts NPS; wrong-item/counterfeit
  deliveries are a multi-billion-dollar returns-fraud cost center.
- *Speaker note:* frame as two connected problems (discovery + trust), not
  two unrelated features bolted together.

## 3. System architecture (one diagram)
Show the mermaid diagram from `docs/TECHNICAL_ARCHITECTURE.md` — client →
API → two-stage retrieval → generation, plus the verification path
alongside it, plus the offline data/MLOps pipeline underneath.

## 4. Two-stage retrieval, with real numbers
- Explain bi-encoder (fast, coarse) → cross-encoder (slow, precise)
  narrowing 10k → 100 → 10.
- **Show `results/day2_retrieval_eval.md`**: Recall@10 +10.5pp, MRR
  +11.3pp, NDCG@10 +14.6pp from reranking; Recall@100 flat (proves the
  reranker reorders, doesn't just get lucky with a bigger net).

## 5. Multimodal: photo search
- Live or recorded demo: upload a photo, watch it CLIP-embed → ANN → get
  BLIP-captioned into a pseudo-query → rerank → grounded answer.
- Call out the design choice: reusing the text reranker via a
  BLIP-generated pseudo-query instead of building a second model.

## 6. Generation: grounded, conversational, safe
- Show `shoptalk/rag/prompts.py`'s delimiting strategy — retrieved product
  text is data, not instructions (prompt-injection defense).
- Demo a follow-up question to show conversation memory.
- Demo an out-of-domain question ("write me a poem") to show the refusal
  behavior.

## 7. Visual verification
- Explain the wrong-item/counterfeit business problem concretely.
- **Show `results/day5_verification_eval.md`**: ROC-AUC 0.926, FAR 0.048,
  FRR 0.200 (smoke-test scale — say so).
- Demo: upload a mismatched item photo → `mismatch` verdict; explain the
  `suspect` band routes ambiguous cases to a human, never an automatic
  accusation.

## 8. Fine-tuning: does it actually help?
- **Show `results/day5_finetune_eval.md`**: Recall@10 base 0.897 → fine-tuned
  0.959 (+6.2pp), from triplet loss + same-category hard negatives mined
  from the golden set.
- One sentence on why hard negatives matter (easy negatives from unrelated
  categories teach the model nothing new).

## 9. Production engineering
- Models loaded once at startup (`shoptalk/api/main.py`'s lifespan
  handler) — never per-request.
- Prometheus `/metrics`, per-stage latency logged to SQLite for every
  request.
- API key auth, rate limiting, request size caps, prompt-injection
  defenses (design doc §8) — name them, don't over-explain.

## 10. MLOps loop
- MLflow tracking + model registry (show a screenshot or live
  `localhost:5000` if presenting live).
- **Show `results/day6_drift_report.md`**: drift correctly detected (7/7
  columns) on a deliberately shifted simulated query set — proves the
  monitoring actually works, not just that code exists.
- Airflow DAG: data → rebuild triplets → fine-tune → evaluate → promote
  only if better → trigger deploy. Show the DAG graph
  (`airflow/dags/retrain_embeddings_dag.py`) — a regression gate, not blind
  auto-promotion.

## 11. Deployment
- Docker Compose stack (ollama + api + ui + mlflow), one command to run
  locally.
- CI/CD: lint → test → build → push → deploy, gated safely (skips
  cloud steps without credentials rather than failing).
- AWS EC2 steps fully documented (`docs/deployment/aws_ec2.md`) —
  state plainly whether this was actually deployed to AWS for the demo or
  run locally, and why (cost/scope decision, not a limitation of the code).

## 12. Honest limitations (own this slide, don't skip it)
- Synthetic pricing (ABO has none) — clearly flagged everywhere in the code
  and UI.
- Verification head trained at smoke-test scale — real calibration needs
  the full catalog.
- Latency numbers from development were captured on CPU-only hardware —
  cite `results/day6_latency_report.md`'s honest caveat, and give the
  *retrieval-side* numbers (which are hardware-independent-ish) rather than
  end-to-end LLM latency if that's what you actually measured.
- Not built: quantity validation (YOLO-based order-count check) — name it
  as the acknowledged, largest remaining stretch item.

## 13. What's next
- Quantity validation (BinSense-style order fulfillment check).
- LLM LoRA/QLoRA fine-tuning execution (code delivered, not run — see
  `docs/finetuning/llm_lora_qlora.md`).
- Kubernetes deployment (k3s steps documented, not executed).
- Real user feedback loop closing into the Airflow retraining DAG.

## 14. Q&A / appendix backup slides
Keep these ready but don't present by default:
- Full metrics table across all components (`results/*.md`).
- Model cards (`docs/model_cards/`) if asked about a specific model's
  training data or failure modes.
- Cost breakdown (design doc §13).

---

## Demo script (if doing it live, not recorded)

1. `docker compose up -d` (or confirm services already running).
2. Text query in the UI → point out the grounded answer + cited IDs +
   latency footer.
3. Follow-up question → point out conversation memory.
4. Upload a photo → point out the visual match + generated caption.
5. Verify tab: upload a genuinely mismatched photo → `mismatch` verdict.
6. Briefly show `/metrics` and the MLflow UI in a second tab.
7. Close on the drift report — "the system can tell when its own inputs
   have shifted, not just serve predictions blindly."

Keep the live demo under 5 minutes; everything else is backup material for
questions.
