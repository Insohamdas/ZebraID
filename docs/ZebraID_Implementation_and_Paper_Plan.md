# ZebraID: A Stripe-Based Biometric Identification System for Individual Zebra Recognition at Continental Scale

**Detailed Implementation & Research Paper Plan**
Prepared for: Soham Das, AgriScore Private Limited
Document Type: Prototype Build Plan + Paper Writing Plan

---

## 0. Executive Summary

ZebraID is a stripe-pattern biometric identification system designed to solve a problem that
existing zebra re-identification tools do **not** solve: cross-population, cross-organization,
edge-deployable identification at continental scale.

Existing tools (StripeSpotter, HotSpotter, Wildbook, MegaDescriptor/WildlifeDatasets) prove that
individual zebras can be identified from stripe patterns. What none of them solve together is:

1. A single embedding space that generalizes across **subspecies and viewpoints** without
   per-site retraining.
2. A **federated matching protocol** that lets independently operated conservation databases
   (different countries, different organizations, different tools) cross-match individuals
   **without centralizing raw images or GPS-sensitive location data**.
3. **Edge-first deployment**, where the compact biometric identifier — not the raw photo — is
   what gets transmitted from a remote, low-bandwidth field station.

This document is split into two parts:

- **Part A — Prototype Implementation Plan**: everything needed to build a working,
  demonstrable system.
- **Part B — Research Paper Plan**: how to turn the prototype's results into a defensible,
  clearly-positioned paper.

Every genuinely novel element is explicitly flagged with a `⭐ NOVEL` tag throughout this
document so nothing gets lost in the implementation detail.

---

# PART A — PROTOTYPE IMPLEMENTATION PLAN

## A.1 System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         ZEBRAID ARCHITECTURE                        │
│                                                                       │
│  ┌────────────────┐        ┌────────────────┐                       │
│  │  Field Camera   │        │  Field Camera   │   ... (many sites)   │
│  │  Node (ESP32 +  │        │  Node (ESP32 +  │                       │
│  │  camera + LoRa) │        │  camera + LoRa) │                       │
│  └────────┬───────┘        └────────┬───────┘                       │
│           │ raw image (local only)   │                               │
│           ▼                          ▼                               │
│  ┌────────────────┐        ┌────────────────┐                       │
│  │  Edge Inference │        │  Edge Inference │                     │
│  │  Node (Jetson/  │        │  Node (Jetson/  │                     │
│  │  Raspberry Pi)  │        │  Raspberry Pi)  │                     │
│  │                 │        │                 │                     │
│  │ - Detect zebra  │        │ - Detect zebra  │                     │
│  │ - Extract stripe │        │ - Extract stripe │                    │
│  │   embedding      │        │   embedding      │                    │
│  │ - Compress to    │        │ - Compress to    │                    │
│  │   Z-Hash (256b)  │        │   Z-Hash (256b)  │                    │
│  └────────┬────────┘        └────────┬────────┘                     │
│           │  Z-Hash only (no image)   │                              │
│           ▼                           ▼                              │
│  ┌──────────────────┐       ┌──────────────────┐                    │
│  │ Org A Local Shard │       │ Org B Local Shard │   ⭐ NOVEL:       │
│  │ (FAISS index +    │◄─────►│ (FAISS index +    │   Federated       │
│  │  metadata DB)     │ match  │  metadata DB)     │   cross-match     │
│  │                    │ query  │                    │   protocol       │
│  └──────────────────┘  API   └──────────────────┘                    │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

**Key architectural decision (flagged from earlier discussion):** an ESP32-class
microcontroller **cannot** run the embedding model. It only handles image capture and LoRa
telemetry. All neural inference happens on a Jetson Orin Nano / Nano, or a Raspberry Pi 5 —
this is the "Edge Inference Node." Be explicit about this tier separation in both the
prototype and the paper; conflating them is a credibility risk.

---

## A.2 Tech Stack

| Layer | Tool / Library | Notes |
|---|---|---|
| Language | Python 3.11 | primary implementation language |
| Deep learning | PyTorch + `timm` | for backbone models (ConvNeXt, ViT, ResNet) |
| Pretrained re-id backbone | MegaDescriptor (via WildlifeDatasets toolkit) | starting point for fine-tuning |
| Metric learning | `pytorch-metric-learning` | triplet / ArcFace loss implementations |
| Vector search | FAISS | exact search for prototype scale; IVF-PQ/HNSW for scale simulation |
| Data handling | WildlifeDatasets toolkit, `pandas`, COCO-format parsers | dataset harmonization |
| Federated demo | FastAPI (2 local service instances) + SQLite | simulate two independent "org" shards |
| Edge deployment | Raspberry Pi 5 or Jetson Orin Nano, ONNX Runtime / TensorRT | model export + on-device inference |
| Sensor node (optional physical demo) | ESP32-CAM + LoRa module (e.g., RFM95) | image capture + telemetry only, no inference |
| Visualization | Matplotlib, existing diagram toolkit | consistent with your AgriScore proposal visuals |

---

## A.3 Datasets

| Dataset | Content | Role in Prototype |
|---|---|---|
| Great Zebra and Giraffe Count (GZGC), hosted on LILA BC | ~4,948 images, plains zebra + Masai giraffe, individual IDs, COCO format | Primary population "A" |
| A second population/subspecies dataset accessed via the WildlifeDatasets toolkit (e.g., a Grevy's zebra or mountain zebra re-id dataset) | Individual IDs from a distinct site/subspecies | Population "B" — required for the cross-population generalization test |
| WildlifeReID-10k (reference benchmark, not necessarily trained on) | ~10,000 individuals across many species | Used as an external sanity-check benchmark for embedding quality, and to justify "at scale" claims responsibly |
| Your own phone/camera footage | Small manual sample | Used only for the edge-deployment latency/power demo, not for accuracy claims |

**Important framing note for the paper:** none of these datasets combined constitute a literal
continent-wide census. The prototype validates the *architecture and generalization capability*
on the best available multi-population data; true continental deployment is described as a
roadmap (see Part B, Section B.5).

**Licensing note:** GZGC is released under the CDLA-Permissive-1.0 license, which requires
attribution and citation of the original publication (Parham et al., 2017) in any work that
uses it. Check the license terms of your second population dataset the same way before
publishing — most Wildbook-sourced datasets carry similar community-data licenses, but the
attribution requirements differ slightly by dataset.

---

## A.4 Implementation Plan — Step by Step

### Step 1: Environment & Baseline Setup
- Set up Python environment with PyTorch, `timm`, WildlifeDatasets, FAISS.
- Pull GZGC and a second population dataset through the toolkit's dataset loaders.
- Reproduce a baseline: run HotSpotter (or an equivalent SIFT-based matcher) and a
  MegaDescriptor zero-shot embedding on both datasets separately. **Record these numbers —
  they are your comparison baselines for the entire paper.**

### Step 2: ⭐ NOVEL — Cross-Population Embedding Training
This is the technical core of the "continental scale" claim.
- Fine-tune MegaDescriptor (or a ConvNeXt/ViT backbone) using triplet loss.
- **Critical detail:** construct training batches that deliberately mix individuals from
  *both* populations in every batch, rather than training separately per dataset. This is
  what forces the embedding space to encode stripe-pattern similarity rather than
  population-specific visual artifacts (lighting, background, camera type, subspecies stripe
  density).
- Train two variants for comparison:
  - **Baseline:** fine-tuned on population A only, tested on population A (expected: high accuracy).
  - **Baseline:** fine-tuned on population A only, tested on population B (expected: accuracy drop — this quantifies the generalization gap).
  - **ZebraID variant:** fine-tuned on mixed A+B batches, tested on held-out individuals from both (expected: smaller gap).
- This three-way comparison table is your single most important result.

### Step 3: ⭐ NOVEL — Z-Hash Compression Layer
- Add a compression head after the embedding backbone: a small MLP projecting the
  high-dimensional embedding down to a fixed-size code (target: 256-bit, matching your
  original naming), using either:
  - PCA + binarization (simplest, fastest to implement), or
  - Product Quantization (via FAISS's built-in PQ trainer — more accurate, reusable for the
    scale-benchmarking step below).
- Validate that compression doesn't meaningfully hurt rank-1 accuracy versus the full
  embedding — report this trade-off curve (accuracy vs. code size) explicitly.

### Step 4: Matching Engine + Scale Simulation
- Build a FAISS `IndexFlatL2` (or cosine) for exact matching at real dataset scale.
- Separately, build an `IndexIVFPQ` or HNSW index, and populate it by augmenting your real
  embeddings with synthetic "phantom" vectors (sampled from the same distribution) up to
  100k, then 1M entries.
- Benchmark recall@1 / recall@5 vs. query latency vs. memory footprint at each scale point.
- **Label this experiment explicitly as a simulated-scale infrastructure benchmark** — don't
  imply you tested on a million real zebras.
- **Add a human-in-the-loop confidence threshold.** Below a chosen similarity threshold,
  don't auto-confirm a match — return a short ranked list of top candidates for a human
  reviewer to confirm, the same pattern Wildbook uses at scale. This matters for two reasons:
  it's a realistic deployment detail reviewers will expect, and it protects you from
  overclaiming fully-automated accuracy in the paper.

### Step 5: ⭐ NOVEL — Federated Cross-Match Protocol
This is your most distinctive systems contribution — build it carefully.
- Split your two population datasets into two separate local services ("Org A" and "Org B"),
  each with its own FAISS index + SQLite metadata store, running as separate FastAPI
  instances (can be two processes on one machine for the prototype).
- Design a minimal cross-query API: Org A can send a Z-Hash (not an image, not GPS data) to
  Org B's `/match` endpoint and receive back a match confidence score and an opaque record ID
  — no raw image or precise location ever crosses the boundary.
- Implement and log this explicitly: for every cross-org query, print/record exactly what
  bytes were transmitted, so the privacy claim is demonstrable, not just asserted.
- Test: take a real individual that appears (or is seeded) in both shards, confirm the
  cross-shard match resolves correctly; take individuals unique to one shard, confirm no
  false cross-matches.

### Step 6: Edge Deployment Benchmark
- Export the trained embedding model to ONNX (or TensorRT if using Jetson).
- Run inference on a Raspberry Pi 5 or Jetson Orin Nano.
- Measure: inference latency, power draw (if you have a USB power meter), and the size in
  bytes of the final Z-Hash payload that would be transmitted over LoRaWAN.
- This produces the concrete "deployable at a remote camera trap" evidence for the paper.

### Step 7: Integration & Demo Packaging
- Wire steps 2–6 into a single demo flow: image in → detection → embedding → Z-Hash →
  local match → (optional) federated cross-org match → result displayed.
- Package with the diagram style you're already using for the AgriScore proposal, for visual
  consistency across your hackathon materials.

---

## A.5 Suggested Timeline

| Week | Milestone |
|---|---|
| 1 | Environment setup, dataset pulls, reproduce HotSpotter/MegaDescriptor baselines |
| 2 | Cross-population triplet training (Step 2), first accuracy comparison table |
| 3 | Z-Hash compression layer + accuracy/size trade-off curve |
| 4 | FAISS scale simulation + federated two-shard demo (Steps 4–5) |
| 5 | Edge deployment on Pi/Jetson, latency/power benchmarking (Step 6) |
| 6 | Integration, demo packaging, buffer for fixes |

---

## A.6 Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Second population dataset harder to source than expected | Fall back to a strong synthetic population split (e.g., geographically distinct camera subsets within GZGC) and state this limitation explicitly in the paper |
| Cross-population accuracy gap too small to be an interesting result | Still valuable — report it as evidence of strong generalization rather than framing it as a negative result |
| Edge hardware unavailable in time | Report latency/power on a laptop CPU-limited profile as a proxy, clearly labeled as a proxy measurement |
| Federated demo scope creep (trying to build real cross-org infra) | Keep it to two local services — the protocol design is the contribution, not a production deployment |

---

## A.7 ⭐ NOVEL — Security & Privacy Threat Model for the Federated Protocol

This section was missing from the original plan and matters a lot: "no raw image or GPS
crosses the boundary" is not automatically safe on its own. A federated matching API that
returns similarity scores can still leak information through **membership-inference
attacks** — an adversary who queries repeatedly with slightly varied inputs can sometimes
reconstruct facts about what's in the other org's private gallery, even without ever seeing
a raw image. Address this explicitly rather than leaving it implicit:

- **Score quantization:** return coarse match/no-match/likely-match buckets rather than a
  raw floating-point similarity score, which reduces the information leaked per query.
- **Rate limiting & query auditing:** cap the number of cross-org queries per time window per
  requester, and log every cross-org query for audit — both are simple to implement in the
  FastAPI demo and make the privacy claim testable rather than assumed.
- **Authentication between shards:** even in the local two-service demo, use a signed
  API key or mutual TLS so the protocol reflects how real inter-organizational trust would
  need to work.
- **State clearly in the paper what is out of scope:** true cryptographic guarantees
  (secure multiparty computation, differential privacy noise on embeddings) are a natural
  extension but not required for the prototype — mention them explicitly as future work
  rather than silently ignoring the attack surface.

## A.8 ⭐ NOVEL — Ethical & Data Sensitivity Considerations

This is worth a dedicated section because it strengthens your motivation, not just your
ethics statement. Precise GPS locations of endangered animals (Grevy's zebra is
endangered; mountain zebra is vulnerable) are genuinely dangerous data — there is a
documented history of poachers exploiting leaked wildlife-tracking data in other species.
This means:

- The reason conservation organizations *don't* pool their raw data today isn't only
  bureaucratic friction — it's often a legitimate, justified security concern. This actually
  **strengthens your paper's motivation**: you're not solving a inconvenience, you're solving
  a problem organizations have good reason to be cautious about.
- State explicitly in the paper that the federated protocol's design goal is to make
  cross-organization identification possible *precisely because* raw location data must stay
  private, not merely as a nice-to-have.
- If you use any real field-collected images (not just published datasets) for the edge
  demo, strip EXIF GPS metadata before any of it touches a shared or public artifact.

## A.9 Reproducibility & Experiment Tracking

Small addition, but it noticeably strengthens a paper submission:
- Use `git` for all code, and a lightweight experiment tracker (Weights & Biases free tier,
  or even a structured CSV log) to record every training run's hyperparameters, dataset
  split, and resulting metrics — you'll run more variants than you expect (per-population
  baseline, mixed-population, different compression sizes, different thresholds).
- Plan to release code (and a model card describing training data and known limitations) on
  GitHub alongside the paper — this is the norm in this specific research community
  (StripeSpotter, SMALST, and WildlifeDatasets were all released openly) and reviewers in
  this space notice and reward it.
- Estimate your compute budget up front: fine-tuning a mid-size ViT/ConvNeXt backbone on a
  few thousand images for a few epochs is feasible on a single free-tier Colab GPU or a
  modest cloud GPU rental (a few hours, not days) — worth confirming early so it doesn't
  become a Week 4 surprise.

---

# PART B — RESEARCH PAPER PLAN

## B.1 Positioning Statement (use this as your paper's core novelty claim)

> Existing zebra re-identification systems (StripeSpotter, HotSpotter, Wildbook, MegaDescriptor)
> demonstrate accurate individual identification within a single population or dataset, but
> none address cross-population generalization, cross-organizational federated matching without
> data centralization, or edge-first deployment together. ZebraID contributes (1) a
> cross-population-invariant compact embedding, (2) a privacy-preserving federated matching
> protocol for independently operated conservation databases, and (3) a validated edge
> deployment path for low-bandwidth field stations.

## B.2 Comparison Table (draft — fill in with your actual measured numbers)

| System | Individual ID | Cross-population generalization tested | Federated / no data centralization | Edge-deployed | Compact hash |
|---|---|---|---|---|---|
| StripeSpotter (2011) | Yes | No | No | No | Partial (stripecode) |
| HotSpotter / Wild-ID | Yes | No | No | No | No |
| Wildbook | Yes | Partial (multi-species, single DB) | No | No | No |
| MegaDescriptor / WildlifeDatasets | Yes | Not explicitly evaluated | No | No | No |
| **ZebraID (this work)** | Yes | **Yes** | **Yes** | **Yes** | **Yes** |

## B.3 Paper Structure

1. **Introduction**
   - The conservation monitoring bottleneck (invasive tagging vs. inaccurate passive camera traps).
   - The fragmentation problem: siloed, non-interoperable ID systems across organizations/countries.
   - Contribution list (map directly to the three ⭐ NOVEL items above).

2. **Related Work**
   - Stripe-pattern re-id: StripeSpotter, HotSpotter/Wild-ID, Wildbook.
   - Foundation models for wildlife re-id: MegaDescriptor / WildlifeDatasets toolkit.
   - 3D animal shape/pose (SMALST/3D Safari) — cited as adjacent but out of scope for this
     paper (you deliberately narrowed scope from the earlier 6-pillar design; say so plainly).
   - Large-scale vector search (FAISS, HNSW, product quantization) — cited to show you're
     building on solved infrastructure, not claiming it as a contribution.
   - Manual/fragmented ID practices still in active use (e.g., the Hartmann's mountain zebra
     spreadsheet-coding approach) — this is your strongest evidence that fragmentation is a
     real, current, documented problem, not a strawman.

3. **Method**
   - 3.1 Cross-population embedding training (triplet loss, mixed-batch construction).
   - 3.2 Z-Hash compression (PQ/PCA + binarization).
   - 3.3 Federated matching protocol (API design, what data crosses boundaries and what doesn't).
   - 3.4 Edge deployment architecture (tiered ESP32 sensor node / Jetson-Pi inference node design).

4. **Experiments**
   - 4.1 Cross-population generalization (the three-way comparison from Step 2). Run this
     across multiple random train/test splits (not just one) and report mean ± standard
     deviation, or a confidence interval — a single-split accuracy number is a common and
     avoidable weakness reviewers flag in animal re-id papers.
   - 4.2 Compression trade-off (accuracy vs. Z-Hash size).
   - 4.3 Scale simulation (recall@k vs. latency/memory at 1k/100k/1M).
   - 4.4 Federated protocol correctness + data-exposure audit (see A.7).
   - 4.5 Edge latency/power benchmarking.
   - 4.6 ⭐ Failure-mode / qualitative error analysis. Deliberately inspect and report the
     cases where ZebraID gets it wrong: near-identical stripe patterns between close
     relatives or herd-mates, heavy mud/vegetation occlusion, low-resolution or motion-blurred
     camera-trap frames, and partial-flank-only shots. A short, honest failure gallery with a
     few example images and a discussion of which failure modes the human-in-the-loop step
     (A step 4 addition) would catch is far more convincing to reviewers than a report of only
     successes, and it directly justifies the confidence-threshold design decision.

5. **Discussion**
   - Explicit, honest scoping statement: validated architecture and generalization across
     tested populations; continent-wide multi-organization deployment is proposed as future
     work, not claimed as achieved.
   - Limitations: dataset size relative to true continental zebra populations; simulated vs.
     real cross-organization deployment; single-species focus (zebra only, extensibility to
     other patterned species discussed but not tested).
   - Ethics & security: state the poaching-risk motivation for the federated design (A.8),
     and be upfront about which privacy guarantees are implemented (rate limiting, score
     quantization, authentication) versus which are future work (formal cryptographic
     guarantees, differential privacy) — see A.7.

6. **Conclusion & Future Work**
   - Roadmap: real pilot with two actual partner organizations (e.g., a Grevy's zebra
     conservation group and a mountain zebra project), extension to giraffes/other patterned
     species using the same architecture, integration path back to the fuller AgriScore
     health-monitoring vision as a longer-term direction.

## B.4 Target Venues

| Venue | Why it fits |
|---|---|
| CV4Animals (CVPR workshop) | Direct audience for animal re-id + conservation CV work |
| AnimalCLEF / LifeCLEF (ImageCLEF) | Active, relevant benchmark community for cross-dataset animal re-id |
| Methods in Ecology and Evolution | Strong fit for the conservation-practice / deployment angle |
| arXiv preprint | Post first regardless, to establish priority and get early feedback |

## B.5 Continental Deployment Roadmap (for Discussion/Future Work section)

Be explicit that this is a roadmap, not a claim:
1. Pilot federation with 2 real partner organizations (not simulated shards).
2. Expand embedding training to 4–5 populations spanning multiple countries/subspecies.
3. Real LoRaWAN field deployment at a small number of sites (not full continental coverage).
4. Formal data-governance agreement template for cross-border wildlife data sharing (a
   genuinely useful non-technical contribution to mention, since this is often the actual
   blocker to "continental scale" systems in practice, not the algorithms).

---

## Appendix: Key Reference Points (for your literature review — verify formal citations before submission)

- Lahiri et al., *StripeSpotter* (2011)
- Crall et al., *HotSpotter: Patterned Species Instance Recognition* (2013)
- Parham et al., *Animal population censusing at scale with citizen science and photographic
  identification* (2017) — source of the GZGC dataset
- Zuffi, Kanazawa, Berger-Wolf, Black, *Three-D Safari: Learning to Estimate Zebra Pose,
  Shape, and Texture from Images "In the Wild"* (ICCV 2019) — adjacent work, cite as related
  but out of scope
- Čermák, Picek, Adam, Papafitsoros, *WildlifeDatasets: An open-source toolkit for animal
  re-identification* (WACV 2024) — source of MegaDescriptor and FAISS integration
- Adam, Čermák, Papafitsoros, Picek, *WildlifeReID-10k* (2025) — external benchmark reference
- Relevant mountain zebra manual-ID literature — evidence of current fragmentation in the field

---

*End of document.*
