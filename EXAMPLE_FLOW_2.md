# 1) What counts as a “similar precedent”?

A case is a *similar precedent* if it matches along several legal dimensions:

* **Legal issues / provisions:** e.g., IBC §60(5)(c) (residuary NCLT jurisdiction), §238 (overriding effect), §14 (moratorium), contract termination during CIRP, arbitrability vs IBC.
* **Rhetorical role & holding type:** Reasoning/Holding about NCLT’s power to stay termination; ipso-facto termination; centrality to “going concern”.
* **Procedural posture & forum:** Supreme Court or coordinated High Court benches; NCLAT/NCLT when relevant.
* **Temporal & jurisdictional proximity:** Indian insolvency regime, post-2016 IBC amendments, decisions after *Gujarat Urja*.
* **Citation neighborhood:** cases *cited by* or *citing* the above (e.g., *Gujarat Urja v. Amit Gupta*, *Embassy Property*, *Indus Biotech*).

# 2) Indexing pipeline (offline)

You build a precedent corpus (Supreme Court, NCLAT, key HCs). For each judgment:

**(a) Parse & segment**

* Split into sentences/paragraphs.
* Run your **Role Classifier** → tag as Facts / Issue / AoP / AoR / Reasoning / Decision.
* Detect **holdings** (short “ratio decidendi” candidates) from Reasoning/Decision with rule+ML (cue phrases: “we hold”, “it was held”, “thus”, “therefore”, section mentions).

**(b) Extract legal signals**

* **Statutes/sections**: IBC §14, §25, §60(5)(c), §238; Arbitration Act §8, etc.
* **Entities**: forum (SC/HC/NCLAT/NCLT), judge names, dates.
* **Topics**: “termination during CIRP”, “ipso facto”, “going concern”, “residuary jurisdiction”.
* **Citations**: normalized case citations; build a **citation graph** (nodes=cases, edges=cites).

**(c) Vectorize (dense) + lexical (sparse)**

* Dense: sentence- or passage-level embeddings (e.g., legal-domain SBERT/GTE). Store vectors **per passage** with role labels.
* Sparse: BM25+ or SPLADE on the same passages.
* Store in:

  * **Vector store** (FAISS/Pinecone) → `embedding`
  * **Search index** (Elasticsearch/Whoosh) → `BM25`
* Metadata schema (per passage):

```json
{
  "case_id": "SC-2021-XYZ",
  "court": "Supreme Court of India",
  "date": "2021-06-24",
  "role": "Reasoning",
  "sections": ["IBC 60(5)(c)", "IBC 238", "IBC 14"],
  "topics": ["termination during CIRP", "residuary jurisdiction", "arbitration vs IBC"],
  "citations_out": ["Gujarat Urja v Amit Gupta", "Embassy Property", "Indus Biotech"],
  "text": "Thus, we are of the view that the NCLT does not have any residuary jurisdiction...",
  "embedding": [ ... 768-d ... ],
  "bm25_terms": "NCLT residuary jurisdiction Section 60(5)(c) termination ..."
}
```

# 3) Query construction (online)

Given your input case (TCS v SK Wheels) and the task “retrieve similar precedents”:

**(a) Build a structured query object `Q`**

* **Issue frame** (from the case’s **Issue/Reasoning** chunks):

  * “Can NCLT exercise residuary jurisdiction under §60(5)(c) IBC to adjudicate a *contractual termination* dispute and stay termination?”
  * “Scope of §238 (override) vs Arbitration clause”
  * “Applicability of §14 (moratorium) to services availed by corporate debtor vs supplied to it”
* **Signals**:

  * roles = {Reasoning, Decision, Holdings}
  * statutes = {IBC 60(5)(c), 238, 14, 25; Arbitration §8}
  * topics = {termination during CIRP, going concern, ipso facto clause}
  * court\_preference = {SC > NCLAT > HCs}
  * time\_window = {2016–present}

**(b) Expand query**

* From the conversation memory (past turns) + the case’s citations, add **seed cases** to boost (e.g., *Gujarat Urja v Amit Gupta*, *Embassy Property*, *Indus Biotech*).
* Generate **dense query vector** by encoding:

  * concatenation of the case’s **Issue + key Reasoning snippets + section list**.

# 4) Two-stage retrieval (hybrid)

**Stage 1 — Candidate generation (recall)**

* **Sparse**: BM25 top-K (e.g., K=500) with heavy boosts:

  * `^role:Reasoning^3 OR role:Decision^2`
  * `sections:(IBC 60(5)(c) OR 238 OR 14)^5`
  * `topics:(termination during CIRP OR going concern)^3`
  * `court:Supreme^2`
  * **Boost citations**: if doc cites or is cited by seed cases → bonus.
* **Dense**: ANN search over vectors with **role filter** `{Reasoning, Decision}`, statutes filter, top-M (e.g., M=500).
* **Union** the IDs → candidates C (e.g., |C|≈800).

**Stage 2 — Re-ranking (precision)**

* For each candidate passage (or merged case-level representation), compute a **composite legal similarity score**:

$$
\text{Score}(d \mid Q) \;=\; 
\alpha \cdot \text{BM25}(Q,d) \;+\;
\beta \cdot \cos(\mathbf{q}, \mathbf{d})
\;+\; \gamma \cdot \text{RoleMatch}(d)
\;+\; \delta \cdot \text{SectionMatch}(Q,d)
\;+\; \eta \cdot \text{CourtWeight}(d)
\;+\; \kappa \cdot \text{CitationBoost}(d)
\;+\; \lambda \cdot \text{Recency}(d)
$$

Where:

* `RoleMatch(d)∈{0,1,2}` (2 if Reasoning/Holding, 1 if Decision headnote, 0 otherwise)
* `SectionMatch` counts exact section overlaps (scaled to \[0,1])
* `CourtWeight`: SC=1.0, NCLAT=0.7, HC=0.5, NCLT=0.3 (tune)
* `CitationBoost`: +0.2 if cites/is-cited-by any seed case
* `Recency`: time decay or stepwise bonus for post-IBC jurisprudence

**Typical weights** (start, then tune by validation):
$\alpha=0.35,\;\beta=0.35,\;\gamma=0.10,\;\delta=0.10,\;\eta=0.04,\;\kappa=0.04,\;\lambda=0.02$

**Maximal Marginal Relevance (MMR)** for diversity across courts/years/issues when selecting top N cases:

$$
\text{MMR} = \arg\max_{d \in C \setminus S} \Big\{ \rho \cdot \text{Score}(d) - (1-\rho)\max_{s\in S} \cos(\mathbf{d},\mathbf{s}) \Big\}
$$

with $\rho \in [0.5,0.8]$.

# 5) Worked miniature example (manual numbers)

Assume 4 candidate cases after Stage-1:

| Case                                                                 | BM25 | Cosine | RoleMatch | SectionMatch | Court | Cites Seed | Recency | Score (α..λ as above) |
| -------------------------------------------------------------------- | ---: | -----: | --------: | -----------: | ----: | ---------: | ------: | --------------------: |
| **Gujarat Urja v Amit Gupta (SC)**                                   |   10 |   0.80 |         2 |         1.00 |    SC |        Yes |    High |                       |
| $0.35·10 + 0.35·0.80 + 0.10·2 + 0.10·1 + 0.04·1.0 + 0.04·1 + 0.02·1$ |      |        |           |              |       |            |         |                       |
| $= 3.5 + 0.28 + 0.20 + 0.10 + 0.04 + 0.04 + 0.02 = \mathbf{4.18}$    |      |        |           |              |       |            |         |                       |
| **Indus Biotech (SC)**                                               |    7 |   0.70 |         2 |         0.66 |    SC |         No |    High |                       |
| $2.45 + 0.245 + 0.20 + 0.066 + 0.04 + 0 + 0.02 = \mathbf{3.02}$      |      |        |           |              |       |            |         |                       |
| **Embassy Property (SC)**                                            |    6 |   0.66 |         2 |         0.50 |    SC |         No |     Med |                       |
| $2.10 + 0.231 + 0.20 + 0.05 + 0.04 + 0 + 0.01 = \mathbf{2.63}$       |      |        |           |              |       |            |         |                       |
| **Random HC contract case**                                          |    8 |   0.40 |         1 |         0.10 |    HC |         No |     Med |                       |
| $2.80 + 0.14 + 0.10 + 0.01 + 0.02 + 0 + 0.01 = \mathbf{3.08}$        |      |        |           |              |       |            |         |                       |

MMR will still rank **Gujarat Urja** first; between the others, diversity may push one SC + one HC if needed.

# 6) Implementation sketch (Python)

**Hybrid candidate generation**

```python
def generate_candidates(query_obj, bm25_index, vector_store, k_bm25=500, k_vec=500):
    # Sparse with boosts
    bm25_query = build_boosted_query(query_obj)
    cand_sparse = bm25_index.search(bm25_query, top_k=k_bm25)

    # Dense with filters
    q_text = build_dense_query_text(query_obj)  # Issues + Reasoning snippets + sections
    q_vec  = encoder.encode(q_text, normalize_embeddings=True)
    filters = {"role": ["Reasoning","Decision"],
               "sections": query_obj.sections,
               "jurisdiction": "India"}
    cand_dense = vector_store.search(q_vec, top_k=k_vec, filters=filters)

    return dedup_by_case_id(cand_sparse + cand_dense)
```

**Re-ranking**

```python
def score_case(case, q_vec, bm25_score, weights):
    cosine = cosine_sim(q_vec, case.embedding)
    role_match = 2 if case.role in {"Reasoning","Decision"} else 0
    section_match = jaccard(case.sections, Q.sections)  # 0..1
    court_weight = {"Supreme Court":1.0, "NCLAT":0.7, "High Court":0.5, "NCLT":0.3}.get(case.court,0.2)
    cites_seed = 1.0 if any(c in Q.seed_cases for c in case.citations_out+case.citations_in) else 0.0
    recency = time_bonus(case.date)  # e.g., 1 (high) / 0.5 (med) / 0.0 (low)

    α,β,γ,δ,η,κ,λ = weights
    return (α*bm25_score + β*cosine + γ*role_match + δ*section_match +
            η*court_weight + κ*cites_seed + λ*recency)
```

**MMR selection**

```python
def mmr_select(candidates, q_vec, top_n=10, rho=0.7):
    selected, selected_vecs = [], []
    while candidates and len(selected) < top_n:
        best, best_score = None, -1e9
        for c in candidates:
            sim_to_S = max((cosine_sim(c.embedding, v) for v in selected_vecs), default=0.0)
            score = rho*c.final_score - (1-rho)*sim_to_S
            if score > best_score:
                best, best_score = c, score
        selected.append(best)
        selected_vecs.append(best.embedding)
        candidates.remove(best)
    return selected
```

# 7) Role-aware use in your case

For **TCS v SK Wheels** (your excerpt), the query object will emphasize:

* **Sections**: IBC §60(5)(c), §238, §14, §25
* **Topics**: termination during CIRP, going concern, residuary jurisdiction, arbitration vs IBC
* **Roles**: Reasoning/Decision/Holding
* **Seed cases** to boost: *Gujarat Urja*, *Embassy Property*, *Indus Biotech*
* **Court preference**: Supreme Court first

This steers retrieval toward precedents that decide:

* When NCLT can (or cannot) use §60(5)(c) to interfere with termination;
* How §238 interacts with arbitration clauses;
* Limits of §14 to supplies/services and property possession;
* “Centrality to going concern” test and ipso facto termination.

# 8) Quality controls & guardrails

* **Dedupe by legal proposition**: if multiple passages from the same case state the same holding, keep the best one.
* **Contradiction check**: cross-encoder reranker to detect opposite holdings; surface both sides if circuit splits exist.
* **Explainability**: return each precedent with:

  * matched **sections/topics**,
  * **role** of the passage (Reasoning/Holding),
  * short **rationale snippet**, and
  * **why it was ranked** (e.g., “Exact §60(5)(c) + Reasoning + cites Gujarat Urja”).

# 9) Evaluation (recommended)

* **IR**: Recall\@K, nDCG\@K, MRR\@K on a labeled query–precedent set.
* **RAG**: Faithfulness/Attribution (e.g., RAGAS), Judge-authored extract match rate.
* **Ablations**: dense-only vs sparse-only vs hybrid; with/without role filters; with/without section boosts.

---

## Takeaway

“Retrieve similar precedents” is a **hybrid, role-aware, statute-aware** pipeline:

1. Role-tag and feature-extract the corpus,
2. Generate dense and sparse candidates under **legal filters** (roles, sections, forum),
3. **Re-rank** with a composite legal similarity that respects sections, holdings, court level, citations, and time,
4. Use **MMR** to produce a compact, diverse precedent set,
5. Return **evidence passages** (Reasoning/Holding) with transparent ranking reasons.
