# Similarity Search Comparison for RAG

**Manual math vs NumPy vs SciPy vs PyTorch** — the same vector similarity metrics (L2, dot product, cosine) computed several ways so you can see *when* to use each library and *how* the mechanisms differ for RAG retrieval.

| | Manual | NumPy | SciPy | PyTorch |
|---|---|---|---|---|
| Style | Explicit formulas | Array ops / broadcasting | Distance utility APIs | Tensor ops + `nn.functional` |
| Best for | Learning the math | Fast CPU batching | Ready-made pairwise distances | GPU / deep-learning stacks |
| Typical RAG role | Teaching / debugging | Embedding matrices on CPU | Quick `cdist` / `pdist` | Online retrieval next to models |

---

## Table of contents

1. [Problem & approach](#problem--approach)
2. [Similarity metrics at a glance](#similarity-metrics-at-a-glance)
3. [Library comparison matrix](#library-comparison-matrix)
4. [Mechanism differences](#mechanism-differences)
5. [When to use which](#when-to-use-which)
6. [Worked example: cosine search by hand](#worked-example-cosine-search-by-hand)
7. [Decision guide](#decision-guide)
8. [Wrap-up](#wrap-up)

---

## Problem & approach

In RAG, you embed a query and documents into vectors, then **rank documents by similarity** to the query. The ranking math is usually one of:

| Metric | What it measures | Higher / lower wins? |
|---|---|---|
| **L2 (Euclidean) distance** | Straight-line distance between vectors | **Lower** is more similar |
| **Dot product** | Alignment × magnitude | **Higher** is more similar (as similarity) |
| **Cosine similarity** | Angle between vectors (direction only) | **Higher** is more similar (max 1 when unit-normalized) |

This comparison shows how to compute those metrics:

1. **By hand** — write the formulas yourself (clearest for learning)
2. **NumPy** — vectorized array math (default for CPU notebooks / scripts)
3. **SciPy** — battle-tested distance helpers (`spatial.distance`)
4. **PyTorch** — tensor APIs, especially when embeddings already live on GPU

---

## Similarity metrics at a glance

### Cheat sheet

| Metric | Formula (idea) | Needs normalization? | Common API shapes |
|---|---|---|---|
| L2 distance | \(\|a - b\|_2\) | Optional (scale still matters) | Manual loop, `np.linalg.norm`, `scipy.spatial.distance.euclidean`, `torch.cdist` |
| Dot product similarity | \(a \cdot b\) | Often yes for fair ranking | `@` / `np.dot`, `torch.matmul` |
| Dot product distance | often \(-a \cdot b\) or \(1 - a\cdot b\) (context-dependent) | Same as above | Derived from similarity |
| Cosine similarity | \(\dfrac{a\cdot b}{\|a\|\|b\|}\) | Built into the formula; **unit-normalize first** → cosine ≡ dot | Manual, `sklearn`/`scipy`, `F.cosine_similarity`, or normalize + `@` |
| Cosine distance | \(1 - \text{cosine similarity}\) | Same as cosine | `scipy.spatial.distance.cosine` |

### Metric comparison matrix (RAG retrieval)

| Question | L2 distance | Dot product | Cosine similarity |
|---|---|---|---|
| Sensitive to vector length? | Yes | Yes | No (direction only) |
| Good when embeddings are unit-normalized? | Yes (then related to cosine) | Yes (then ≡ cosine) | Ideal |
| Typical RAG embedding models | Works | Common with normalized embeddings | Very common default |
| Ranking rule | `argmin` distance | `argmax` similarity | `argmax` similarity |
| Intuition | “How far apart?” | “How aligned *and* strong?” | “How aligned in meaning space?” |

> **Practical RAG tip:** Many embedding models already L2-normalize vectors. After that, **cosine similarity ≡ dot product**, so retrieval can use a single matrix multiply: `docs @ query.T`.
