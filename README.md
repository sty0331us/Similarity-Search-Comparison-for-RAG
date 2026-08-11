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
