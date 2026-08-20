# Retrieval Evaluation

RAGOps includes a manually labelled retrieval benchmark for comparing
Semantic Search, BM25, and Hybrid Search using Reciprocal Rank Fusion.

## Metrics

- Hit@1
- Hit@3
- Hit@5
- MRR@5

## Pilot evaluation

The initial pilot contains 12 natural-language questions from one
document. These results were used to validate the evaluation pipeline
before creating the final benchmark.

| Method | Hit@1 | Hit@3 | Hit@5 | MRR@5 |
| --- | ---: | ---: | ---: | ---: |
| Semantic | 0.583 | 0.833 | 0.833 | 0.708 |
| BM25 | 0.500 | 0.667 | 0.667 | 0.583 |
| Hybrid | 0.583 | 0.750 | 0.917 | 0.676 |

These results are preliminary and should not be interpreted as the
final benchmark because the pilot contains only 12 queries from a
single document.