# Retrieval Evaluation

RAGOps includes a manually labelled retrieval benchmark for comparing
three retrieval strategies:

- Semantic Search
- BM25
- Hybrid Search using Reciprocal Rank Fusion (RRF)

The goal of the benchmark is to measure how reliably each method retrieves
useful evidence from the indexed document collection.

## Evaluation dataset

The final benchmark contains **40 natural-language queries** across
three documents:

- `Cat`
- `Black_hole`
- `Cave_painting`

Queries were written to resemble questions a real user might ask rather
than artificial keyword-matching tests.

The dataset contains both:

- `direct` queries, where the wording is relatively close to the source;
- `paraphrase` queries, where the same information is requested using
  different wording.

Each query includes one or more manually labelled relevant chunk IDs.

Example:

```json
{
  "id": "q012",
  "query": "How long do domestic cats usually live?",
  "document_id": "Cat",
  "category": "direct",
  "relevant_chunk_ids": ["Cat:55"]
}
```

Multiple chunks can be labelled as relevant when the same evidence appears
in different parts of a document or when overlapping chunks both contain
sufficient information to answer the question.

A chunk is considered relevant when it contains enough evidence to answer
the question correctly or provides a substantial part of the expected
answer.

## Evaluation setup

Each query is executed against the complete set of evaluation documents.

The retriever is therefore not told which document contains the answer.
Relevant chunks must compete against chunks from all three documents.

The same indexed corpus and retrieval implementations used by the RAGOps
application are used by the evaluator.

Evaluation is performed with a maximum retrieval depth of 5 results.

## Metrics

The benchmark reports:

* **Hit@1** — proportion of queries with at least one relevant chunk ranked first.
* **Hit@3** — proportion of queries with at least one relevant chunk in the top 3.
* **Hit@5** — proportion of queries with at least one relevant chunk in the top 5.
* **MRR@5** — Mean Reciprocal Rank of the first relevant result within the top 5.

MRR gives more credit when relevant evidence appears closer to the top of
the ranking.

## Pilot evaluation

The evaluation pipeline was first tested using 12 natural-language queries
from a single document.

| Method   | Hit@1 | Hit@3 | Hit@5 | MRR@5 |
| -------- | ----: | ----: | ----: | ----: |
| Semantic | 0.583 | 0.833 | 0.833 | 0.708 |
| BM25     | 0.500 | 0.667 | 0.667 | 0.583 |
| Hybrid   | 0.583 | 0.750 | 0.917 | 0.676 |

These results were used only to validate the evaluation pipeline before
building the larger benchmark.

## Final benchmark results

The final evaluation contains 40 queries across three documents.

| Method   |     Hit@1 |     Hit@3 |     Hit@5 |     MRR@5 |
| -------- | --------: | --------: | --------: | --------: |
| Semantic | **0.500** |     0.600 |     0.700 | **0.574** |
| BM25     |     0.450 | **0.625** |     0.700 |     0.545 |
| Hybrid   |     0.400 |     0.575 | **0.750** |     0.516 |

## Interpretation

No retrieval method dominated every metric.

Semantic Search achieved the best **Hit@1** and **MRR@5**, meaning that it
was generally the strongest method at placing relevant evidence near the
top of the ranking.

BM25 remained highly competitive and achieved the best **Hit@3**. It was
particularly useful for queries where important terminology in the query
also appeared directly in the source text.

Hybrid Search achieved the highest **Hit@5**, retrieving relevant evidence
for 75% of the evaluation queries within the first five results.

However, its lower MRR@5 shows that relevant evidence was often ranked
lower than with Semantic Search.

This demonstrates an important trade-off in the current system:

* Semantic Search provides the strongest ranking quality.
* BM25 provides strong lexical retrieval and complements semantic search.
* Hybrid Search improves top-5 coverage, but does not always improve ranking
  quality.

## Error analysis

Manual inspection of failed queries revealed an important limitation of the
current exact-chunk Reciprocal Rank Fusion implementation.

Semantic Search and BM25 sometimes retrieve **different chunks containing
valid evidence for the same question**.

For example, one relevant chunk may rank highly in Semantic Search while a
different relevant chunk ranks highly in BM25.

Because RRF combines rankings using exact `chunk_id` identity, these two
relevant chunks do not reinforce each other.

At the same time, an unrelated or weaker chunk appearing moderately high in
both rankings receives contributions from both retrieval methods and may be
promoted above individually stronger results.

As a result, Hybrid Search can sometimes demote a highly relevant result
even when one of the underlying retrieval methods ranked it very strongly.

This explains why Hybrid Search achieved the highest Hit@5 while producing
a lower MRR@5 than Semantic Search.

## Ground-truth review

An initial evaluation exposed another methodological issue: some questions
had several valid answer locations while only one chunk had originally been
labelled as relevant.

The benchmark was manually reviewed so that all chunks containing sufficient
supporting evidence could be included in the relevance labels.

Relevant chunks were added based on manual inspection of their content, not
because a retrieval method happened to return them.

This prevents valid retrieved evidence from being incorrectly counted as a
failure while avoiding tuning the ground truth to favour a particular
retrieval method.

## Limitations

This benchmark is intentionally small and is intended for project-level
evaluation rather than as a general information-retrieval benchmark.

Current limitations include:

* 40 evaluation queries;
* three source documents;
* only direct and paraphrased query categories;
* relevance is evaluated at chunk level;
* MRR only considers the first relevant result;
* multi-chunk answer completeness is not measured;
* Hybrid Search uses standard exact-chunk RRF without learned or tuned
  fusion weights.

The benchmark is therefore best interpreted as a reproducible comparison of
the retrieval strategies used by RAGOps rather than a claim about retrieval
performance in general.

## Running the evaluation

From the project root:

```bash
python -m evaluation.evaluate
```

The evaluator loads `evaluation/queries.json`, runs all three retrieval
methods against the evaluation corpus, and prints the rank of the first
relevant chunk together with the aggregate metrics.