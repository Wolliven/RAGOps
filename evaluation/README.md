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

## Results by document

Retrieval performance varied substantially depending on the source document.

### Black_hole

| Method | Hit@1 | Hit@3 | Hit@5 | MRR@5 |
| --- | ---: | ---: | ---: | ---: |
| Semantic | **0.571** | 0.643 | **0.786** | **0.639** |
| BM25 | 0.429 | **0.714** | 0.714 | 0.560 |
| Hybrid | 0.429 | **0.714** | **0.786** | 0.562 |

Semantic Search produced the strongest ranking quality, while Hybrid matched
its top-5 coverage.

### Cat

| Method | Hit@1 | Hit@3 | Hit@5 | MRR@5 |
| --- | ---: | ---: | ---: | ---: |
| Semantic | **0.667** | **0.750** | 0.750 | **0.708** |
| BM25 | 0.500 | 0.667 | 0.750 | 0.600 |
| Hybrid | 0.583 | 0.667 | **1.000** | 0.690 |

Hybrid Search retrieved relevant evidence within the first five results for
all Cat queries, while Semantic Search generally ranked relevant evidence
higher.

### Cave_painting

| Method | Hit@1 | Hit@3 | Hit@5 | MRR@5 |
| --- | ---: | ---: | ---: | ---: |
| Semantic | 0.286 | 0.429 | 0.571 | 0.393 |
| BM25 | **0.429** | **0.500** | **0.643** | **0.485** |
| Hybrid | 0.214 | 0.357 | 0.500 | 0.321 |

BM25 performed best on the Cave_painting document. This corpus contains many
semantically related passages using similar terminology, which made semantic
ranking less discriminative. Lexical matching was more effective for several
fact-oriented questions in this document.

These differences show that retrieval effectiveness depends not only on the
query but also on the structure and vocabulary of the source corpus.

## Results by query category

The benchmark contains 26 direct queries and 14 paraphrased queries.

### Direct queries

| Method | Hit@1 | Hit@3 | Hit@5 | MRR@5 |
| --- | ---: | ---: | ---: | ---: |
| Semantic | **0.538** | 0.654 | 0.731 | 0.615 |
| BM25 | **0.538** | **0.769** | **0.846** | **0.663** |
| Hybrid | **0.538** | 0.615 | 0.808 | 0.623 |

BM25 was the strongest method for direct queries, where important terms in
the question often appeared explicitly in the source text.

### Paraphrased queries

| Method | Hit@1 | Hit@3 | Hit@5 | MRR@5 |
| --- | ---: | ---: | ---: | ---: |
| Semantic | **0.429** | **0.500** | **0.643** | **0.496** |
| BM25 | 0.286 | 0.357 | 0.429 | 0.327 |
| Hybrid | 0.143 | **0.500** | **0.643** | 0.318 |

Semantic Search performed substantially better than BM25 at ranking
paraphrased queries near the top of the results.

Hybrid Search eventually matched Semantic Search at Hit@3 and Hit@5, but its
much lower Hit@1 and MRR@5 show that fusion often pushed relevant semantic
matches further down the ranking.

## Interpretation

The evaluation did not produce a single retrieval strategy that dominated
across all query types and documents.

Semantic Search achieved the strongest overall ranking quality, with the
highest Hit@1 (0.500) and MRR@5 (0.574). Its advantage was particularly clear
for paraphrased queries, where semantic similarity helped retrieve evidence
despite differences in wording.

BM25 achieved the highest Hit@3 overall (0.625) and was the strongest method
for direct queries. It also substantially outperformed the other methods on
the Cave_painting corpus, showing that lexical retrieval remains valuable for
fact-oriented and terminology-heavy documents.

Hybrid Search achieved the highest overall Hit@5 (0.750), demonstrating that
combining both retrieval methods improved evidence coverage. However, its
lower MRR@5 (0.516) indicates that this additional coverage came at the cost
of ranking quality.

The results support exposing multiple retrieval strategies rather than
assuming that one method is universally superior.

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