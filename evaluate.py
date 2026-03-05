"""
RAG Evaluation Module
=====================
Implements RAGAS-inspired metrics to measure RAG pipeline quality — without
requiring the full ragas library (which has heavy dependencies).

Metrics implemented:
  1. Answer Faithfulness      — Is the answer grounded in the retrieved context?
  2. Answer Relevancy         — Does the answer actually address the question?
  3. Context Precision        — Are the retrieved chunks relevant to the question?
  4. Context Recall           — Does the context cover the ground-truth answer?
  5. Answer Correctness       — How similar is the answer to a ground-truth answer?

All metrics return a float in [0.0, 1.0]. Higher = better.

Usage:
    from evaluate import RAGEvaluator, EvalSample

    evaluator = RAGEvaluator(embedding_manager, llm_generator)

    sample = EvalSample(
        question="What is the attention mechanism?",
        answer=response.answer,
        contexts=[doc["content"] for doc in retrieved_docs],
        ground_truth="Attention maps query-key similarities to weight values.",  # optional
    )

    report = evaluator.evaluate(sample)
    print(report.summary())
    report.to_csv("eval_results.csv")
"""

import csv
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalSample:
    """One Q&A sample to evaluate."""
    question: str
    answer: str
    contexts: List[str]                   # retrieved chunk texts
    ground_truth: Optional[str] = None    # reference answer (needed for recall & correctness)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricResult:
    name: str
    score: float                          # 0.0 – 1.0
    explanation: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        bar = "█" * int(self.score * 20) + "░" * (20 - int(self.score * 20))
        return f"{self.name:<25} {bar}  {self.score:.3f}"


@dataclass
class EvalReport:
    """Aggregated evaluation output for one sample."""
    question: str
    answer: str
    metrics: List[MetricResult] = field(default_factory=list)
    latency_ms: float = 0.0
    timestamp: str = ""

    @property
    def scores(self) -> Dict[str, float]:
        return {m.name: m.score for m in self.metrics}

    @property
    def mean_score(self) -> float:
        if not self.metrics:
            return 0.0
        return float(np.mean([m.score for m in self.metrics]))

    def summary(self) -> str:
        lines = [
            "=" * 60,
            f"QUESTION : {self.question[:80]}",
            f"ANSWER   : {self.answer[:120]}...",
            "-" * 60,
            "METRICS:",
        ]
        for m in self.metrics:
            lines.append(f"  {repr(m)}")
        lines.append("-" * 60)
        lines.append(f"  {'MEAN SCORE':<25} {self.mean_score:.3f}")
        lines.append(f"  Latency: {self.latency_ms:.0f} ms")
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "question": self.question,
            "answer": self.answer,
            "latency_ms": self.latency_ms,
            "mean_score": self.mean_score,
            "timestamp": self.timestamp,
        }
        d.update(self.scores)
        return d

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Report saved to {path}")


@dataclass
class BatchEvalReport:
    """Aggregated report over multiple samples."""
    reports: List[EvalReport] = field(default_factory=list)

    @property
    def mean_scores(self) -> Dict[str, float]:
        if not self.reports:
            return {}
        all_scores: Dict[str, List[float]] = {}
        for r in self.reports:
            for name, score in r.scores.items():
                all_scores.setdefault(name, []).append(score)
        return {k: float(np.mean(v)) for k, v in all_scores.items()}

    @property
    def overall_mean(self) -> float:
        means = list(self.mean_scores.values())
        return float(np.mean(means)) if means else 0.0

    def summary(self) -> str:
        lines = [
            "=" * 60,
            f"BATCH EVALUATION  ({len(self.reports)} samples)",
            "-" * 60,
            "MEAN SCORES ACROSS ALL SAMPLES:",
        ]
        for name, score in self.mean_scores.items():
            bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            lines.append(f"  {name:<25} {bar}  {score:.3f}")
        lines.append("-" * 60)
        overall_bar = "█" * int(self.overall_mean * 20) + "░" * (20 - int(self.overall_mean * 20))
        lines.append(f"  {'OVERALL MEAN':<25} {overall_bar}  {self.overall_mean:.3f}")
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_csv(self, path: str) -> None:
        if not self.reports:
            return
        rows = [r.to_dict() for r in self.reports]
        fieldnames = list(rows[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"Batch report saved to {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Individual Metric Implementations
# ─────────────────────────────────────────────────────────────────────────────

class FaithfulnessMetric:
    """
    Measures whether the answer is factually grounded in the retrieved context.

    Approach (LLM-based):
      - Ask the LLM to extract claims from the answer.
      - For each claim, ask if it can be inferred from the context.
      - Score = supported_claims / total_claims

    Falls back to embedding-based overlap if no LLM is provided.
    """

    CLAIM_EXTRACTION_PROMPT = """Extract all factual claims from the following answer.
Return ONLY a JSON array of short claim strings. No explanation.

Answer:
{answer}

JSON array of claims:"""

    CLAIM_VERIFICATION_PROMPT = """Given the context below, determine if the claim is supported.
Answer with ONLY "yes" or "no".

Context:
{context}

Claim: {claim}

Supported (yes/no):"""

    def __init__(self, embedding_manager=None, generator=None):
        self.embedder = embedding_manager
        self.generator = generator

    def score(self, sample: EvalSample) -> MetricResult:
        context = "\n\n".join(sample.contexts)

        if self.generator:
            return self._llm_faithfulness(sample, context)
        elif self.embedder:
            return self._embedding_faithfulness(sample, context)
        else:
            raise ValueError("FaithfulnessMetric requires either an embedder or generator.")

    def _llm_faithfulness(self, sample: EvalSample, context: str) -> MetricResult:
        """Full LLM-based faithfulness check."""
        try:
            # Step 1: Extract claims
            claim_prompt = self.CLAIM_EXTRACTION_PROMPT.format(answer=sample.answer)
            claim_response = self.generator.backend.complete(
                claim_prompt, self.generator.config
            )
            raw_text = claim_response.answer.strip()

            # Parse JSON safely
            start = raw_text.find("[")
            end = raw_text.rfind("]") + 1
            claims = json.loads(raw_text[start:end]) if start >= 0 else []

            if not claims:
                return MetricResult(
                    name="faithfulness",
                    score=0.0,
                    explanation="Could not extract claims from answer.",
                )

            # Step 2: Verify each claim against context
            supported = 0
            verdicts = []
            for claim in claims[:10]:  # cap at 10 to limit API calls
                verify_prompt = self.CLAIM_VERIFICATION_PROMPT.format(
                    context=context[:3000], claim=claim
                )
                verdict_response = self.generator.backend.complete(
                    verify_prompt, self.generator.config
                )
                verdict = verdict_response.answer.strip().lower()
                is_supported = verdict.startswith("yes")
                if is_supported:
                    supported += 1
                verdicts.append({"claim": claim, "supported": is_supported})

            score = supported / len(claims)
            return MetricResult(
                name="faithfulness",
                score=round(score, 4),
                explanation=f"{supported}/{len(claims)} claims supported by context.",
                raw={"claims": verdicts},
            )

        except Exception as e:
            logger.warning(f"LLM faithfulness failed, falling back to embedding: {e}")
            return self._embedding_faithfulness(sample, context)

    def _embedding_faithfulness(self, sample: EvalSample, context: str) -> MetricResult:
        """
        Embedding-based faithfulness proxy.
        Measures semantic overlap between answer sentences and context.
        """
        sentences = [s.strip() for s in sample.answer.split(".") if len(s.strip()) > 10]
        if not sentences:
            return MetricResult(name="faithfulness", score=0.0, explanation="Empty answer.")

        answer_embs = self.embedder.embed(sentences)
        context_chunks = [c for c in sample.contexts if c.strip()]
        context_embs = self.embedder.embed(context_chunks)

        # For each sentence, find max similarity to any context chunk
        sims = answer_embs @ context_embs.T  # shape (n_sentences, n_chunks)
        max_sims = sims.max(axis=1)
        score = float(np.mean(max_sims))

        return MetricResult(
            name="faithfulness",
            score=round(min(max(score, 0.0), 1.0), 4),
            explanation=f"Mean max cosine sim of answer sentences to context: {score:.3f}",
            raw={"per_sentence_max_sim": max_sims.tolist()},
        )


class AnswerRelevancyMetric:
    """
    Measures whether the answer is relevant to the question.

    Approach (embedding-based):
      - Generate N paraphrased questions from the answer using LLM.
      - Measure cosine similarity between original question and each generated question.
      - Score = mean similarity.

    Falls back to direct question-answer cosine similarity without LLM.
    """

    QUESTION_GEN_PROMPT = """Given the following answer, generate {n} different questions
that this answer could be responding to. Return ONLY a JSON array of question strings.

Answer:
{answer}

JSON array of questions:"""

    def __init__(self, embedding_manager, generator=None, n_questions: int = 3):
        self.embedder = embedding_manager
        self.generator = generator
        self.n_questions = n_questions

    def score(self, sample: EvalSample) -> MetricResult:
        if self.generator:
            return self._llm_relevancy(sample)
        return self._embedding_relevancy(sample)

    def _llm_relevancy(self, sample: EvalSample) -> MetricResult:
        try:
            prompt = self.QUESTION_GEN_PROMPT.format(
                n=self.n_questions, answer=sample.answer
            )
            response = self.generator.backend.complete(prompt, self.generator.config)
            raw = response.answer.strip()
            start, end = raw.find("["), raw.rfind("]") + 1
            gen_questions = json.loads(raw[start:end]) if start >= 0 else []

            if not gen_questions:
                return self._embedding_relevancy(sample)

            orig_emb = self.embedder.embed_query(sample.question)
            gen_embs = self.embedder.embed(gen_questions)
            sims = gen_embs @ orig_emb
            score = float(np.mean(sims))

            return MetricResult(
                name="answer_relevancy",
                score=round(min(max(score, 0.0), 1.0), 4),
                explanation=f"Mean sim of {len(gen_questions)} generated questions to original.",
                raw={"generated_questions": gen_questions, "similarities": sims.tolist()},
            )
        except Exception as e:
            logger.warning(f"LLM answer relevancy failed, falling back: {e}")
            return self._embedding_relevancy(sample)

    def _embedding_relevancy(self, sample: EvalSample) -> MetricResult:
        q_emb = self.embedder.embed_query(sample.question)
        a_emb = self.embedder.embed_query(sample.answer)
        score = float(q_emb @ a_emb)
        return MetricResult(
            name="answer_relevancy",
            score=round(min(max(score, 0.0), 1.0), 4),
            explanation="Direct cosine similarity between question and answer embeddings.",
        )


class ContextPrecisionMetric:
    """
    Measures whether the retrieved contexts are actually relevant to the question.

    Score = proportion of retrieved chunks with similarity >= threshold to the question.
    """

    def __init__(self, embedding_manager, relevance_threshold: float = 0.3):
        self.embedder = embedding_manager
        self.threshold = relevance_threshold

    def score(self, sample: EvalSample) -> MetricResult:
        if not sample.contexts:
            return MetricResult(name="context_precision", score=0.0,
                                explanation="No contexts provided.")

        q_emb = self.embedder.embed_query(sample.question)
        ctx_embs = self.embedder.embed(sample.contexts)
        sims = ctx_embs @ q_emb

        relevant_count = int(np.sum(sims >= self.threshold))
        precision = relevant_count / len(sample.contexts)

        return MetricResult(
            name="context_precision",
            score=round(precision, 4),
            explanation=(
                f"{relevant_count}/{len(sample.contexts)} chunks above "
                f"threshold={self.threshold}. Mean sim: {np.mean(sims):.3f}"
            ),
            raw={"chunk_similarities": sims.tolist(), "threshold": self.threshold},
        )


class ContextRecallMetric:
    """
    Measures whether the retrieved context covers the ground-truth answer.

    Requires ground_truth. Score = semantic coverage of ground_truth by contexts.
    """

    def __init__(self, embedding_manager):
        self.embedder = embedding_manager

    def score(self, sample: EvalSample) -> MetricResult:
        if not sample.ground_truth:
            return MetricResult(
                name="context_recall",
                score=float("nan"),
                explanation="Skipped: no ground_truth provided.",
            )

        gt_sentences = [s.strip() for s in sample.ground_truth.split(".") if len(s.strip()) > 10]
        if not gt_sentences:
            return MetricResult(name="context_recall", score=0.0,
                                explanation="Empty ground truth.")

        gt_embs = self.embedder.embed(gt_sentences)
        ctx_embs = self.embedder.embed(sample.contexts)

        # For each ground-truth sentence, find max similarity to any context chunk
        sims = gt_embs @ ctx_embs.T
        max_sims = sims.max(axis=1)

        # A sentence is "recalled" if max similarity > 0.5
        recall_threshold = 0.5
        recalled = int(np.sum(max_sims >= recall_threshold))
        score = recalled / len(gt_sentences)

        return MetricResult(
            name="context_recall",
            score=round(score, 4),
            explanation=(
                f"{recalled}/{len(gt_sentences)} ground-truth sentences "
                f"covered by context (threshold={recall_threshold})."
            ),
            raw={"per_sentence_max_sim": max_sims.tolist()},
        )


class AnswerCorrectnessMetric:
    """
    Measures semantic similarity between the generated answer and ground truth.

    Requires ground_truth.
    Score = cosine similarity between their embeddings.
    """

    def __init__(self, embedding_manager):
        self.embedder = embedding_manager

    def score(self, sample: EvalSample) -> MetricResult:
        if not sample.ground_truth:
            return MetricResult(
                name="answer_correctness",
                score=float("nan"),
                explanation="Skipped: no ground_truth provided.",
            )

        answer_emb = self.embedder.embed_query(sample.answer)
        gt_emb = self.embedder.embed_query(sample.ground_truth)
        score = float(answer_emb @ gt_emb)

        return MetricResult(
            name="answer_correctness",
            score=round(min(max(score, 0.0), 1.0), 4),
            explanation="Cosine similarity between generated answer and ground truth.",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main Evaluator
# ─────────────────────────────────────────────────────────────────────────────

class RAGEvaluator:
    """
    Orchestrates all metrics for a single sample or a batch.

    Args:
        embedding_manager: EmbeddingManager instance (required).
        generator:         RAGGenerator instance (optional, enables LLM-based metrics).
        metrics:           List of metric names to run. Defaults to all.

    Available metrics:
        "faithfulness", "answer_relevancy", "context_precision",
        "context_recall", "answer_correctness"

    Example:
        evaluator = RAGEvaluator(embedding_manager=embedder, generator=gen)
        report = evaluator.evaluate(sample)
        print(report.summary())
    """

    ALL_METRICS = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "answer_correctness",
    ]

    def __init__(
        self,
        embedding_manager,
        generator=None,
        metrics: Optional[List[str]] = None,
        context_relevance_threshold: float = 0.3,
    ):
        self.embedder = embedding_manager
        self.generator = generator
        self.active_metrics = metrics or self.ALL_METRICS

        # Instantiate metric objects
        self._metrics = {
            "faithfulness": FaithfulnessMetric(embedding_manager, generator),
            "answer_relevancy": AnswerRelevancyMetric(embedding_manager, generator),
            "context_precision": ContextPrecisionMetric(
                embedding_manager, context_relevance_threshold
            ),
            "context_recall": ContextRecallMetric(embedding_manager),
            "answer_correctness": AnswerCorrectnessMetric(embedding_manager),
        }

        logger.info(
            f"RAGEvaluator initialized | metrics={self.active_metrics} | "
            f"LLM={'enabled' if generator else 'disabled (embedding fallback)'}"
        )

    def evaluate(self, sample: EvalSample) -> EvalReport:
        """
        Run all active metrics on a single sample.

        Returns:
            EvalReport with per-metric scores and explanations.
        """
        import datetime

        start = time.time()
        results: List[MetricResult] = []

        for metric_name in self.active_metrics:
            if metric_name not in self._metrics:
                logger.warning(f"Unknown metric '{metric_name}', skipping.")
                continue

            try:
                result = self._metrics[metric_name].score(sample)
                results.append(result)
                logger.info(f"  {metric_name}: {result.score:.4f}")
            except Exception as e:
                logger.error(f"  {metric_name}: FAILED — {e}")
                results.append(MetricResult(
                    name=metric_name,
                    score=float("nan"),
                    explanation=f"Error: {e}",
                ))

        latency_ms = (time.time() - start) * 1000
        return EvalReport(
            question=sample.question,
            answer=sample.answer,
            metrics=results,
            latency_ms=round(latency_ms, 1),
            timestamp=datetime.datetime.utcnow().isoformat(),
        )

    def evaluate_batch(
        self,
        samples: List[EvalSample],
        output_csv: Optional[str] = None,
    ) -> BatchEvalReport:
        """
        Evaluate multiple samples and optionally save results to CSV.

        Args:
            samples:    List of EvalSample instances.
            output_csv: If provided, writes results to this CSV path.

        Returns:
            BatchEvalReport with per-sample and aggregated scores.
        """
        logger.info(f"Starting batch evaluation of {len(samples)} samples...")
        reports = []

        for i, sample in enumerate(samples, 1):
            logger.info(f"Evaluating sample {i}/{len(samples)}: '{sample.question[:60]}...'")
            report = self.evaluate(sample)
            reports.append(report)

        batch_report = BatchEvalReport(reports=reports)
        logger.info("\n" + batch_report.summary())

        if output_csv:
            batch_report.to_csv(output_csv)

        return batch_report


# ─────────────────────────────────────────────────────────────────────────────
# Quick-start helper
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_pipeline_response(
    pipeline,
    question: str,
    ground_truth: Optional[str] = None,
    top_k: int = 5,
    score_threshold: float = 0.2,
    use_llm_metrics: bool = True,
) -> EvalReport:
    """
    Convenience function: run a query through the pipeline and evaluate it.

    Args:
        pipeline:       RAGPipeline instance.
        question:       User question.
        ground_truth:   Reference answer (optional, enables recall & correctness).
        top_k:          Retrieval top-K.
        score_threshold: Min similarity for retrieval.
        use_llm_metrics: If True, use LLM for faithfulness + relevancy.

    Returns:
        EvalReport.

    Example:
        from evaluate import evaluate_pipeline_response
        report = evaluate_pipeline_response(
            pipeline=pipeline,
            question="What is the attention mechanism?",
            ground_truth="Attention weights input positions using query-key similarities.",
        )
        print(report.summary())
    """
    # Run retrieval + generation
    retrieved = pipeline.retriever.retrieve(
        question, top_k=top_k, score_threshold=score_threshold
    )
    response = pipeline.generator.generate(question, retrieved)

    sample = EvalSample(
        question=question,
        answer=response.answer,
        contexts=[doc["content"] for doc in retrieved],
        ground_truth=ground_truth,
    )

    evaluator = RAGEvaluator(
        embedding_manager=pipeline.embedder,
        generator=pipeline.generator if use_llm_metrics else None,
    )

    return evaluator.evaluate(sample)