from __future__ import annotations

import textgrad as tg
from textgrad.optimizer import TextualGradientDescent
from reward_model import TPORewardModel


import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from urllib.parse import urlparse
from urllib.request import urlopen

############################################################
# Prompt Templates
############################################################

EVALUATION_SYS_TEMPLATE = """You are a language model tasked with evaluating a chosen response by comparing it with a rejected response to a user query. Analyze the strengths and weaknesses of each response, step by step, and explain why one is chosen or rejected.

**User Query**:
{query}

**Rejected Response**:
{rejected_response}

**Do NOT generate a response to the query. Be concise.** Below is the chosen response."""

EVALUATION_SYS_TEMPLATE_REVISION = """You are a language model tasked with evaluating a model response to a user query. Analyze the strengths and weaknesses of the response, step by step.

**User Query**:
{query}

**Do NOT generate a response to the query. Be concise.** Below is the model response."""

############################################################
# Sample preparation utilities
############################################################

PreparedSample = Dict[str, Any]


def _is_url(path: str) -> bool:
    try:
        parsed = urlparse(path)
    except ValueError:
        return False
    return parsed.scheme in {"http", "https"} and parsed.netloc != ""


def _resolve_image_reference(image_ref: Union[str, bytes, bytearray]) -> bytes:
    if isinstance(image_ref, (bytes, bytearray)):
        return bytes(image_ref)

    if not isinstance(image_ref, str):
        raise TypeError(f"Unsupported image reference type: {type(image_ref)}")

    if _is_url(image_ref):
        with urlopen(image_ref) as response:  # type: ignore[arg-type]
            return response.read()

    image_path = Path(image_ref).expanduser()
    if not image_path.is_file():
        raise FileNotFoundError(f"Unable to locate image at '{image_ref}'")

    return image_path.read_bytes()


def _default_sample_id(prompt: str, image_sources: Sequence[str]) -> str:
    digest_source = prompt + ("::" + "::".join(image_sources) if image_sources else "")
    return hashlib.sha256(digest_source.encode("utf-8")).hexdigest()[:16]


def _sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    def _sanitize(value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, list):
            sanitized_list = [
                _sanitize(v)
                for v in value
                if isinstance(v, (str, int, float, bool, list, dict)) or v is None
            ]
            return [v for v in sanitized_list if v is not None]
        if isinstance(value, dict):
            return {
                str(k): _sanitize(v)
                for k, v in value.items()
                if isinstance(k, (str, int, float, bool))
            }
        return str(value)

    return {str(k): _sanitize(v) for k, v in metadata.items()}


def prepare_sample(
    raw_sample: Union[str, Dict[str, Any]], sample_idx: Optional[int] = None
) -> PreparedSample:
    """Normalize raw data into a structured sample dictionary."""

    if isinstance(raw_sample, dict) and {"id", "prompt", "llm_input"}.issubset(
        raw_sample.keys()
    ):
        metadata = raw_sample.get("metadata", {})
        return {
            "id": str(raw_sample["id"]),
            "prompt": raw_sample["prompt"],
            "llm_input": raw_sample["llm_input"],
            "images": raw_sample.get("images", []),
            "metadata": _sanitize_metadata(metadata),
        }

    if isinstance(raw_sample, str):
        base_prompt = raw_sample
        sample_id = _default_sample_id(base_prompt, [])
        metadata: Dict[str, Any] = {"original_query": base_prompt}
        if sample_idx is not None:
            metadata["source_index"] = sample_idx
        return {
            "id": sample_id,
            "prompt": base_prompt,
            "llm_input": base_prompt,
            "images": [],
            "metadata": metadata,
        }

    if not isinstance(raw_sample, dict):
        raise TypeError("Each sample must be either a string or a dictionary.")

    working_copy = dict(raw_sample)

    prompt = (
        working_copy.pop("prompt", None)
        or working_copy.pop("query", None)
        or working_copy.pop("question", None)
        or working_copy.pop("instruction", None)
    )
    if prompt is None:
        raise ValueError(
            "Sample dictionary must include a 'prompt', 'query', 'question', or 'instruction' field."
        )

    context_text = working_copy.pop("context", None)
    context_text = context_text.strip() if isinstance(context_text, str) else None
    combined_prompt = f"{context_text}\n\n{prompt}" if context_text else prompt

    image_sources_raw = None
    for key in (
        "images",
        "image_paths",
        "image_path",
        "image",
        "image_urls",
        "image_url",
    ):
        if key in working_copy:
            image_sources_raw = working_copy.pop(key)
            break

    if image_sources_raw is None:
        image_sources: List[str] = []
    elif isinstance(image_sources_raw, (list, tuple)):
        image_sources = [str(item) for item in image_sources_raw]
    else:
        image_sources = [str(image_sources_raw)]

    image_payloads = [_resolve_image_reference(ref) for ref in image_sources]

    system_prompt = working_copy.pop("system_prompt", None)
    constraints = working_copy.pop("constraints", None)

    sample_id = str(
        working_copy.pop("id", None)
        or working_copy.pop("uid", None)
        or working_copy.pop("question_id", None)
        or working_copy.pop("query_id", None)
        or _default_sample_id(combined_prompt, image_sources)
    )

    metadata: Dict[str, Any] = {"original_query": prompt}
    if context_text:
        metadata["context"] = context_text
    if system_prompt:
        metadata["system_prompt"] = system_prompt
    if isinstance(constraints, (list, tuple)):
        metadata["constraints"] = [str(c) for c in constraints]
    if image_sources:
        metadata["image_refs"] = image_sources
    if sample_idx is not None:
        metadata["source_index"] = sample_idx

    metadata.update({k: v for k, v in working_copy.items() if isinstance(k, str)})

    sanitized_metadata = _sanitize_metadata(metadata)

    llm_input: Union[str, List[Union[str, bytes]]] = combined_prompt
    if image_payloads:
        llm_input = list(image_payloads) + [combined_prompt]

    return {
        "id": sample_id,
        "prompt": combined_prompt,
        "llm_input": llm_input,
        "images": image_payloads,
        "metadata": sanitized_metadata,
    }


def _ensure_list(responses: Union[str, Sequence[str], None]) -> List[str]:
    if responses is None:
        return []
    if isinstance(responses, list):
        return responses
    if isinstance(responses, tuple):
        return list(responses)
    return [responses]


############################################################
# Caching Utilities
############################################################


def cache_scores(
    score_cache: dict,
    scores: Sequence[float],
    qa_pairs: Sequence[Union[Tuple[str, str], Dict[str, Any]]],
    index: int = -1,
) -> None:
    """
    Caches the reward model scores for a set of (question, answer) pairs.

    """
    for score, qa_pair in zip(scores, qa_pairs):
        if isinstance(qa_pair, dict):
            q = qa_pair.get("prompt") or qa_pair.get("query") or qa_pair.get("question")
            a = qa_pair.get("answer") or qa_pair.get("response")
        else:
            q, a = qa_pair

        if q is None or a is None:
            raise ValueError("Each QA pair must include both a prompt and an answer.")

        key = f"INDEX{index}<SEP>{q}<SEP>{a}"
        if key not in score_cache:
            score_cache[key] = float(score)


############################################################
# Best-of-N (BoN) Inference-time Alignment
############################################################


def run_test_time_training_bon(
    sample: Union[str, Dict[str, Any]],
    llm_engine,
    rm: TPORewardModel,
    gen_params: dict,
    **kwargs,
) -> Dict[str, Any]:
    """
    Runs the Best-of-N (BoN) sampling approach at test time, without iterative refinement.
    Samples responses, computes reward model scores, and returns a cache of scores.

    :param query: The user query (string).
    :param llm_engine: LLM inference engine from textgrad.get_engine().
    :param rm: TPORewardModel instance for reward scoring.
    :param gen_params: Generation parameters for the LLM engine.
    :return: Dictionary of all scores keyed by 'INDEX-1<SEP>{q}<SEP>{a}'.
    """
    prepared = prepare_sample(sample)
    metadata = prepared.get("metadata", {})
    system_prompt = metadata.get("system_prompt")

    tg.set_backward_engine(llm_engine, override=True)

    all_scores: Dict[str, float] = {}
    sample_responses = _ensure_list(
        llm_engine(prepared["llm_input"], system_prompt=system_prompt, **gen_params)
    )
    sample_qas = [
        {"prompt": prepared["prompt"], "answer": resp} for resp in sample_responses
    ]

    sample_scores = rm.perform_rm(sample_qas)
    cache_scores(all_scores, sample_scores, sample_qas, index=-1)

    result: Dict[str, Any] = {
        "sample_id": prepared["id"],
        "prompt": prepared["prompt"],
        "scores": all_scores,
        "mode": "bon",
    }

    return result


############################################################
# Test-time Preference Optimization (TPO)
############################################################


def run_test_time_training_tpo(
    sample: Union[str, Dict[str, Any]],
    llm_engine,
    rm: TPORewardModel,
    gen_params: dict,
    tpo_mode: str = "tpo",
    max_iters: int = 5,
) -> Dict[str, Any]:
    """
    Runs the Test-time Preference Optimization (TPO) process by repeatedly
    refining the chosen response according to reward model feedback.

    :param query: The user query (string).
    :param llm_engine: LLM inference engine from textgrad.
    :param rm: TPORewardModel for scoring.
    :param gen_params: Generation parameters for sampling responses.
    :param tpo_mode: Mode of TPO - 'tpo', 'revision', or 'bon'.
    :param max_iters: Number of optimization iterations to perform.
    :return: Dictionary of all scored (query, answer) pairs.
    """
    prepared = prepare_sample(sample)
    metadata = prepared.get("metadata", {})
    system_prompt = metadata.get("system_prompt")

    if prepared["images"] and not getattr(llm_engine, "is_multimodal", False):
        raise ValueError(
            "The provided engine is not multimodal, but the sample includes image inputs."
        )

    tg.set_backward_engine(llm_engine, override=True)
    all_scores: Dict[str, float] = {}

    def _score_and_select(
        responses: Sequence[str], iteration_index: int
    ) -> Tuple[str, str]:
        sanitized: List[str] = []
        for resp in responses:
            if resp is None:
                continue
            if not isinstance(resp, str):
                resp = str(resp)
            candidate = resp.strip()
            if not candidate:
                continue
            sanitized.append(candidate)

        if not sanitized:
            return "", ""

        qa_records = [
            {"prompt": prepared["prompt"], "answer": resp} for resp in sanitized
        ]
        sample_scores = rm.perform_rm(qa_records)
        cache_scores(all_scores, sample_scores, qa_records, index=iteration_index)

        merged: List[Tuple[str, str, float]] = []
        for key, value in all_scores.items():
            parts = key.split("<SEP>")
            if len(parts) < 3:
                continue
            merged.append((parts[1], parts[2], value))

        if not merged:
            return "", ""

        # Identify best and worst samples from the updated cache
        sample_scores_vals = [m[2] for m in merged]
        sample_qas_vals = [(m[0], m[1]) for m in merged]

        contrastive_responses, _ = rm.get_contrastive_samples(
            sample_scores_vals, sample_qas_vals
        )

        return contrastive_responses["best"], contrastive_responses["worst"]

    # 1) Initial sampling for candidates
    init_responses = _ensure_list(
        llm_engine(prepared["llm_input"], system_prompt=system_prompt, **gen_params)
    )
    if not init_responses:
        raise ValueError("LLM engine did not produce any initial responses.")

    chosen_resp_text, rej_resp_text = _score_and_select(
        init_responses, iteration_index=-1
    )
    if chosen_resp_text == "":
        raise ValueError(
            "Unable to identify a chosen response from the initial samples."
        )

    # 2) Define the variable to be optimized
    response_role = (
        "a model response to a user query"
        if tpo_mode == "revision"
        else "a chosen response to a user query"
    )

    response = tg.Variable(
        chosen_resp_text,
        requires_grad=True,
        role_description=response_role,
    )

    # 3) Constraints for textual updates
    default_constraints = (
        ["Only generate a model response."]  # revision
        if tpo_mode == "revision"
        else [
            "Only generate a chosen response.",
            "Do NOT generate a rejected response.",
        ]
    )

    constraints = metadata.get("constraints") or default_constraints

    # 4) Create the TPO optimizer
    optimizer = TextualGradientDescent(
        engine=llm_engine,
        parameters=[response],
        constraints=constraints,
    )

    # 5) Define the loss function (TextLoss)
    if tpo_mode == "revision":
        # No rejected sample provided
        evaluation_sys_text = EVALUATION_SYS_TEMPLATE_REVISION.format(
            query=prepared["prompt"]
        )
    else:
        # TPO mode, includes rejected response
        evaluation_sys_text = EVALUATION_SYS_TEMPLATE.format(
            query=prepared["prompt"],
            rejected_response=rej_resp_text,
        )
    loss_fn = tg.TextLoss(evaluation_sys_text)

    # 6) Start test-time training loop
    iterations_run = 0
    last_rejected = rej_resp_text

    for iteration in range(max_iters):
        optimizer.zero_grad()

        # 6.1) Compute textual loss
        loss = loss_fn(response)

        # 6.2) Compute textual gradients
        loss.backward()

        # 6.3) Update variable using textual gradients
        new_responses = _ensure_list(optimizer.step(**gen_params))
        if not new_responses:
            break

        # 6.4) Update cache with new responses, get chosen and rejected
        chosen_resp_text, rej_resp_text = _score_and_select(
            new_responses, iteration_index=iteration
        )
        if chosen_resp_text == "":
            break

        # 6.5) Update the variable's content
        response.set_value(chosen_resp_text)
        last_rejected = rej_resp_text

        # 6.6) Update the loss function if needed
        if tpo_mode == "tpo":
            # In TPO mode, update the rejected response for the next iteration
            evaluation_sys_text = EVALUATION_SYS_TEMPLATE.format(
                query=prepared["prompt"],
                rejected_response=rej_resp_text,
            )
        loss_fn = tg.TextLoss(evaluation_sys_text)
        iterations_run += 1

    result: Dict[str, Any] = {
        "sample_id": prepared["id"],
        "prompt": prepared["prompt"],
        "scores": all_scores,
        "mode": tpo_mode,
        "final_response": response.get_value(),
        "iterations": iterations_run,
    }
    if last_rejected:
        result["final_rejected_response"] = last_rejected
    # if metadata:
    # result["metadata"] = _sanitize_metadata(metadata)

    return result
