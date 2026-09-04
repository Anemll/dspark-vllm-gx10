#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Deterministic coding proxy, fixed-history prefix replay, and tool smoke.

This synthetic workload is not NVIDIA SpeedBench or a coding quality score.
The first coding prompt is exact-length only after the serving /tokenize API
verifies its fully templated messages. Inference usage must agree as well.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import threading
import time
import urllib.request

try:
    from .benchmark_dsv4_api import (
        add_common_arguments, distribution, finish_trial, json_hash, make_body, new_report,
        run_stream, summarize_trials, validate_arguments, write_report,
    )
except ImportError:
    from benchmark_dsv4_api import (
        add_common_arguments, distribution, finish_trial, json_hash, make_body, new_report,
        run_stream, summarize_trials, validate_arguments, write_report,
    )


FIXTURE_PATH = Path(__file__).with_name("fixtures") / "agent-v1.json"


def request_json(base_url: str, endpoint: str, body: dict | None = None, timeout: float = 30) -> dict:
    request = urllib.request.Request(
        base_url.rstrip("/") + endpoint,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"} if body is not None else {},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def token_evidence(body: dict, tokenize) -> dict:
    request = {key: body[key] for key in ("model", "messages", "tools", "chat_template_kwargs") if key in body}
    response = tokenize(request)
    tokens = response.get("tokens")
    count = response.get("count")
    if not isinstance(tokens, list) or not tokens or any(type(token) is not int for token in tokens):
        raise ValueError("/tokenize did not return nonempty integer token IDs")
    if type(count) is not int or count != len(tokens):
        raise ValueError("/tokenize count disagrees with token IDs")
    return {"count": count, "token_ids_sha256": json_hash(tokens), "tokenize_request_sha256": json_hash(request)}


def fit_coding_body(model: str, target: int, max_tokens: int, seed: int, identity: str, fixture: dict, tokenize) -> tuple[dict, dict]:
    """Bounded character search; fail rather than label an approximate 8K exact."""
    nonce = json_hash([fixture["version"], seed, identity])[:32]
    prefix = f"Case {nonce}\n" + fixture["coding_request"]
    corpus = []
    length = 0
    index = 0
    while length < target * 12:
        snippet = fixture["snippet_template"].replace("{index}", str(index))
        corpus.append(snippet)
        length += len(snippet)
        index += 1
    code = "".join(corpus)
    base = make_body(model, max_tokens, seed)
    observed = {}

    def evaluate(characters):
        if characters not in observed:
            if len(observed) >= 128:
                raise ValueError("could not find exact token length within 128 tokenizer calls")
            body = copy.deepcopy(base)
            body["messages"] = [
                {"role": "system", "content": fixture["system"]},
                {"role": "user", "content": prefix + code[:characters]},
            ]
            observed[characters] = (body, token_evidence(body, tokenize))
        return observed[characters]

    low, high = 0, len(code)
    if evaluate(low)[1]["count"] > target or evaluate(high)[1]["count"] < target:
        raise ValueError("target token length is outside the generated fixture range")
    while low <= high:
        middle = (low + high) // 2
        body, evidence = evaluate(middle)
        count = evidence["count"]
        if count == target:
            return body, evidence
        if count < target:
            low = middle + 1
        else:
            high = middle - 1
    # Token merges can make character-prefix counts locally non-monotonic.
    for characters in range(max(0, low - 48), min(len(code), low + 48) + 1):
        body, evidence = evaluate(characters)
        if evidence["count"] == target:
            return body, evidence
    raise ValueError(f"could not verify an exact {target}-token templated prompt")


def conversation_bodies(base: dict, mode: str, fixture: dict) -> list[tuple[str, str, dict]]:
    turns = [("initial", "unique-prefix-intended", copy.deepcopy(base))]
    if mode == "multi-turn":
        turns.append(("exact-repeat", "repeat-prefix-intended", copy.deepcopy(base)))
        for phase, prompt in (("growing-turn", fixture["followup"]), ("divergent-suffix", fixture["divergent_followup"])):
            body = copy.deepcopy(base)
            body["messages"].extend([
                {"role": "assistant", "content": fixture["fixed_assistant"]},
                {"role": "user", "content": prompt},
            ])
            turns.append((phase, "shared-prefix-intended", body))
    return turns


def prepare_cases(args, levels: list[int], fixture: dict, tokenize) -> list[dict]:
    cases = []
    for concurrency in levels:
        for trial in range(1, args.trials + 1):
            for conversation in range(concurrency):
                identity = f"{concurrency}:{trial}:{conversation}"
                if args.mode == "tool-smoke":
                    body = make_body(args.model, args.max_tokens, args.seed)
                    body.update(messages=copy.deepcopy(fixture["tool_messages"]), tools=copy.deepcopy(fixture["tools"]), tool_choice="auto", ignore_eos=False)
                    turns = [("tool-smoke", "unspecified", body)]
                    initial_evidence = token_evidence(body, tokenize)
                else:
                    body, initial_evidence = fit_coding_body(args.model, args.target_tokens, args.max_tokens, args.seed, identity, fixture, tokenize)
                    turns = conversation_bodies(body, args.mode, fixture)
                for turn, (phase, cache_label, body) in enumerate(turns):
                    evidence = initial_evidence if turn < 2 else token_evidence(body, tokenize)
                    cases.append({
                        "concurrency": concurrency, "trial": trial, "conversation": conversation,
                        "turn": turn, "phase": phase, "cache_intent": cache_label,
                        "body": body, "request_sha256": json_hash(body),
                        "tokenization": evidence,
                        "expect": "tool" if args.mode == "tool-smoke" else "text",
                        "expected_tool": fixture["expected_tool"] if args.mode == "tool-smoke" else None,
                    })
    return cases


def run_conversation(cases: list[dict], args, on_result, runner=None) -> bool:
    runner = runner or run_stream
    first_content = None
    for case in cases:
        result = runner(
            args.base_url, args.model, args.max_tokens, case["conversation"],
            body=case["body"], timeout=args.timeout,
            expect=case["expect"], expected_tool=case["expected_tool"],
        )
        if result.prompt_tokens != case["tokenization"]["count"]:
            result.errors.append("inference prompt usage disagrees with /tokenize verification")
            result.ok = False
        record = asdict(result)
        record.update({key: case[key] for key in ("conversation", "turn", "phase", "cache_intent", "tokenization")})
        if case["phase"] == "exact-repeat":
            record["repeat_content_matches_initial"] = result.content == first_content
        if case["turn"] == 0:
            first_content = result.content
        on_result(record)
        if not result.ok:
            return False
    return True


def phase_summary(trials: list[dict]) -> list[dict]:
    streams = [(trial["concurrency"], stream) for trial in trials for stream in trial["streams"]]
    output = []
    for concurrency, phase in sorted({(concurrency, stream["phase"]) for concurrency, stream in streams}):
        group = [stream for level, stream in streams if level == concurrency and stream["phase"] == phase]
        good = [stream for stream in group if stream["ok"]]
        output.append({
            "concurrency": concurrency, "phase": phase,
            "successful_requests": len(good), "failed_requests": len(group) - len(good),
            **{key: distribution(stream[key] for stream in good) for key in ("ttft_s", "content_ttft_s", "elapsed_s", "tpot_proxy_s", "cached_prompt_tokens", "computed_prompt_tokens")},
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.set_defaults(concurrency="1")
    parser.add_argument("--mode", choices=("coding-8k", "multi-turn", "tool-smoke"), default="coding-8k")
    parser.add_argument("--target-tokens", type=int, default=8192)
    parser.add_argument("--prepare-timeout", type=float, default=300)
    parser.add_argument("--replay-report", help="Replay exact saved cases; require unchanged request hashes and tokenizer output")
    args = parser.parse_args()
    levels = validate_arguments(parser, args)
    if args.target_tokens <= 0 or not 0 < args.prepare_timeout < float("inf"):
        parser.error("target-tokens and finite prepare-timeout must be positive")
    if not args.output:
        parser.error("--output is required to retain prompts, evidence, and partial failures")
    report = new_report(args, "synthetic-agent-proxy")
    report["manifest"]["agent_client_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report["manifest"]["quality_assessment"] = "No coding quality score; text presence and tool schema semantics only. Multi-turn uses fixed history, not generated-history interaction."
    fixture = json.loads(FIXTURE_PATH.read_text())
    report["manifest"]["fixture"] = {"version": fixture["version"], "sha256": json_hash(fixture)}
    write_report(args.output, report)
    try:
        preparation_deadline = time.perf_counter() + args.prepare_timeout

        def tokenize(body):
            remaining = preparation_deadline - time.perf_counter()
            if remaining <= 0:
                raise TimeoutError("tokenizer preparation deadline exceeded")
            return request_json(args.base_url, "/tokenize", body, min(30, remaining))

        for endpoint in ("/version", "/v1/models"):
            try:
                report["manifest"].setdefault("server", {})[endpoint] = request_json(args.base_url, endpoint, timeout=min(10, args.prepare_timeout))
            except Exception as exc:
                report["manifest"].setdefault("metadata_errors", {})[endpoint] = str(exc)
        if args.replay_report:
            cases = json.loads(Path(args.replay_report).read_text())["manifest"]["cases"]
            if not cases:
                raise ValueError("replay report contains no request cases")
            for case in cases:
                if case["body"]["model"] != args.model or case["body"]["max_tokens"] != args.max_tokens:
                    raise ValueError("replay model/max-tokens differ from the requested settings")
                if json_hash(case["body"]) != case["request_sha256"]:
                    raise ValueError("replay request hash mismatch")
                if token_evidence(case["body"], tokenize) != case["tokenization"]:
                    raise ValueError("tokenizer output changed for an exact replay request")
            report["manifest"]["replay_source"] = args.replay_report
        else:
            cases = prepare_cases(args, levels, fixture, tokenize)
        report["manifest"]["cases"] = cases
        report["manifest"]["request_count"] = len(cases)
        report["manifest"]["cases_sha256"] = json_hash(cases)
        write_report(args.output, report)
        failed = False
        for concurrency, trial_number in sorted({(case["concurrency"], case["trial"]) for case in cases}):
            selected = [case for case in cases if (case["concurrency"], case["trial"]) == (concurrency, trial_number)]
            trial = {"concurrency": concurrency, "trial": trial_number, "streams": [], "complete": False}
            report["trials"].append(trial)
            lock = threading.Lock()

            def save_result(record):
                with lock:
                    trial["streams"].append(record)
                    write_report(args.output, report)

            started = time.perf_counter()
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                pending = [executor.submit(run_conversation, [case for case in selected if case["conversation"] == conversation], args, save_result) for conversation in range(concurrency)]
                for future in as_completed(pending):
                    failed = not future.result() or failed
            trial["streams"].sort(key=lambda stream: (stream["conversation"], stream["turn"]))
            finish_trial(trial, time.perf_counter() - started)
            report["summary"] = summarize_trials(report["trials"])
            write_report(args.output, report)
            print(f"mode={args.mode} concurrency={concurrency} trial={trial_number} aggregate={trial['aggregate_token_tps']:.2f} tok/s errors={trial['errors']}", flush=True)
            if failed:
                break
        report["status"] = "failed" if failed else "complete"
    except BaseException as exc:
        report["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["summary"] = summarize_trials(report["trials"])
        report["phase_summary"] = phase_summary(report["trials"])
        write_report(args.output, report)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
