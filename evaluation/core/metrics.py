"""Metrics Calculator for computing TSR, PRR, and RC.

Metric definitions:
- TSR (Task Success Rate): pass rate for each mode
- PRR (Perturbation Recovery Rate): hit-conditioned resolution rate
- RC (Recovery Cost): exposure-conditioned normalized recovery burden
"""

from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
from datetime import datetime


class MetricsCalculator:
    """Calculate evaluation metrics from judgement results."""

    def __init__(self):
        """Initialize metrics calculator."""
        self.results: List[Dict[str, Any]] = []

    def add_result(
        self,
        task_json: Dict[str, Any],
        inference_data: Dict[str, Any],
        judgement: Dict[str, Any]
    ) -> None:
        """Add a single task result with trace-aware recovery annotations."""
        task_id = task_json["task_id"]
        mode = task_json.get("perturbation_mode", "P0")
        passed = judgement.get("pass", False)
        tokens_used = self._extract_total_tokens(inference_data)

        tool_events = self._extract_tool_events(inference_data)
        hit_info = self._extract_first_hit_info(tool_events)
        hit = hit_info is not None
        victim_tool = hit_info["victim_tool"] if hit else judgement.get("trace_check", {}).get("victim_tool")
        successful_prefix_tools = hit_info["successful_prefix_tools"] if hit else []
        actual_recovery_calls = hit_info["actual_recovery_calls"] if hit else None

        oracle_recovery_calls = None
        oracle_method = None
        rc = None
        rc_case = None
        prr_hits = 0
        prr_resolved = 0
        if mode != "P0":
            complexity = task_json.get("complexity", "C1")
            if not hit:
                # No fault exposure => no recovery episode occurred, so RC is
                # undefined for this run and is excluded from the RC average
                # (exposure-conditioned, mirroring PRR). Use `hit`/exposure rate
                # to report how often the model was actually tested on recovery.
                rc = None
                rc_case = "no_hit"
            else:
                oracle_recovery_calls, oracle_method = self._compute_oracle_recovery_calls(
                    task_json=task_json,
                    mode=mode,
                    victim_tool=victim_tool,
                    successful_prefix_tools=successful_prefix_tools,
                )
                rc = self._compute_run_rc(
                    passed=passed,
                    actual_recovery_calls=actual_recovery_calls,
                    oracle_recovery_calls=oracle_recovery_calls,
                )
                rc_case = "hit_pass" if passed else "hit_fail"

            all_hits = self._find_all_first_hits(tool_events)
            prr_hits = len(all_hits)
            for hit_idx, victim_tool_hit in all_hits:
                if self._check_hit_resolved(task_json, tool_events, hit_idx, victim_tool_hit, mode, complexity):
                    prr_resolved += 1

        self.results.append({
            "task_id": task_id,
            "mode": mode,
            "passed": passed,
            "tokens": tokens_used,
            "hit": hit,
            "victim_tool": victim_tool,
            "actual_recovery_calls": actual_recovery_calls,
            "oracle_recovery_calls": oracle_recovery_calls,
            "oracle_method": oracle_method,
            "rc": rc,
            "rc_case": rc_case,
            "prr_hits": prr_hits,
            "prr_resolved": prr_resolved,
        })

    def compute_tsr(self, mode: str) -> float:
        """Compute Task Success Rate for a mode.

        TSR_mode = Count(Pass runs in mode) / Total runs in mode

        Args:
            mode: Perturbation mode

        Returns:
            Success rate (0.0 to 1.0)
        """
        mode_results = [r for r in self.results if r["mode"] == mode]

        if not mode_results:
            return 0.0

        passed_count = sum(1 for r in mode_results if r["passed"])
        return passed_count / len(mode_results)

    def compute_prr(self, p_mode: str) -> Optional[float]:
        """Compute hit-conditioned Perturbation Recovery Rate.

        PRR_mode = #ResolvedHits / #TotalHits

        A hit is the first occurrence of each unique perturbed tool in a run.
        A hit is resolved when the agent correctly handles the perturbation:
        retry for transient (P1/P3), path-switch for permanent (P2/P4),
        abort for C1 permanent.
        """
        mode_results = [r for r in self.results if r["mode"] == p_mode]
        total_hits = sum(r.get("prr_hits", 0) for r in mode_results)
        resolved = sum(r.get("prr_resolved", 0) for r in mode_results)
        if total_hits == 0:
            return None
        return resolved / total_hits

    def compute_rc(self, p_mode: str) -> Optional[float]:
        """Compute exposure-conditioned normalized Recovery Cost.

        Only runs that actually hit a fault are counted (no-hit runs are
        excluded from the denominator, mirroring PRR):
        - no hit: excluded (rc is None)
        - hit and fail: rc = 1
        - hit and pass: rc = 1 - C_oracle / max(C_act, C_oracle)

        RC_mode = mean(rc_run | hit runs in mode)
        """
        samples = [
            r["rc"] for r in self.results
            if r["mode"] == p_mode and r["hit"] and r["rc"] is not None
        ]
        if not samples:
            return None
        return sum(samples) / len(samples)

    def generate_report(self) -> Dict[str, Any]:
        """Generate complete metrics report.

        Returns:
            Dictionary containing all metrics
        """
        # Compute TSR for all modes
        modes = ["P0", "P1", "P2", "P3", "P4"]
        tsr = {mode: self.compute_tsr(mode) for mode in modes}

        # Compute PRR for P1-P4
        prr = {mode: self.compute_prr(mode) for mode in ["P1", "P2", "P3", "P4"]}

        # Compute RC for P1-P4
        rc = {mode: self.compute_rc(mode) for mode in ["P1", "P2", "P3", "P4"]}

        # Per-mode breakdown
        mode_breakdown = self._compute_mode_breakdown()
        recovery_breakdown = self._compute_recovery_breakdown()
        insights = self._generate_insights(tsr, prr, rc, recovery_breakdown)

        return {
            "timestamp": datetime.now().isoformat(),
            "total_tasks": len(self.results),
            "tsr": tsr,
            "prr": prr,
            "rc": rc,
            "insights": insights,
            "mode_breakdown": mode_breakdown,
            "recovery_breakdown": recovery_breakdown,
        }

    def _generate_insights(
        self,
        tsr: Dict[str, float],
        prr: Dict[str, Optional[float]],
        rc: Dict[str, Optional[float]],
        recovery_breakdown: Dict[str, Dict[str, Any]],
    ) -> Dict[str, str]:
        """Generate natural language insights.
        """
        insights = {}

        # The Trust Issue: P3 vs P1
        if prr.get("P1") is not None and prr.get("P3") is not None:
            p1_prr = prr["P1"]
            p3_prr = prr["P3"]
            if p1_prr > 0.7 and p3_prr < 0.5:
                gap = (p1_prr - p3_prr) * 100
                insights["trust_issue"] = (
                    f"P3 PRR ({p3_prr:.2%}) is {gap:.1f}% lower than P1 ({p1_prr:.2%}), "
                    f"revealing over-trust in tool outputs (Sanity Ignorance)"
                )

        # The Validation Gap: P4
        if prr.get("P4") is not None:
            p4_prr = prr["P4"]
            if p4_prr < 0.5:
                insights["validation_gap"] = (
                    f"P4 PRR ({p4_prr:.2%}) shows weak data validation capabilities after fault exposure"
                )
            elif p4_prr > 0.7:
                insights["validation_success"] = (
                    f"P4 PRR ({p4_prr:.2%}) demonstrates strong data validation after fault exposure"
                )

        # Recovery efficiency
        if rc.get("P1") is not None and rc.get("P3") is not None:
            if rc["P3"] > rc["P1"] + 0.2:
                insights["recovery_cost"] = (
                    f"P3 recovery cost ({rc['P3']:.2f}) is materially higher than P1 ({rc['P1']:.2f}), "
                    f"suggesting more failures or extra detours after semantic corruption"
                )

        # Fault exposure coverage
        for mode in ("P1", "P2", "P3", "P4"):
            stats = recovery_breakdown.get(mode, {})
            hit_count = stats.get("hit_count", 0)
            total = stats.get("total", 0)
            if total > 0 and hit_count == 0:
                insights[f"{mode.lower()}_no_hits"] = (
                    f"{mode} recorded no fault hits, so PRR is unavailable and RC reflects zero observed recovery cost"
                )

        return insights

    def _compute_mode_breakdown(self) -> Dict[str, Dict[str, Any]]:
        """Compute detailed breakdown for each mode.

        Returns:
            Per-mode statistics
        """
        breakdown = defaultdict(lambda: {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "avg_tokens": 0,
            "min_tokens": float('inf'),
            "max_tokens": 0
        })

        for result in self.results:
            mode = result["mode"]
            breakdown[mode]["total"] += 1
            if result["passed"]:
                breakdown[mode]["passed"] += 1
            else:
                breakdown[mode]["failed"] += 1

            tokens = result["tokens"]
            breakdown[mode]["min_tokens"] = min(breakdown[mode]["min_tokens"], tokens)
            breakdown[mode]["max_tokens"] = max(breakdown[mode]["max_tokens"], tokens)

        # Compute averages
        for mode, stats in breakdown.items():
            mode_results = [r for r in self.results if r["mode"] == mode]
            if mode_results:
                stats["avg_tokens"] = sum(r["tokens"] for r in mode_results) / len(mode_results)
            if stats["min_tokens"] == float('inf'):
                stats["min_tokens"] = 0

        return dict(breakdown)

    def _compute_recovery_breakdown(self) -> Dict[str, Dict[str, Any]]:
        """Compute recovery statistics for P1-P4."""
        breakdown = {}
        for mode in ["P1", "P2", "P3", "P4"]:
            mode_results = [r for r in self.results if r["mode"] == mode]
            hit_results = [r for r in mode_results if r["hit"]]
            hit_pass_results = [r for r in hit_results if r["passed"]]
            hit_fail_results = [r for r in hit_results if not r["passed"]]
            rc_samples = [r["rc"] for r in mode_results if r["rc"] is not None]
            actual_calls = [
                r["actual_recovery_calls"] for r in hit_results
                if r["actual_recovery_calls"] is not None
            ]
            oracle_calls = [
                r["oracle_recovery_calls"] for r in hit_results
                if r["oracle_recovery_calls"] is not None
            ]
            breakdown[mode] = {
                "total": len(mode_results),
                "hit_count": len(hit_results),
                "no_hit_count": len(mode_results) - len(hit_results),
                "hit_pass_count": len(hit_pass_results),
                "hit_fail_count": len(hit_fail_results),
                "prr_denominator": sum(r.get("prr_hits", 0) for r in mode_results),
                "prr_numerator": sum(r.get("prr_resolved", 0) for r in mode_results),
                "rc_sample_count": len(rc_samples),
                "avg_actual_recovery_calls": (
                    sum(actual_calls) / len(actual_calls) if actual_calls else None
                ),
                "avg_oracle_recovery_calls": (
                    sum(oracle_calls) / len(oracle_calls) if oracle_calls else None
                ),
            }
        return breakdown

    def _compute_run_rc(
        self,
        passed: bool,
        actual_recovery_calls: Optional[int],
        oracle_recovery_calls: Optional[int],
    ) -> Optional[float]:
        """Compute per-run normalized Recovery Cost.

        Semantics:
        - no-hit runs are handled earlier and assigned rc=None (excluded from RC)
        - hit-and-fail runs take the maximum cost rc=1
        - hit-and-pass runs pay only for detours beyond the oracle path
        """
        if not passed:
            return 1.0

        if not actual_recovery_calls or not oracle_recovery_calls:
            return None

        denominator = max(actual_recovery_calls, oracle_recovery_calls)
        if denominator <= 0:
            return None

        raw_rc = 1.0 - (oracle_recovery_calls / denominator)
        return max(0.0, min(1.0, raw_rc))

    def _extract_total_tokens(self, inference_data: Dict[str, Any]) -> int:
        """Extract total tokens from saved inference payload."""
        tokens = inference_data.get("tokens", {})
        if isinstance(tokens, dict):
            for key in ("total_tokens", "total"):
                value = tokens.get(key)
                if isinstance(value, int):
                    return value
        legacy = inference_data.get("token_usage", {})
        if isinstance(legacy, dict):
            value = legacy.get("total_tokens")
            if isinstance(value, int):
                return value
        return 0

    def _extract_tool_events(self, inference_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract ordered tool-call events from trace messages."""
        messages = inference_data.get("messages", [])
        pending_calls: Dict[str, Dict[str, Any]] = {}
        events: List[Dict[str, Any]] = []

        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("type") == "tool_call":
                call = msg.get("tool_call", {})
                call_id = call.get("id")
                if call_id:
                    pending_calls[call_id] = {
                        "tool_name": call.get("name"),
                        "arguments": call.get("arguments", {}),
                    }
            elif msg.get("role") == "tool":
                call_id = msg.get("call_id")
                call = pending_calls.pop(call_id, {})
                tool_name = msg.get("name") or call.get("tool_name")
                output = msg.get("content", {})
                has_error = isinstance(output, dict) and "error" in output
                events.append({
                    "tool_name": tool_name,
                    "success": not has_error,
                    "perturbed": msg.get("metadata", {}).get("perturbation_status") == "perturbed",
                })

        return events

    def _extract_first_hit_info(
        self,
        tool_events: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Locate the first actual perturbation hit in a run."""
        successful_prefix_tools: List[str] = []

        for idx, event in enumerate(tool_events):
            if event["perturbed"]:
                return {
                    "victim_tool": event["tool_name"],
                    "actual_recovery_calls": len(tool_events) - idx,
                    "successful_prefix_tools": list(successful_prefix_tools),
                }
            if event["success"] and event["tool_name"]:
                successful_prefix_tools.append(event["tool_name"])

        return None

    def _compute_oracle_recovery_calls(
        self,
        task_json: Dict[str, Any],
        mode: str,
        victim_tool: Optional[str],
        successful_prefix_tools: List[str],
    ) -> Tuple[Optional[int], Optional[str]]:
        """Compute task-derived canonical recovery suffix cost."""
        complexity = task_json.get("complexity", "C1")

        if complexity in ("C2", "C3", "C4"):
            return self._compute_multi_path_oracle_calls(
                task_json=task_json,
                mode=mode,
                complexity=complexity,
                victim_tool=victim_tool,
                successful_prefix_tools=successful_prefix_tools,
            )

        return self._compute_single_path_oracle_calls(task_json, mode)

    def _compute_single_path_oracle_calls(
        self,
        task_json: Dict[str, Any],
        mode: str,
    ) -> Tuple[Optional[int], Optional[str]]:
        """Compute canonical recovery suffix cost for single-path tasks."""
        execution_trace = task_json.get("execution_trace", [])
        if not execution_trace:
            return None, None

        if mode in ("P2", "P4"):
            # Legacy single-path permanent modes require immediate graceful stop.
            return 1, "graceful_stop_single_path"

        perturbed_index = self._find_first_perturbed_index(execution_trace)
        if perturbed_index is None:
            return None, None

        return len(execution_trace) - perturbed_index, "single_path_suffix"

    def _compute_multi_path_oracle_calls(
        self,
        task_json: Dict[str, Any],
        mode: str,
        complexity: str,
        victim_tool: Optional[str],
        successful_prefix_tools: List[str],
    ) -> Tuple[Optional[int], Optional[str]]:
        """Compute canonical recovery suffix cost for multi-path tasks."""
        valid_paths = task_json.get("valid_paths", [])
        if not valid_paths:
            return None, None

        source_path_ids = {
            vp.get("path_id")
            for vp in valid_paths
            if victim_tool and victim_tool in vp.get("tools", [])
        }

        candidates: List[Tuple[int, str]] = []
        for vp in valid_paths:
            path_id = vp.get("path_id")
            exec_trace = vp.get("execution_trace", [])
            if not exec_trace:
                continue

            if mode in ("P2", "P4") and path_id in source_path_ids:
                continue

            # C2/C3 only perturb the first-touched path. Switched paths stay clean.
            if complexity in ("C2", "C3") and path_id not in source_path_ids:
                candidate_trace = [{"tool_name": tool_name} for tool_name in vp.get("tools", [])]
                method = "clean_switch_path"
            else:
                candidate_trace = exec_trace
                method = "canonical_path_suffix"

            satisfied_calls = self._count_satisfied_candidate_calls(
                candidate_trace=candidate_trace,
                successful_prefix_tools=successful_prefix_tools,
            )
            # If the candidate path does not contain the actual victim tool, the
            # already-paid fault hit is outside that path but still part of the
            # recovery suffix by definition.
            hit_call_cost = 0 if path_id in source_path_ids else 1
            candidate_cost = len(candidate_trace) - satisfied_calls + hit_call_cost
            if candidate_cost > 0:
                candidates.append((candidate_cost, method))

        if candidates:
            return min(candidates, key=lambda item: item[0])

        if mode in ("P2", "P4"):
            # Fallback for legacy permanent tasks without a recoverable alternate path.
            return 1, "graceful_stop_fallback"

        return None, None

    def _count_satisfied_candidate_calls(
        self,
        candidate_trace: List[Dict[str, Any]],
        successful_prefix_tools: List[str],
    ) -> int:
        """Count candidate calls already satisfied before the fault hit.

        The judge for multi-path tasks is intentionally order-tolerant, so a
        model may complete some downstream tools before it first hits the
        perturbed tool. We therefore subtract all already successful non-
        perturbed candidate calls, not just strict path prefixes.
        """
        remaining_successes = defaultdict(int)
        for tool_name in successful_prefix_tools:
            remaining_successes[tool_name] += 1

        satisfied = 0
        for step in candidate_trace:
            if step.get("is_perturbed", False):
                continue
            tool_name = step.get("tool_name")
            if tool_name and remaining_successes[tool_name] > 0:
                remaining_successes[tool_name] -= 1
                satisfied += 1

        return satisfied

    def _find_first_perturbed_index(self, execution_trace: List[Dict[str, Any]]) -> Optional[int]:
        """Find the first is_perturbed step in a canonical trace."""
        for idx, step in enumerate(execution_trace):
            if step.get("is_perturbed", False):
                return idx
        return None

    # ── PRR helpers ───────────────────────────────────────────────────────────

    def _find_all_first_hits(self, tool_events: List[Dict[str, Any]]) -> List[Tuple[int, str]]:
        """Return the first occurrence index of each unique perturbed tool."""
        seen: set = set()
        hits: List[Tuple[int, str]] = []
        for i, event in enumerate(tool_events):
            if event["perturbed"] and event["tool_name"] not in seen:
                seen.add(event["tool_name"])
                hits.append((i, event["tool_name"]))
        return hits

    def _check_c1_abort(
        self,
        tool_events: List[Dict[str, Any]],
        hit_idx: int,
        victim_tool: Optional[str] = None,
    ) -> bool:
        """C1 permanent: resolved iff the agent makes no successful call to any
        tool *other than the victim* after the hit.

        A retry of the victim itself is tolerated. This matters for P4
        (implicit-permanent): a victim retry returns corrupted-but-error-free
        data and would otherwise be counted as a "successful call", wrongly
        marking a "detect -> retry-to-confirm -> graceful abort" run as
        unresolved. Tolerating victim retries mirrors the TSR judge
        (`_check_p2_p4_stop_after_victim`), which allows victim retries after
        the hit and only forbids proceeding with *other* tools. Calling any
        other tool means the agent kept going on corrupted data, i.e. did not
        recover.
        """
        return not any(
            e["success"] and e["tool_name"] != victim_tool
            for e in tool_events[hit_idx + 1:]
        )

    def _check_retry_succeeded(
        self, tool_events: List[Dict[str, Any]], hit_idx: int, victim_tool: Optional[str]
    ) -> bool:
        """Transient: resolved iff the same tool later succeeds without perturbation."""
        for e in tool_events[hit_idx + 1:]:
            if e["tool_name"] == victim_tool and e["success"] and not e["perturbed"]:
                return True
        return False

    def _get_source_path_ids(self, task_json: Dict[str, Any], victim_tool: Optional[str]) -> set:
        """Return path IDs that contain a perturbed step for victim_tool."""
        source: set = set()
        for vp in task_json.get("valid_paths", []):
            for step in vp.get("execution_trace", []):
                if step.get("tool_name") == victim_tool and step.get("is_perturbed"):
                    source.add(vp["path_id"])
                    break
        return source

    def _check_path_switched(
        self,
        task_json: Dict[str, Any],
        tool_events: List[Dict[str, Any]],
        victim_tool: Optional[str],
    ) -> bool:
        """Permanent multi-path: resolved iff agent completed an alternate path."""
        source_ids = self._get_source_path_ids(task_json, victim_tool)
        successful_tools = {e["tool_name"] for e in tool_events if e["success"]}
        for vp in task_json.get("valid_paths", []):
            if vp["path_id"] in source_ids:
                continue
            required = set(vp.get("tools", []))
            if required and required.issubset(successful_tools):
                return True
        return False

    def _check_hit_resolved(
        self,
        task_json: Dict[str, Any],
        tool_events: List[Dict[str, Any]],
        hit_idx: int,
        victim_tool: Optional[str],
        mode: str,
        complexity: str,
    ) -> bool:
        """Check whether a single perturbation hit was correctly recovered from."""
        if complexity == "C1":
            if mode in ("P2", "P4"):
                return self._check_c1_abort(tool_events, hit_idx, victim_tool)
            return self._check_retry_succeeded(tool_events, hit_idx, victim_tool)
        if mode in ("P1", "P3"):
            return (
                self._check_retry_succeeded(tool_events, hit_idx, victim_tool)
                or self._check_path_switched(task_json, tool_events, victim_tool)
            )
        return self._check_path_switched(task_json, tool_events, victim_tool)

    def print_summary(self, report: Dict[str, Any]) -> str:
        """Generate human-readable summary.

        Args:
            report: Report from generate_report()

        Returns:
            Formatted summary string
        """
        lines = [
            "=" * 60,
            "EVALUATION METRICS REPORT",
            "=" * 60,
            f"Timestamp: {report['timestamp']}",
            f"Total Tasks: {report['total_tasks']}",
            "",
            "Task Success Rate (TSR):",
            "-" * 60
        ]

        for mode, rate in report["tsr"].items():
            lines.append(f"  {mode}: {rate:.2%}")

        lines.extend([
            "",
            "Perturbation Recovery Rate (PRR):",
            "-" * 60
        ])

        for mode, rate in report["prr"].items():
            if rate is None:
                lines.append(f"  {mode}: N/A")
            else:
                lines.append(f"  {mode}: {rate:.2%}")

        lines.extend([
            "",
            "Recovery Cost (RC - All-Sample Recovery Burden):",
            "-" * 60
        ])

        for mode, cost in report["rc"].items():
            if cost is None:
                lines.append(f"  {mode}: N/A")
            else:
                lines.append(f"  {mode}: {cost:.2f}")

        lines.extend([
            "",
            "Recovery Breakdown:",
            "-" * 60
        ])
        for mode, stats in report["recovery_breakdown"].items():
            lines.append(
                f"  {mode}: hits={stats['hit_count']} "
                f"| prr={stats['prr_numerator']}/{stats['prr_denominator']} "
                f"| hit+pass={stats['hit_pass_count']} "
                f"| hit+fail={stats['hit_fail_count']} "
                f"| rc_samples={stats['rc_sample_count']}"
            )

        if report["insights"]:
            lines.extend([
                "",
                "Key Insights:",
                "-" * 60
            ])
            for key, insight in report["insights"].items():
                lines.append(f"  • {insight}")

        lines.append("=" * 60)

        return "\n".join(lines)
