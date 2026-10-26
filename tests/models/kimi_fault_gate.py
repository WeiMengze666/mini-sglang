"""Harness-only silent-state corruption checks for the tiny Kimi runtime gate.

The production cache/model does not automatically scan for or recover from
non-finite state. This gate supplies its own detector and explicitly releases
or resets state after detection. It makes no serving fault-tolerance claim.
Only owned, completed requests and their initialized live MLA pages are read;
unused MLA storage is deliberately poisoned to expose accidental whole-pool scans.
"""

from __future__ import annotations

import torch


class NonFiniteStateError(AssertionError):
    """A known harness detection, distinct from an unrelated execution failure."""

    def __init__(self, component: str, slot: int, uid: int):
        self.component, self.slot, self.uid = component, slot, uid
        super().__init__(f"Non-finite {component} state for slot={slot}, uid={uid}")


def _request_state(cache, table, slot, uid):
    """Return only the state belonging to one live, completed request."""
    if cache.recurrent._owners[slot] != uid or cache.recurrent._active[slot]:
        raise AssertionError("Fault gate requires the expected completed request owner")
    length = cache.recurrent._processed_lens[slot]
    if not 0 < length <= table.shape[1]:
        raise AssertionError("Fault gate requires a valid initialized live request prefix")
    locations = table[slot, :length].to(device=cache.device, dtype=torch.long)
    return {
        "recurrent": cache.recurrent._recurrent[:, slot],
        "convolution": cache.recurrent._conv[:, slot],
        "mla": cache._mla.index_select(1, locations),
    }


def _assert_finite_request(cache, table, slot, uid):
    for component, state in _request_state(cache, table, slot, uid).items():
        if not bool(torch.isfinite(state).all()):
            raise NonFiniteStateError(component, slot, uid)


def assert_finite_live_state(cache, table):
    """Harness detector: never scan unowned slots or unwritten MLA capacity."""
    for slot, uid in enumerate(cache.recurrent._owners):
        if uid is not None:
            _assert_finite_request(cache, table, slot, uid)


def _snapshot(cache, table, slot, uid):
    return {name: value.clone() for name, value in _request_state(cache, table, slot, uid).items()}


def _close(actual, expected, dtype, metrics, name, *, exact=False):
    if not bool(torch.isfinite(actual).all()) or not bool(torch.isfinite(expected).all()):
        raise AssertionError(f"{name}: comparison contains non-finite data")
    rtol, atol = (0.04, 0.03) if dtype == torch.bfloat16 else (3e-5, 3e-6)
    if exact:
        rtol, atol = 0.0, 0.0
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol, msg=name)
    metrics[name] = {
        "max_abs_error": float((actual.float() - expected.float()).abs().max())
        if actual.numel()
        else 0.0,
        "rtol": rtol,
        "atol": atol,
    }


def _compare_state(cache, table, slot, uid, expected, dtype, metrics, prefix, *, exact=False):
    for component, state in _request_state(cache, table, slot, uid).items():
        _close(state, expected[component], dtype, metrics, f"{prefix}_{component}", exact=exact)


def _expect_recurrent_fault(cache, table, slot, uid):
    # Catch ONLY the detector's exception. Model errors, bad indexing, and other
    # assertions must fail the gate, rather than masquerade as successful detection.
    try:
        assert_finite_live_state(cache, table)
    except NonFiniteStateError as error:
        if (error.component, error.slot, error.uid) != ("recurrent", slot, uid):
            raise AssertionError("Detector identified an unexpected corrupt component") from error
        return str(error)
    raise AssertionError("Silent recurrent corruption was not detected by the harness")


def _assert_cleared(cache, slots):
    for slot in slots:
        if (
            cache.recurrent._owners[slot] is not None
            or cache.recurrent._processed_lens[slot] != 0
            or cache.recurrent._active[slot]
        ):
            raise AssertionError("Explicit cleanup did not clear request ownership/liveness")
        for tensor in (cache.recurrent._recurrent[:, slot], cache.recurrent._conv[:, slot]):
            if not bool(torch.isfinite(tensor).all()) or bool(torch.count_nonzero(tensor)):
                raise AssertionError("Explicit cleanup did not zero request-local state")


@torch.inference_mode()
def run_nonfinite_fault_gate(model, config, dtype, device) -> dict:
    """Inject NaN and +Inf after valid logits; verify detection/isolation/cleanup.

    Each fault has its own cache and independent clean request baselines. Decode
    continues only for the unaffected request until the contaminated request is
    explicitly released. No non-finite production recovery is implied.
    """
    # The runtime CLI imports this helper; defer the reverse import until called.
    from run_kimi_runtime_gate import PROMPT, batch_for, cache_for, page_table

    table = page_table(device)
    reports = {}
    for fault, value in (("nan", float("nan")), ("inf", float("inf"))):
        cache = cache_for(config, dtype, device)
        baseline_a = cache_for(config, dtype, device)
        baseline_b = cache_for(config, dtype, device)
        if not cache.recurrent._recurrent.numel():
            raise ValueError("Silent recurrent fault gate requires at least one KDA layer")
        for pool in (cache, baseline_a, baseline_b):
            # MLA is normally torch.empty. Sentinel values make it deterministic
            # that scanning the unused suffix or unowned third slot is incorrect.
            pool._mla.fill_(float("nan"))
        metrics = {}
        uid_a, uid_b, uid_successor = 610, 611, 612
        prompt_b = [3, 15, 7]

        def forward(pool, tokens, slot, uid, start=0):
            return model.forward_batch(
                batch_for(tokens, device, slot=slot, uid=uid, start=start),
                pool,
                table,
                all_logits=True,
            )

        logits_a = forward(cache, PROMPT, 0, uid_a)
        expected_a = forward(baseline_a, PROMPT, 0, uid_a)
        logits_b = forward(cache, prompt_b, 1, uid_b)
        expected_b = forward(baseline_b, prompt_b, 1, uid_b)
        _close(logits_a, expected_a, dtype, metrics, "valid_prefill_a")
        _close(logits_b, expected_b, dtype, metrics, "valid_prefill_b")
        _compare_state(
            cache,
            table,
            0,
            uid_a,
            _snapshot(baseline_a, table, 0, uid_a),
            dtype,
            metrics,
            "valid_prefill_a",
        )
        _compare_state(
            cache,
            table,
            1,
            uid_b,
            _snapshot(baseline_b, table, 1, uid_b),
            dtype,
            metrics,
            "valid_prefill_b",
        )
        assert_finite_live_state(cache, table)
        prior_logits = logits_a.clone()
        unrelated_state = _snapshot(cache, table, 1, uid_b)
        owners = (
            list(cache.recurrent._owners),
            list(cache.recurrent._processed_lens),
            list(cache.recurrent._active),
        )

        # Silent write occurs AFTER a successful forward returned finite logits.
        cache.recurrent._recurrent[0, 0, 0, 0, 0] = value
        if int((~torch.isfinite(cache.recurrent._recurrent[:, 0])).sum()) != 1:
            raise AssertionError("Fault injection must corrupt exactly one recurrent scalar")
        detection = _expect_recurrent_fault(cache, table, 0, uid_a)
        _close(logits_a, prior_logits, dtype, metrics, "finite_prior_logits", exact=True)
        if owners != (
            cache.recurrent._owners,
            cache.recurrent._processed_lens,
            cache.recurrent._active,
        ):
            raise AssertionError("Injection/detection changed request ownership or liveness")
        _compare_state(
            cache,
            table,
            1,
            uid_b,
            unrelated_state,
            dtype,
            metrics,
            "unrelated_after_injection",
            exact=True,
        )
        _assert_finite_request(cache, table, 1, uid_b)

        for step, token in enumerate((6, 2, 9)):
            start = len(prompt_b) + step
            actual = forward(cache, [token], 1, uid_b, start)
            expected = forward(baseline_b, [token], 1, uid_b, start)
            _close(actual, expected, dtype, metrics, f"unrelated_decode_{step}_logits")
            _compare_state(
                cache,
                table,
                1,
                uid_b,
                _snapshot(baseline_b, table, 1, uid_b),
                dtype,
                metrics,
                f"unrelated_decode_{step}",
            )
            _assert_finite_request(cache, table, 1, uid_b)
        # The harness detector and unrelated decode must not silently heal A.
        _expect_recurrent_fault(cache, table, 0, uid_a)
        unrelated_state = _snapshot(cache, table, 1, uid_b)
        unrelated_metadata = (
            cache.recurrent._owners[1],
            cache.recurrent._processed_lens[1],
            cache.recurrent._active[1],
        )
        cache.release_request(0, uid_a)
        _assert_cleared(cache, [0])
        if unrelated_metadata != (
            cache.recurrent._owners[1],
            cache.recurrent._processed_lens[1],
            cache.recurrent._active[1],
        ):
            raise AssertionError("Releasing the contaminated slot changed unrelated metadata")
        _compare_state(
            cache,
            table,
            1,
            uid_b,
            unrelated_state,
            dtype,
            metrics,
            "unrelated_after_release",
            exact=True,
        )
        assert_finite_live_state(cache, table)

        # A different, shorter prompt exposes stale-prefix/slot leakage on reuse.
        successor_prompt = list(reversed(PROMPT[:4]))
        baseline_a.reset()
        actual = forward(cache, successor_prompt, 0, uid_successor)
        expected = forward(baseline_a, successor_prompt, 0, uid_successor)
        _close(actual, expected, dtype, metrics, "released_slot_reuse_logits")
        _compare_state(
            cache,
            table,
            0,
            uid_successor,
            _snapshot(baseline_a, table, 0, uid_successor),
            dtype,
            metrics,
            "released_slot_reuse",
        )
        actual = forward(cache, [13], 0, uid_successor, len(successor_prompt))
        expected = forward(baseline_a, [13], 0, uid_successor, len(successor_prompt))
        _close(actual, expected, dtype, metrics, "reused_slot_decode_logits")
        _compare_state(
            cache,
            table,
            0,
            uid_successor,
            _snapshot(baseline_a, table, 0, uid_successor),
            dtype,
            metrics,
            "reused_slot_decode",
        )
        _compare_state(
            cache,
            table,
            1,
            uid_b,
            unrelated_state,
            dtype,
            metrics,
            "unrelated_after_reuse",
            exact=True,
        )
        assert_finite_live_state(cache, table)

        # Re-contaminate before reset so this checks cleanup of bad state, not
        # merely repeatability after an already successful per-request release.
        cache.recurrent._recurrent[0, 0, 0, 0, 0] = value
        _expect_recurrent_fault(cache, table, 0, uid_successor)
        cache.reset()
        _assert_cleared(cache, range(cache.max_running_req))
        assert_finite_live_state(cache, table)  # No live MLA pages after reset.
        baseline_a.reset()
        baseline_b.reset()
        for slot, uid, prompt, baseline in (
            (0, uid_a, PROMPT, baseline_a),
            (1, uid_b, prompt_b, baseline_b),
        ):
            actual = forward(cache, prompt, slot, uid)
            expected = forward(baseline, prompt, slot, uid)
            _close(actual, expected, dtype, metrics, f"reset_repeat_{slot}_logits")
            _compare_state(
                cache,
                table,
                slot,
                uid,
                _snapshot(baseline, table, slot, uid),
                dtype,
                metrics,
                f"reset_repeat_{slot}",
            )
        assert_finite_live_state(cache, table)
        reports[fault] = {"status": "passed", "detection": detection, "checks": metrics}

    return {
        "status": "passed",
        "scope": "harness_detection_and_explicit_release_reset_only",
        "production_automatic_nonfinite_detection": False,
        "production_automatic_nonfinite_recovery": False,
        "mla_scan": "owned_initialized_live_pages_only",
        "unused_mla_poisoned": True,
        "faults": reports,
    }
