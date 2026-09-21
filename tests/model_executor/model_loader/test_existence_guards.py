# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""No post-load pass returns early on "have I ever built this" unreviewed.

At cold start that question is the same as "is this current", so the guard is
right. After a weight update it is not: the pass returns, and the model goes on
serving state derived from the previous checkpoint, with no error anywhere. A
pass that can be re-run inverts its guard on `rebuilding_derived_state()`; one
that cannot is listed here with the reason, which is the audit record.
"""

import ast
import pathlib

# Post-load entry points. A guard anywhere else runs per forward, not per load.
POST_LOAD_NAMES = (
    "process_weights_after_loading",
    "finalize_weights",
    "finalize_loaded_weights",
)

# Guards that cannot be inverted, and why. Removing an entry means the pass
# became re-runnable; adding one means a model cannot take a weight update, and
# `check_post_load_is_reload_safe` has to keep refusing it.
KNOWN_NOT_RERUNNABLE = {
    (
        "vllm/models/deepseek_v4/nvidia/fi_moe.py",
        "finalize_weights",
    ): (
        "the fused weights are handed to flashinfer's MoEEpMegaLayer, a "
        "third-party object with no in-place update, so re-running allocates "
        "storage a captured graph does not read"
    ),
    (
        "vllm/models/deepseek_v4/xpu/model.py",
        "finalize_weights",
    ): (
        "drops w13_weight once transformed, so a rebuild has no input; needs "
        "the treatment KimiK3MegaMoEExperts got, which no XPU here can verify"
    ),
    (
        "vllm/models/dots3_note/nvidia/vision.py",
        "process_weights_after_loading",
    ): (
        "deletes self.experts once fused, so a reload has nowhere to put the "
        "expert weights and they are reported as orphaned instead"
    ),
}

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]


def _existence_guard(node: ast.stmt, assigned_here: set[str]) -> str | None:
    """`if <self.x is not None>: return` or `if <self.flag>: return`.

    An attribute the function assigns first is a dispatch branch recomputed on
    every call, not a record of a previous one.
    """
    if not isinstance(node, ast.If) or node.orelse:
        return None
    body = node.body
    if len(body) != 1 or not isinstance(body[0], ast.Return) or body[0].value:
        return None

    test = node.test
    if (
        isinstance(test, ast.Compare)
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.IsNot)
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value is None
    ):
        target = test.left
    elif isinstance(test, ast.Attribute):
        target = test
    else:
        return None

    name = ast.unparse(target)
    if not name.startswith("self.") or name in assigned_here:
        return None
    return name


def _guards_in(fn: ast.FunctionDef) -> list[str]:
    assigned = {
        ast.unparse(t)
        for node in ast.walk(fn)
        if isinstance(node, ast.Assign)
        for t in node.targets
        if isinstance(t, ast.Attribute)
    }
    inverted = "rebuilding_derived_state" in ast.unparse(fn)
    if inverted:
        return []
    return [g for stmt in fn.body[:4] if (g := _existence_guard(stmt, assigned))]


def test_no_unreviewed_post_load_returns_early_on_a_previous_build():
    found = {}
    for path in sorted(REPO_ROOT.joinpath("vllm").rglob("*.py")):
        if "third_party" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text())
        except (SyntaxError, UnicodeDecodeError):
            continue
        relative = path.relative_to(REPO_ROOT).as_posix()
        for fn in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)):
            if fn.name not in POST_LOAD_NAMES:
                continue
            if (guards := _guards_in(fn)) and (relative, fn.name) not in (
                KNOWN_NOT_RERUNNABLE
            ):
                found[f"{relative}:{fn.lineno}:{fn.name}"] = guards

    assert not found, (
        "these post-load passes return early once they have run, so a weight "
        "update leaves the model on the previous checkpoint's derived state. "
        "Invert the guard on `rebuilding_derived_state()`, or add the pass to "
        f"KNOWN_NOT_RERUNNABLE with the reason it cannot be: {found}"
    )


def test_the_known_list_names_passes_that_still_exist():
    """A stale entry would hide a guard that came back under the same name."""
    missing = [
        f"{path}:{name}"
        for path, name in KNOWN_NOT_RERUNNABLE
        if not REPO_ROOT.joinpath(path).exists()
        or name not in REPO_ROOT.joinpath(path).read_text()
    ]
    assert not missing, f"KNOWN_NOT_RERUNNABLE names nothing in tree: {missing}"
