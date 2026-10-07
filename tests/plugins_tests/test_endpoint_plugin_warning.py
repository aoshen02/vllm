# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`VLLM_PLUGINS` is one allowlist for all plugin groups: when it names an
endpoint plugin, the plugins of the other groups it does not name are
listed in a warning (and nothing is logged without endpoint plugins)."""

import pytest


class _DummyEndpointPlugin:
    name = "wanted"
    required_tasks = None


@pytest.mark.parametrize(
    "excluded_group",
    [
        "vllm.general_plugins",
        "vllm.io_processor_plugins",
        "vllm.platform_plugins",
        "vllm.stat_logger_plugins",
    ],
)
@pytest.mark.parametrize("endpoint_plugin_loaded", [True, False])
def test_vllm_plugins_exclusion_warning(
    monkeypatch, endpoint_plugin_loaded, excluded_group
):
    """Claude r17: warn only when an endpoint plugin is actually loaded (so a
    plugin-absent server logs as on base), naming excluded other plugins."""
    import importlib.metadata

    from vllm import plugins

    general = [
        importlib.metadata.EntryPoint(
            name="other", value="json:dumps", group=excluded_group
        )
    ]
    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda group: general if group == excluded_group else [],
    )
    monkeypatch.setattr(
        plugins,
        "load_plugins_by_group",
        lambda group: {"wanted": _DummyEndpointPlugin}
        if endpoint_plugin_loaded
        else {},
    )
    monkeypatch.setenv("VLLM_PLUGINS", "wanted" if endpoint_plugin_loaded else "")
    warnings = []
    monkeypatch.setattr(
        plugins.logger, "warning", lambda msg, *args: warnings.append(msg % args)
    )
    loaded = plugins.load_endpoint_plugins(("generate",))
    assert len(loaded) == int(endpoint_plugin_loaded)
    assert any("other" in w for w in warnings) is endpoint_plugin_loaded
