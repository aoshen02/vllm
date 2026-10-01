"""Static performance inventory; not a numerical or throughput acceptance gate."""

import hashlib

from humming_stage4_candidate import validate_inventory


def validate_bench_inventory(record, scripts):
    validate_inventory(record, expected_moe_layers=23)
    if record.get("parameter_identity_unchanged") is not True:
        raise ValueError("Performance installation replaced parameters")
    if record.get("host_selection_observer_installed") is not False:
        raise ValueError("Performance worker must exclude selection observers")
    for row in record["layers"]:
        if row.get("observed_selections") != []:
            raise ValueError("Diagnostic observations in performance inventory")
    for field, filename in {
        "adapter_sha256": "humming_stage4_candidate.py",
        "transform_sha256": "humming_stage4_contract.py",
        "worker_sha256": "humming_stage4_bench_worker.py",
    }.items():
        expected = hashlib.sha256((scripts / filename).read_bytes()).hexdigest()
        if record.get(field) != expected:
            raise ValueError(f"Performance source mismatch: {filename}")
    return record
