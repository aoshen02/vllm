# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Encoder outputs of `aux_namespace` requests delivered as aux objects."""

import hashlib
import threading
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from safetensors.numpy import save

from tests.distributed.aux_output_connector.utils import fake_store, wait_for_writer
from tests.v1.core.test_aux_objects import (
    _BLOCK,
    _NAMESPACE,
    _PREFIX,
    _abort,
    _Engine,
)
from tests.v1.core.utils import EOS_TOKEN_ID
from vllm.config import AuxOutputConfig
from vllm.distributed.aux_output_connector.connector import (
    AuxOutputConnectorMetadata,
    AuxOutputSchedulerConnector,
    retained_objects,
)
from vllm.distributed.aux_output_connector.object_store import (
    AuxObjectWriter,
    aux_namespace_cache_salt,
    is_mm_embeds_key,
    mm_embeds_fields,
    read_aux_mm_embeds,
)
from vllm.distributed.aux_output_connector.worker import (
    AuxOutputWorkerConnector,
    _MmEmbeds,
    copy_aux_mm_embeds,
)
from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalKwargsItem,
    PlaceholderRange,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.request import Request

pytestmark = pytest.mark.cpu_test

_MODEL = "llava-hf/llava-1.5-7b-hf"


def _embeds(identifier: str, num_embeds: int) -> torch.Tensor:
    """A distinct BF16 encoder output per item."""
    seed = int(hashlib.sha256(identifier.encode()).hexdigest()[:8], 16)
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(num_embeds, 8, generator=generator).to(torch.bfloat16)


def _key(identifier: str, namespace: str = _NAMESPACE) -> str:
    return f"{_PREFIX}/{namespace}/mm-{identifier}"


def _request(
    request_id: str,
    prompt: list[int],
    items: list[tuple[str, int, int]],
    max_tokens: int = 3,
    *,
    namespace: str | None = _NAMESPACE,
    prompt_start: int = 0,
) -> Request:
    """A request whose items `(identifier, offset, length)` are images."""
    init_none_hash(sha256)
    params = SamplingParams(
        max_tokens=max_tokens,
        ignore_eos=True,
        aux_namespace=namespace,
        aux_prompt_start=prompt_start,
    )
    params.update_from_generation_config({}, EOS_TOKEN_ID)
    return Request(
        request_id=request_id,
        prompt_token_ids=prompt,
        sampling_params=params,
        pooling_params=None,
        mm_features=[
            MultiModalFeatureSpec(
                data=MultiModalKwargsItem.dummy(),
                mm_position=PlaceholderRange(offset=offset, length=length),
                identifier=identifier,
                modality="image",
            )
            for identifier, offset, length in items
        ],
        cache_salt=None
        if namespace is None
        else aux_namespace_cache_salt(_PREFIX, namespace, None),
        block_hasher=get_request_block_hasher(_BLOCK, sha256, 0),
    )


class _MmEngine(_Engine):
    """The aux-objects harness on a multimodal model, with the runner's
    encoder outputs: every item's output is in the encoder cache when the
    step's copy runs."""

    def __init__(self, mm_embeds: bool = True, **kwargs: Any) -> None:
        super().__init__(model=_MODEL, **kwargs)
        manager = self.scheduler.kv_cache_manager
        self.connector = AuxOutputSchedulerConnector(
            manager,
            retained_objects(self.scheduler.vllm_config, manager.kv_cache_config)[0],
            _PREFIX,
            mm_embeds=mm_embeds,
        )
        self.scheduler.aux_output_connector = self.connector
        if mm_embeds:
            self.worker._mm = _MmEmbeds()
        self.encoder_cache = SimpleNamespace(encoder_outputs={})
        self.mm_writes: list[list[tuple[str, str]]] = []
        self.mm_listings: dict[str, list[tuple[str, int, int]] | None] = {}
        prepare_output = self.worker.prepare_output

        def copy_then_prepare(input_batch):
            copy_aux_mm_embeds(self.worker, self.encoder_cache)  # type: ignore[arg-type]
            return prepare_output(input_batch)

        self.worker.prepare_output = copy_then_prepare  # type: ignore[method-assign]

    def add(self, request: Request) -> Request:
        for feature in request.mm_features:
            position = feature.mm_position
            self.encoder_cache.encoder_outputs.setdefault(
                feature.identifier, _embeds(feature.identifier, position.length)
            )
        self.scheduler.add_request(request)
        return request

    def _queue_writes(self, metadata):
        self.mm_writes.append(list(metadata.mm_embed_writes or ()))
        super()._queue_writes(metadata)

    def _final(self, request, objects, aborted):
        # The routed-experts entries go to the aux-objects checks.
        mm = None if objects is None else [o for o in objects if is_mm_embeds_key(o[0])]
        self.mm_listings[request.request_id] = mm
        rows = None if objects is None else [o for o in objects if o not in mm]
        if aborted and not rows:
            rows = None
        super()._final(request, rows, aborted)

    def tags(self, key):
        return [] if is_mm_embeds_key(key) else super().tags(key)

    def written(self, key: str) -> int:
        return sum(keys.count(key) for keys in self.client.writes)

    def read(self, key: str) -> torch.Tensor:
        return read_aux_mm_embeds(self.store, key, timeout_s=0)


def _equal_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.dtype == b.dtype and torch.equal(a.view(torch.int16), b.view(torch.int16))


@pytest.mark.parametrize("async_scheduling", [False, True], ids=["sync", "async"])
def test_siblings_list_their_shared_image_written_once(async_scheduling):
    engine = _MmEngine(async_scheduling=async_scheduling)
    prompt = list(range(64))
    for request_id in ("a", "b"):
        engine.add(_request(request_id, prompt, [("img0", 8, 32)]))
    engine.run()
    engine.pause()

    entry = (_key("img0"), 8, 40)
    assert engine.mm_listings == {"a": [entry], "b": [entry]}
    assert engine.written(_key("img0")) == 1
    assert _equal_bits(
        engine.read(_key("img0")), engine.encoder_cache.encoder_outputs["img0"]
    )
    engine.close()


@pytest.mark.parametrize(
    "prompt_start,listed", [(0, True), (70, False)], ids=["from-0", "after-image"]
)
def test_turn_hitting_its_image_lists_it_from_its_prompt_start(prompt_start, listed):
    engine = _MmEngine()
    first = engine.add(_request("t1", list(range(64)), [("img0", 8, 32)], 4))
    engine.run()
    prompt = list(first.all_token_ids) + list(range(100, 120))
    engine.add(_request("t2", prompt, [("img0", 8, 32)], 2, prompt_start=prompt_start))
    engine.run()
    engine.pause()

    # The second turn hits the image's blocks: it is written once.
    assert engine.hits["t2"] >= 40
    assert engine.written(_key("img0")) == 1
    expected = [(_key("img0"), 8, 40)] if listed else []
    assert engine.mm_listings["t2"] == expected
    engine.close()


def test_image_computed_over_several_steps_is_written_by_the_first():
    engine = _MmEngine(max_num_batched_tokens=16)
    engine.add(_request("a", list(range(64)), [("img0", 8, 32)]))
    engine.run()
    engine.pause()

    steps = [index for index, writes in enumerate(engine.mm_writes) if writes]
    assert len(steps) == 1
    assert engine.mm_writes[steps[0]] == [(_key("img0"), "img0")]
    assert engine.mm_listings["a"] == [(_key("img0"), 8, 40)]
    engine.close()


def test_request_aborted_before_its_second_image_lists_the_first():
    engine = _MmEngine(max_num_batched_tokens=32)
    engine.add(_request("a", list(range(96)), [("img0", 8, 16), ("img1", 64, 16)]))
    scheduler_output = engine.schedule()
    engine.step(scheduler_output)
    _abort(engine, "a")
    engine.pause()

    assert engine.mm_listings["a"] == [(_key("img0"), 8, 24)]
    assert engine.written(_key("img0")) == 1
    assert engine.written(_key("img1")) == 0
    engine.close()


@pytest.mark.parametrize(
    "mm_embeds,namespace", [(False, _NAMESPACE), (True, None)], ids=["off", "inline"]
)
def test_no_encoder_output_objects_without_the_flag_or_a_namespace(
    mm_embeds, namespace
):
    engine = _MmEngine(mm_embeds=mm_embeds)
    engine.add(_request("a", list(range(64)), [("img0", 8, 32)], namespace=namespace))
    engine.run()
    engine.pause()

    assert not any(engine.mm_writes)
    assert engine.mm_listings["a"] in (None, [])
    assert not [key for key in engine.client.objects if is_mm_embeds_key(key)]
    engine.close()


def test_identifier_that_is_not_a_key_segment_is_hashed():
    engine = _MmEngine()
    identifier = "user/image 1"
    engine.add(_request("a", list(range(64)), [(identifier, 8, 32)]))
    engine.run()
    engine.pause()

    key = _key(hashlib.sha256(identifier.encode()).hexdigest())
    assert engine.mm_listings["a"] == [(key, 8, 40)]
    assert _equal_bits(
        engine.read(key), engine.encoder_cache.encoder_outputs[identifier]
    )
    engine.close()


@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float16, torch.float32], ids=str
)
def test_encoder_output_object_keeps_the_bits(dtype):
    client, store = fake_store()
    embeds = torch.randn(5, 7).to(dtype)
    store.put_batch(["k/mm-x"], [save(mm_embeds_fields(embeds))])

    read = read_aux_mm_embeds(store, "k/mm-x", timeout_s=0)
    assert read.dtype == dtype
    assert torch.equal(read.view(torch.uint8), embeds.view(torch.uint8))


def test_worker_writes_copies_no_step_output_took_at_its_next_step():
    client, store = fake_store()
    worker = object.__new__(AuxOutputWorkerConnector)
    worker._buffer = object()
    worker._writer = AuxObjectWriter(store, timeout_s=600.0)
    worker._mm = _MmEmbeds()
    worker._requests = {}
    worker._generation = 0
    worker._pending_outputs = []
    worker._publish_blocks = lambda *args, **kwargs: None  # type: ignore[method-assign]
    worker._lock = threading.Lock()

    def metadata(writes):
        return AuxOutputConnectorMetadata(0, {}, {}, (), mm_embed_writes=writes)

    embeds = _embeds("img0", 4)
    worker.begin_step(metadata([("k/g/mm-img0", "img0")]))
    copy_aux_mm_embeds(worker, SimpleNamespace(encoder_outputs={"img0": embeds}))  # type: ignore[arg-type]
    assert not client.objects
    worker.begin_step(metadata(None))
    wait_for_writer(worker._writer)

    assert _equal_bits(read_aux_mm_embeds(store, "k/g/mm-img0", timeout_s=0), embeds)
    worker._writer.close()


@pytest.mark.parametrize(
    "kwargs,ok",
    [
        ({"enable_return_routed_experts": True, "object_key_prefix": "run"}, True),
        ({"enable_return_routed_experts": True}, False),
        ({"object_key_prefix": "run"}, False),
    ],
    ids=["routed-experts-and-prefix", "no-prefix", "no-routed-experts"],
)
def test_mm_embeds_needs_routed_experts_and_an_object_key_prefix(kwargs, ok):
    config = AuxOutputConfig(enable_return_mm_embeds=True, **kwargs)
    if ok:
        config.verify_mm_embeds()
        return
    with pytest.raises(ValueError, match="enable_return_mm_embeds requires"):
        config.verify_mm_embeds()
