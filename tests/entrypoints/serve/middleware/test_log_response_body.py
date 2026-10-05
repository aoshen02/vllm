# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The debug response log shows a non-streaming body sent in several messages."""

from vllm.entrypoints.serve.middleware import log_response


def test_non_streaming_body_in_several_messages_is_logged_whole(monkeypatch):
    logged = []
    monkeypatch.setattr(
        log_response.logger, "info", lambda msg, *args: logged.append(msg % args)
    )
    log_response._log_non_streaming_response([b'{"a": ', b"1}"])
    assert logged == ['response_body={{"a": 1}}']
