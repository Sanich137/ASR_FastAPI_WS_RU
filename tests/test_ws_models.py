import json
import pytest
from pydantic import ValidationError

from models.ws_models import (
    WSMessageType,
    WSConfigMessage,
    WSAudioMessage,
    WSStatusRequest,
    WSStatusResponse,
    WSResultMessage,
    WSWordItem,
    WSRecognitionData,
    WSErrorMessage,
    WSEosMessage,
    WSPingMessage,
    WSPongMessage,
    wrap_binary_audio,
    ws_message_adapter,
    parse_ws_message,
)


class TestWSConfigMessage:
    def test_defaults(self):
        msg = WSConfigMessage()
        assert msg.type == WSMessageType.config
        assert msg.sample_rate == 16000
        assert msg.wait_null_answers is True
        assert msg.enable_diarization is False
        assert msg.audio_transport == "json_base64"

    def test_audio_transport_binary(self):
        raw = '{"type": "config", "audio_transport": "binary"}'
        msg = WSConfigMessage.model_validate_json(raw)
        assert msg.audio_transport == "binary"

    def test_do_dialogue_and_punctuation(self):
        raw = '{"type": "config", "do_dialogue": true, "do_punctuation": true}'
        msg = WSConfigMessage.model_validate_json(raw)
        assert msg.do_dialogue is True
        assert msg.do_punctuation is True

    def test_from_json(self):
        raw = '{"type": "config", "sample_rate": 8000, "enable_punctuation": true}'
        msg = WSConfigMessage.model_validate_json(raw)
        assert msg.sample_rate == 8000
        assert msg.enable_punctuation is True

    def test_invalid_sample_rate_too_high(self):
        with pytest.raises(ValidationError):
            WSConfigMessage(sample_rate=96000)

    def test_invalid_sample_rate_too_low(self):
        with pytest.raises(ValidationError):
            WSConfigMessage(sample_rate=4000)


class TestWSAudioMessage:
    def test_from_json(self):
        raw = '{"type": "audio_chunk", "audio_base64": "YWJj", "seq_num": 5}'
        msg = WSAudioMessage.model_validate_json(raw)
        assert msg.seq_num == 5
        assert msg.audio_base64 == "YWJj"

    def test_seq_num_must_be_non_negative(self):
        with pytest.raises(ValidationError):
            WSAudioMessage(seq_num=-1)


class TestWSStatusRequest:
    def test_from_json(self):
        raw = '{"type": "status_request"}'
        msg = WSStatusRequest.model_validate_json(raw)
        assert msg.command == "get_status"


class TestWSStatusResponse:
    def test_from_json(self):
        raw = (
            '{"type": "status_response", "adapter_status": "busy", '
            '"gpu_memory_free_mb": 1024, "active_tasks_count": 2}'
        )
        msg = WSStatusResponse.model_validate_json(raw)
        assert msg.adapter_status == "busy"
        assert msg.gpu_memory_free_mb == 1024
        assert msg.active_tasks_count == 2

    def test_invalid_adapter_status(self):
        with pytest.raises(ValidationError):
            WSStatusResponse(adapter_status="dead")

    def test_serialization(self):
        msg = WSStatusResponse(
            adapter_status="overloaded",
            gpu_memory_free_mb=512,
            gpu_memory_total_mb=4096,
            active_connections_count=50,
            uptime_sec=123.45,
        )
        payload = json.loads(msg.model_dump_json())
        assert payload["adapter_status"] == "overloaded"
        assert payload["gpu_memory_free_mb"] == 512
        assert payload["active_connections_count"] == 50
        assert "timestamp" in payload


class TestWSWordItem:
    def test_from_dict(self):
        item = WSWordItem(conf=1.0, start=80.05, end=80.09, word="до")
        assert item.word == "до"
        assert item.conf == 1.0


class TestWSRecognitionData:
    def test_defaults(self):
        data = WSRecognitionData()
        assert data.result == []
        assert data.text == ""

    def test_with_words(self):
        data = WSRecognitionData(
            result=[WSWordItem(conf=1.0, start=0.0, end=1.0, word="привет")],
            text="привет",
        )
        assert len(data.result) == 1
        assert data.text == "привет"


class TestWSResultMessage:
    def test_partial_result_defaults(self):
        data = WSRecognitionData(
            result=[WSWordItem(conf=1.0, start=0.0, end=1.0, word="привет")],
            text="привет",
        )
        msg = WSResultMessage(data=data, last_message=False)
        assert msg.silence is False
        assert msg.type == WSMessageType.partial_result
        assert msg.channel_name == "Null"
        assert msg.data.text == "привет"

    def test_final_result(self):
        data = WSRecognitionData(
            result=[
                WSWordItem(conf=1.0, start=80.05, end=80.09, word="до"),
                WSWordItem(conf=1.0, start=80.21, end=80.57, word="свидания"),
            ],
            text="до свидания",
        )
        msg = WSResultMessage(
            channel_name="channel_1",
            silence=False,
            data=data,
            last_message=True,
            sentenced_data={},
        )
        assert msg.last_message is True
        assert msg.data.text == "до свидания"
        assert len(msg.data.result) == 2

    def test_from_json(self):
        raw = (
            '{"type": "final_result", "channel_name": "ch1", "silence": false, '
            '"data": {"result": [{"conf": 1, "start": 1.0, "end": 2.0, "word": "test"}], '
            '"text": "test"}, "error": null, "last_message": true, "sentenced_data": {}}'
        )
        msg = WSResultMessage.model_validate_json(raw)
        assert msg.channel_name == "ch1"
        assert msg.data.text == "test"
        assert msg.data.result[0].word == "test"


class TestWSErrorMessage:
    def test_parse_error(self):
        msg = WSErrorMessage(code="parse_error", message="Invalid JSON", is_fatal=False)
        assert msg.code == "parse_error"
        assert not msg.is_fatal

    def test_fatal_error(self):
        msg = WSErrorMessage(code="internal_error", message="GPU OOM", is_fatal=True)
        assert msg.is_fatal is True


class TestWSEosMessage:
    def test_type(self):
        msg = WSEosMessage()
        assert msg.type == WSMessageType.eos


class TestWrapBinaryAudio:
    def test_wrap_bytes(self):
        raw = b"\x00\x01\x02"
        msg = wrap_binary_audio(raw, seq_num=7)
        assert msg.type == WSMessageType.audio_chunk
        assert msg.seq_num == 7
        assert msg.audio_base64 == "AAEC"

    def test_wrap_empty(self):
        msg = wrap_binary_audio(b"", seq_num=0)
        assert msg.audio_base64 == ""


class TestWSMessageDiscriminatedUnion:
    def test_union_config(self):
        raw = '{"type": "config", "sample_rate": 16000}'
        msg = ws_message_adapter.validate_json(raw)
        assert isinstance(msg, WSConfigMessage)

    def test_union_audio(self):
        raw = '{"type": "audio_chunk", "seq_num": 0}'
        msg = ws_message_adapter.validate_json(raw)
        assert isinstance(msg, WSAudioMessage)

    def test_union_ping(self):
        raw = '{"type": "ping"}'
        msg = ws_message_adapter.validate_json(raw)
        assert isinstance(msg, WSPingMessage)

    def test_union_pong(self):
        raw = '{"type": "pong"}'
        msg = ws_message_adapter.validate_json(raw)
        assert isinstance(msg, WSPongMessage)

    def test_union_status_request(self):
        raw = '{"type": "status_request"}'
        msg = ws_message_adapter.validate_json(raw)
        assert isinstance(msg, WSStatusRequest)

    def test_union_error(self):
        raw = '{"type": "error", "code": "x", "message": "m"}'
        msg = ws_message_adapter.validate_json(raw)
        assert isinstance(msg, WSErrorMessage)

    def test_union_unknown_type_raises(self):
        raw = '{"type": "foobar"}'
        with pytest.raises(ValidationError):
            ws_message_adapter.validate_json(raw)

    def test_union_invalid_json_raises(self):
        raw = "not a json"
        with pytest.raises(ValidationError):
            ws_message_adapter.validate_json(raw)


class TestParseWsMessageHelper:
    def test_parse_from_str(self):
        raw = '{"type": "pong"}'
        msg = parse_ws_message(raw)
        assert isinstance(msg, WSPongMessage)

    def test_parse_from_dict(self):
        raw = {"type": "eos"}
        msg = parse_ws_message(raw)
        assert isinstance(msg, WSEosMessage)

    def test_parse_from_bytes(self):
        raw = b'{"type": "status_request"}'
        msg = parse_ws_message(raw)
        assert isinstance(msg, WSStatusRequest)
