from data_engineering_copilot.ui.chat_stream_reducer import (
    ChatTurnState,
    apply_chat_event,
    extract_streaming_answer,
    stall_alarm_message,
)


def test_status_updates_and_stage_timer_resets():
    st = ChatTurnState(turn_started=100.0)
    apply_chat_event(st, {"type": "status", "message": "Retrieving documents"}, now=101.0)
    assert st.last_status == "Retrieving documents"
    assert st.stage_started == 101.0


def test_done_breaks_and_sets_text():
    st = ChatTurnState()
    apply_chat_event(st, {"type": "done", "text": "answer", "groundedness_score": 0.9}, now=1.0)
    assert st.done is True and st.should_break is True and st.full_text == "answer"


def test_error_breaks_immediately():
    st = ChatTurnState()
    apply_chat_event(st, {"type": "error", "message": "boom"}, now=1.0)
    assert st.should_break is True
    assert st.error_msg == "boom"


def test_stall_alarm_fires_only_after_threshold():
    st = ChatTurnState(last_status="Reranking", last_event_ts=100.0)
    assert stall_alarm_message(st, now=110.0, stall_seconds=20.0, budget_seconds=90.0) is None
    msg = stall_alarm_message(st, now=122.0, stall_seconds=20.0, budget_seconds=90.0)
    assert msg is not None and "Reranking" in msg and "90" in msg


def test_extract_streaming_answer_still_works():
    assert extract_streaming_answer('{"answer": "hi", "missing_info": null}') == "hi"


def test_interrupted_turn_flag_when_no_done_no_error():
    st = ChatTurnState(full_text="partial …", raw_buffer='{"answer": "partial …', done=False)
    assert st.error_msg is None and st.done is False
    from data_engineering_copilot.ui.streamlit_app import _chat_turn_terminated

    assert _chat_turn_terminated(st) is False


def test_completed_turn_not_interrupted():
    from data_engineering_copilot.ui.streamlit_app import _chat_turn_terminated

    st = ChatTurnState(full_text="full", done=True)
    assert _chat_turn_terminated(st) is True
