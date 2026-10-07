from datetime import datetime, timedelta, timezone

import pytest

from coilgen.gui.job_history import format_job_timestamp, job_origin


def test_session_classification_uses_exact_submission_identity():
    initial = {"persisted-running", "persisted-completed"}
    submitted = {"new-job", "resumed-study-new-id"}
    assert job_origin("new-job", initial_ids=initial, session_ids=submitted) == "current"
    assert job_origin("resumed-study-new-id", initial_ids=initial, session_ids=submitted) == "current"
    assert job_origin("persisted-running", initial_ids=initial, session_ids=submitted) == "previous"
    assert job_origin("persisted-completed", initial_ids=initial, session_ids=submitted) == "previous"
    assert job_origin("external-job", initial_ids=initial, session_ids=submitted) == "other"
    assert job_origin("new", initial_ids=initial, session_ids=submitted) == "other"


def test_successful_submission_marker_wins_without_mutating_snapshots():
    initial = {"same-id"}
    submitted = {"same-id"}
    assert job_origin("same-id", initial_ids=initial, session_ids=submitted) == "current"
    assert initial == submitted == {"same-id"}


def test_local_display_crosses_day_boundary_and_keeps_exact_utc_instant():
    local = timezone(timedelta(hours=-4), "EDT")
    display = format_job_timestamp("2026-10-04T02:03:04.123456+00:00", tz=local)
    assert display.text == "2026-10-03 22:03:04"
    assert "Local: 2026-10-03T22:03:04.123456-04:00 (EDT)" in display.tooltip
    assert "UTC: 2026-10-04T02:03:04.123456+00:00" in display.tooltip
    assert display.timestamp == datetime(2026, 10, 4, 2, 3, 4, 123456, tzinfo=timezone.utc).timestamp()


def test_non_utc_source_offset_and_zulu_strings_sort_by_instant():
    first = format_job_timestamp("2026-10-04T08:30:00+05:30", tz=timezone.utc)
    second = format_job_timestamp("2026-10-04T03:00:01Z", tz=timezone.utc)
    assert first.text == "2026-10-04 03:00:00"
    assert second.timestamp > first.timestamp
    assert "UTC: 2026-10-04T03:00:00+00:00" in first.tooltip


def test_legacy_naive_time_is_explicitly_assumed_utc():
    display = format_job_timestamp("2026-10-04T03:02:01", tz=timezone(timedelta(hours=2)))
    assert display.text == "2026-10-04 05:02:01"
    assert "Legacy timestamp had no timezone offset; interpreted as UTC." in display.tooltip


@pytest.mark.parametrize("value", (None, "", " ", "invalid", "2026-10-04", "2026-13-04T03:00:00Z", 0, True))
def test_missing_or_malformed_timestamp_is_unknown(value):
    display = format_job_timestamp(value, tz=timezone.utc)
    assert display.text == "Not recorded"
    assert display.timestamp is None
    assert "No valid execution timestamp" in display.tooltip


def test_host_local_timezone_is_resolved_at_display_time():
    source = "2026-10-04T01:02:03Z"
    expected = datetime.fromisoformat(source.replace("Z", "+00:00")).astimezone()
    display = format_job_timestamp(source)
    assert display.text == expected.strftime("%Y-%m-%d %H:%M:%S")
    assert expected.isoformat() in display.tooltip


def test_repeated_local_time_in_distinct_offsets_retains_chronological_identity():
    earlier = format_job_timestamp("2026-11-01T01:30:00-04:00", tz=timezone.utc)
    later = format_job_timestamp("2026-11-01T01:30:00-05:00", tz=timezone.utc)
    assert later.timestamp - earlier.timestamp == 3600
    assert earlier.text == "2026-11-01 05:30:00"
    assert later.text == "2026-11-01 06:30:00"
