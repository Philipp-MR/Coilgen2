"""Borrowing display arrays never grants mutation of a resumable checkpoint."""

from types import SimpleNamespace

import numpy as np
import pytest

from coilgen import CoilGenPipeline
from coilgen.pipeline_stages import PipelineCheckpoint
from coilgen.application import RunRecord, RunRecordView, JobRuntime
from coilgen.research.jobs import JobManager
from _coilgen_test_support.pipeline_stages import config


def test_selected_checkpoint_arrays_are_read_only_and_resume_remains_detached(tmp_path):
    original = config()
    checkpoint = CoilGenPipeline(original).run_stage("sheet")
    baseline = checkpoint.sf_b_field
    view = checkpoint.array("sheet_field_T", copy=False)
    assert np.shares_memory(view, checkpoint.array("sheet_field_T", copy=False))
    with pytest.raises(ValueError):
        view[:] = 3
    with pytest.raises(ValueError):
        view.flags.writeable = True
    copied = checkpoint.array("sheet_field_T")
    copied[:] = 4
    detached = checkpoint.state
    detached.sf_b_field[:] = 5
    np.testing.assert_array_equal(checkpoint.sf_b_field, baseline)
    only_field = checkpoint.select_arrays(["sheet_field_T"], copy=False, max_bytes=baseline.nbytes)
    assert tuple(only_field) == ("sheet_field_T",)
    with pytest.raises(ValueError, match="size"):
        checkpoint.select_arrays(["sheet_field_T"], max_bytes=baseline.nbytes - 1)
    with pytest.raises(TypeError):
        checkpoint.select_arrays("sheet_field_T")
    restored = PipelineCheckpoint.load(checkpoint.save(tmp_path / "sheet.cgstage"))
    np.testing.assert_array_equal(restored.sf_b_field, baseline)
    resumed = CoilGenPipeline(original).run_stage("contours", restored)
    direct = CoilGenPipeline(original).run_stage("contours")
    for name, array in resumed.scientific_arrays().items():
        np.testing.assert_array_equal(array, direct.array(name))


def test_production_results_and_runtime_satisfy_shared_contracts(tmp_path):
    record = RunRecord(tmp_path, {"ok": True}, {"field": np.ones((2, 3))})
    assert isinstance(record, RunRecordView)
    with JobManager(tmp_path / "jobs") as jobs:
        assert isinstance(jobs, JobRuntime)
        assert jobs.list() == []


def test_pcb_availability_accepts_legacy_records_without_copying_native_metadata():
    from coilgen.application.pcb import pcb_available

    record_metadata = {"completed_stage": "routed", "artifacts": [{"name": "pipeline_checkpoint.cgstage"}]}

    class BorrowedRecord:
        metadata_view = record_metadata

        @property
        def metadata(self):
            pytest.fail("Copied native result metadata for a capability check")

    assert pcb_available(BorrowedRecord())
    assert pcb_available(SimpleNamespace(metadata=record_metadata))
    assert not pcb_available(SimpleNamespace(metadata={}))
    assert not pcb_available(None)
