"""Tests for classroom mode and lecture protocols."""
from pathlib import Path
import pytest
from conftest import needs_pack


def test_curated_lectures_exist():
    from kickthefly.lab import classroom

    assert len(classroom.CURATED_LECTURES) == 5
    for key in ("looming", "tmaze", "moonwalker", "sugar", "gf_lesion"):
        assert key in classroom.CURATED_LECTURES
        proto = classroom.CURATED_LECTURES[key]
        assert proto.id == key
        assert len(proto.title) > 0
        assert len(proto.steps) >= 3
        for step in proto.steps:
            assert len(step.title) > 0
            assert len(step.explanation) > 0
            assert len(step.key_takeaway) > 0
            assert len(step.citation) > 0
            assert len(step.neurons) > 0
            assert "connectome" in step.grounding.lower() or "sim" in step.grounding.lower() or "em" in step.grounding.lower()


def test_classroom_session_navigation():
    from kickthefly.lab.classroom import ClassroomSession

    sess = ClassroomSession("looming")
    assert sess.step_idx == 0
    assert sess.total_steps == 4
    assert not sess.prev_step()  # Cannot go back from step 0

    # Step forward
    assert sess.next_step()
    assert sess.step_idx == 1

    # Goto step
    assert sess.goto_step(3)
    assert sess.step_idx == 3
    assert not sess.next_step()  # Cannot go past last step

    # Step backward
    assert sess.prev_step()
    assert sess.step_idx == 2

    # Reset
    sess.reset()
    assert sess.step_idx == 0


@pytest.mark.parametrize("lec_file", [
    "lecture_looming.yaml",
    "lecture_tmaze.yaml",
    "lecture_moonwalker.yaml",
    "lecture_sugar.yaml",
    "lecture_gf_lesion.yaml",
])
def test_lecture_yaml_files(lec_file):
    from kickthefly.lab import protocol
    path = Path("protocols") / lec_file
    assert path.exists(), f"{lec_file} does not exist in protocols/"
    p = protocol.load(path)
    assert p.get("classroom") is True
    assert len(p.get("steps", [])) >= 3
    for s in p["steps"]:
        assert "title" in s
        assert "explanation" in s
        assert "citation" in s
        assert "neurons" in s


@needs_pack
def test_classroom_step_action_execution():
    from kickthefly.core import simcore
    from kickthefly.lab.classroom import ClassroomSession

    br = simcore.new_brain(seed=42, warmup=10)
    sess = ClassroomSession("looming", brain=br)

    # Step 1: reset action
    res1 = sess.execute_step_action(br)
    assert res1["status"] == "ok"

    # Step 2: stimulus action
    sess.next_step()
    res2 = sess.execute_step_action(br)
    assert res2["status"] == "ok"
    assert ("classroom", "type:LPLC2,LC4") in br.sense

    # Step 3: drive action
    sess.next_step()
    res3 = sess.execute_step_action(br)
    assert res3["status"] == "ok"
