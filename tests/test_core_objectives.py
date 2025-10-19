from qumater.core import ObjectivePlanner, default_objectives


def test_default_objectives_progress_tracking():
    objectives = default_objectives()
    planner = ObjectivePlanner(objectives)

    summary = planner.summary()
    for stats in summary.values():
        assert stats["completed"] == 0.0
        assert stats["progress"] == 0.0

    planner.mark_completed("material-catalogue-selection")
    planner.mark_completed("ansatz-construction")

    summary = planner.summary()
    assert summary["数据到模型的稳健流程"]["completed"] == 1.0
    assert summary["可演化的算法与编排"]["completed"] == 1.0
    assert "material-catalogue-selection" in planner.completed_tasks()


