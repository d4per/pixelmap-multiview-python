"""What a run reports while it works, and stopping it from the callback."""

import numpy as np
import pixelmap_multiview as pmv
import pytest


def test_the_run_announces_how_much_work_it_is(events):
    first = events[0]
    assert isinstance(first, pmv.Started)
    assert (first.views, first.pairs) == (4, 6)
    assert first.stage == "input"


def test_progress_only_moves_forward_and_reaches_the_end(events):
    progress = [e.progress for e in events if e.progress is not None]
    assert progress == sorted(progress)
    assert progress[-1] == pytest.approx(1.0, abs=1e-4)


def test_every_event_names_a_known_stage(events):
    names = {name for name, _ in pmv.STAGES}
    assert {e.stage for e in events} <= names
    assert all(isinstance(e.message, str) and e.message for e in events)


def test_each_pair_hands_over_its_correspondence(events):
    mapped = [e for e in events if isinstance(e, pmv.PairMapped)]
    assert sorted(e.pair for e in mapped) == [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 3),
    ]
    for event in mapped:
        assert event.of == 6
        assert event.points.dtype == np.float32
        assert event.points.ndim == 3 and event.points.shape[2] == 2
        mapped_cells = ~np.isnan(event.points[..., 0])
        assert mapped_cells.any()
        # Mapped points land inside the second photo, give or take a cell.
        xs = event.points[..., 0][mapped_cells]
        assert xs.min() > -event.cell_size and xs.max() < 640 + event.cell_size


def test_pair_progress_counts_solver_steps(events):
    steps = [e for e in events if isinstance(e, pmv.PairProgress)]
    assert steps
    assert all(0 <= e.step <= e.steps and e.index < e.of for e in steps)


def test_an_exception_in_the_callback_stops_the_run_and_propagates(small_photos):
    seen = []

    class Enough(Exception):
        pass

    def on_event(event):
        seen.append(event)
        if isinstance(event, pmv.PairProgress):
            raise Enough

    with pytest.raises(Enough):
        pmv.reconstruct(small_photos, on_event=on_event)
    # Stopped at the checkpoint that raised, not at the end of the run.
    assert isinstance(seen[-1], pmv.PairProgress)
    assert not any(isinstance(e, pmv.PairMapped) for e in seen)
