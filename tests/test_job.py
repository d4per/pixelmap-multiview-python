"""A run on a background thread: followed, polled and stopped."""

import pixelmap_multiview as pmv
import pytest


def test_a_job_says_how_much_work_it_is_before_doing_any(small_photos):
    job = pmv.Job(small_photos)
    first = job.next_event(timeout=30)
    assert isinstance(first, pmv.Started)
    assert (first.views, first.pairs) == (3, 3)
    assert job.status.pairs_total == 3
    job.cancel()
    with pytest.raises(pmv.CancelledError) as caught:
        job.join()
    assert caught.value.reason == "cancelled"
    assert caught.value.stage in {name for name, _ in pmv.STAGES}


def test_a_job_can_be_followed_to_the_end(photos, model):
    job = pmv.Job(photos)
    events = list(job)
    assert isinstance(events[0], pmv.Started)
    assert job.status.progress == pytest.approx(1.0, abs=1e-4)
    assert job.status.pairs_mapped == 6
    finished = job.join()
    # The same photos and seed as the blocking run give the same model.
    assert len(finished.vertices) == len(model.vertices)


def test_polling_without_waiting_returns_at_once(small_photos):
    job = pmv.Job(small_photos)
    job.cancel()
    assert job.cancelled
    while True:
        try:
            job.next_event(timeout=0)
        except StopIteration:
            break
    with pytest.raises(pmv.CancelledError):
        job.join()


def test_bad_photos_come_back_from_join(small_photos):
    job = pmv.Job(small_photos[:2])
    with pytest.raises(pmv.InvalidInputError):
        job.join()


def test_a_job_joins_only_once(small_photos):
    job = pmv.Job(small_photos)
    job.cancel()
    with pytest.raises(pmv.CancelledError):
        job.join()
    with pytest.raises(RuntimeError, match="already been joined"):
        job.join()
