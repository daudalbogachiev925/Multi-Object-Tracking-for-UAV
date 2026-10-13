from src.tracker import MultiObjectTracker

def test_tracker_assigns_ids():
    tr = MultiObjectTracker()
    tr.update([[0, 0, 100, 100]])
    out = tr.update([[5, 5, 105, 105]])
    assert len(out) >= 1
