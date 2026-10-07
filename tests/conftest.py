"""Pytest configuration and shared fixtures"""

import sys
import pytest
from tqdm import tqdm


@pytest.fixture
def tracker():
    """Shared progress tracker fixture"""

    class ProgressTracker:
        def __init__(self, desc="Test", total=100):
            self.bar = tqdm(
                total=total,
                desc=desc,
                unit="%",
                file=sys.stderr,
                leave=False,
                dynamic_ncols=True,
                position=1,  # Position below main progress bar
                mininterval=0.1,
            )

        def update(self, amount, status=None):
            self.bar.update(amount)
            if status:
                self.bar.set_description(f"{self.bar.desc} [{status}]")
            self.bar.refresh()

        def close(self):
            self.bar.close()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    return ProgressTracker


# Global progress bar for all tests
_test_progress = None
_test_total = 0


def pytest_configure(config):
    """Configure pytest to disable output capture and show progress"""
    # Disable output capture so tqdm can show real-time progress
    config.option.capture = "no"
    config.option.verbose = 0


def pytest_collection_finish(session):
    """Count total tests after collection"""
    global _test_total
    _test_total = len(session.items)

    # Initialize progress bar immediately after collection
    global _test_progress
    if _test_total > 0:
        _test_progress = tqdm(
            total=_test_total,
            desc="Tests",
            unit=" test",
            file=sys.stderr,
            leave=True,
            dynamic_ncols=True,
            position=0,  # Main progress bar at top
            bar_format="{desc}: {n_fmt}/{total_fmt} |{bar}| {percentage:3.0f}%",
            mininterval=0.05,  # Update every 50ms minimum
        )


def pytest_runtest_logreport(report):
    """Update progress after each test phase completes"""
    global _test_progress

    # Only update on the 'call' phase (when test actually runs)
    if report.when == "call" and _test_progress is not None:
        _test_progress.update(1)
        _test_progress.refresh()


def pytest_sessionfinish(session, exitstatus):
    """Close progress bar after all tests"""
    global _test_progress
    if _test_progress is not None:
        _test_progress.close()
        _test_progress = None
        sys.stderr.write("\n")
        sys.stderr.flush()
