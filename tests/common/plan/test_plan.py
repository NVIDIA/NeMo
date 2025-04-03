import pytest
from unittest.mock import patch, MagicMock
import time
import functools
from typing import Literal, Callable
from nemo.common.plan.plan import Plan, plan_execution_behavior, PlanExecutionError


@pytest.fixture
def mock_distributed():
    with patch('torch.distributed', autospec=True) as mock:
        mock.is_initialized.return_value = True
        mock.get_rank.return_value = 0
        mock.barrier = MagicMock()
        yield mock


class TestBasicPlan:
    def test_no_children(self):
        """Test that a Plan with no children returns the input or None."""
        plan = Plan()
        assert plan() is None
        assert plan(5) == 5

    def test_single_child(self):
        """Test that a Plan with a single child executes correctly."""
        class ChildPlan(Plan):
            def execute(self, x):
                return x * 2
        plan = Plan(child=ChildPlan())
        assert plan(5) == 10

    def test_multiple_children(self):
        """Test that a Plan with multiple children executes sequentially."""
        class AddPlan(Plan):
            def execute(self, x):
                return x + 1
        class MulPlan(Plan):
            def execute(self, x):
                return x * 2
        plan = Plan(add=AddPlan(), mul=MulPlan())
        assert plan(5) == 12  # (5 + 1) * 2

    def test_nested_plans(self):
        """Test that nested Plans work with overridden execute methods."""
        class InnerPlan(Plan):
            def execute(self, x):
                return x * 2
        class OuterPlan(Plan):
            def __init__(self):
                super().__init__(inner=InnerPlan())
            def execute(self, x):
                result = self.inner(x)
                return result + 1
        plan = OuterPlan()
        assert plan(5) == 11  # (5 * 2) + 1


# class TestThreadedPlan:
#     def test_threaded_sync(self):
#         """Test that threaded='sync' blocks until execution completes."""
#         class SyncPlan(Plan, threaded="sync"):
#             def execute(self, x):
#                 time.sleep(0.1)
#                 return x * 2
#         plan = SyncPlan()
#         start = time.time()
#         result = plan(5)
#         duration = time.time() - start
#         assert result == 10
#         assert duration >= 0.1  # Blocked for at least 0.1s

#     def test_threaded_async(self):
#         """Test that threaded='async' returns immediately and wait() retrieves the result."""
#         class AsyncPlan(Plan, threaded="async"):
#             def execute(self, x):
#                 time.sleep(0.1)
#                 return x * 2
#         plan = AsyncPlan()
#         start = time.time()
#         result = plan(5)
#         duration = time.time() - start
#         assert result is None  # Returns immediately
#         assert duration < 0.05  # Very quick
#         final_result = plan.wait()
#         assert final_result == 10
#         assert time.time() - start >= 0.1  # Total time includes sleep


class TestPrimaryWorkerOnlyPlan:
    def test_skip_mode(self, mock_distributed):
        """Test that primary_worker_only='skip' executes only on rank 0."""
        class SkipPlan(Plan, primary_worker_only="skip"):
            def execute(self, x):
                return x * 2
        plan = SkipPlan()
        # Rank 0
        mock_distributed.get_rank.return_value = 0
        assert plan(5) == 10
        # Rank 1
        mock_distributed.get_rank.return_value = 1
        assert plan(5) is None

    def test_sync_mode(self, mock_distributed):
        """Test that primary_worker_only='sync' executes on rank 0 with barriers."""
        class SyncPlan(Plan, primary_worker_only="sync"):
            def execute(self, x):
                return x * 2
        plan = SyncPlan()
        # Rank 0
        mock_distributed.get_rank.return_value = 0
        mock_distributed.barrier.reset_mock()  # Reset call count
        assert plan(5) == 10
        assert mock_distributed.barrier.call_count == 2  # Two barriers
        # Rank 1
        mock_distributed.get_rank.return_value = 1
        mock_distributed.barrier.reset_mock()
        assert plan(5) is None
        assert mock_distributed.barrier.call_count == 2

    def test_broadcast_mode(self, mock_distributed):
        """Test that primary_worker_only='broadcast' calls broadcast on rank 0."""
        class BroadcastPlan(Plan, primary_worker_only="broadcast"):
            def execute(self, x):
                return x * 2
            
            def broadcast(self, result):
                # Simulate broadcasting: return the computed result for all ranks
                if mock_distributed.get_rank() == 0:
                    return result  # Rank 0 returns the computed result
                else:
                    return 10  # Other ranks receive the "broadcasted" value (hardcoded for this test)

        plan = BroadcastPlan()
        # Rank 0
        mock_distributed.get_rank.return_value = 0
        mock_distributed.barrier.reset_mock()
        assert plan(5) == 10
        assert mock_distributed.barrier.call_count == 2
        # Rank 1
        mock_distributed.get_rank.return_value = 1
        mock_distributed.barrier.reset_mock()
        assert plan(5) == 10  # Now returns 10 due to simulated broadcast
        assert mock_distributed.barrier.call_count == 2 # Barrier calls should still happen

    def test_broadcast_missing_method(self, mock_distributed):
        """Test that missing broadcast method raises the correct error type, possibly wrapped."""
        # Expect PlanExecutionError because __call__ wraps errors at the root.
        # We will check that the original underlying error is NotImplementedError.
        with pytest.raises(PlanExecutionError) as excinfo:
            class NoBroadcastPlan(Plan, primary_worker_only="broadcast"):
                def execute(self, x):
                    return x * 2
            plan = NoBroadcastPlan()
            # Rank 0 execution path where the error occurs
            mock_distributed.get_rank.return_value = 0
            plan(5) # This call will raise PlanExecutionError wrapping NotImplementedError

        # Explicitly check if the raised exception is indeed a PlanExecutionError
        assert isinstance(excinfo.value, PlanExecutionError)
        # Verify the original error type and message stored within PlanExecutionError
        assert isinstance(excinfo.value.original_error, NotImplementedError)
        assert "Plan subclass must implement 'broadcast' method" in str(excinfo.value.original_error)


class TestCustomBehavior:
    def test_log_behavior(self, capsys):
        """Test that a custom log behavior prints when enabled."""
        @plan_execution_behavior(["on", "off"])
        def log(mode: Literal["on", "off"]) -> Callable:
            def decorator(method: Callable) -> Callable:
                @functools.wraps(method)
                def wrapper(self, *args, **kwargs):
                    if mode == "on":
                        print(f"Executing {self.__class__.__name__}")
                    return method(self, *args, **kwargs)
                return wrapper
            return decorator

        class LogPlan(Plan, log="on"):
            def execute(self, x):
                return x * 2

        plan = LogPlan()
        result = plan(5)
        captured = capsys.readouterr()
        assert "Executing LogPlan" in captured.out
        assert result == 10
