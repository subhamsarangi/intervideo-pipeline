"""LLM call tracking utilities with cost calculation"""

from typing import Optional, Dict, List
from contextlib import contextmanager


# Model pricing (per 1M tokens, 2026 rates)
MODEL_COSTS = {
    "gpt-5.4-nano": {
        "input": 0.075,  # $0.075 per 1M input tokens
        "output": 0.30,  # $0.30 per 1M output tokens
    },
    "gpt-5.4-mini": {
        "input": 0.15,
        "output": 0.60,
    },
    "gpt-4o-mini": {
        "input": 0.15,
        "output": 0.60,
    },
    "gpt-4o": {
        "input": 2.50,
        "output": 10.00,
    },
    "text-embedding-3-small": {
        "input": 0.02,  # $0.02 per 1M tokens
        "output": 0.0,
    },
    "text-embedding-3-large": {
        "input": 0.13,  # $0.13 per 1M tokens
        "output": 0.0,
    },
    "tavily-search": {
        "per_call": 0.005,  # $0.005 per search API call (1 Tavily credit)
    },
    "llamaparse": {
        "per_call": 0.003,  # $0.003 per page/doc parsed (standard tier rate)
    },
}


class LLMCallTracker:
    """Track LLM API calls with actual token counts for cost and performance analysis"""

    def __init__(self):
        self.call_count = 0
        self.calls: List[Dict] = []

    def increment(
        self,
        operation: str = "api_call",
        model: str = "gpt-5.4-nano",
        input_tokens: int = 0,
        output_tokens: int = 0,
    ):
        """Record an LLM call with actual token counts"""
        self.call_count += 1

        self.calls.append(
            {
                "count": self.call_count,
                "operation": operation,
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
        )

    def get_count(self) -> int:
        """Get total call count"""
        return self.call_count

    def get_calls(self) -> List[Dict]:
        """Get all recorded calls"""
        return self.calls

    def get_total_tokens(self) -> Dict[str, int]:
        """Get total input and output tokens"""
        total_input = sum(c["input_tokens"] for c in self.calls)
        total_output = sum(c["output_tokens"] for c in self.calls)
        return {"input": total_input, "output": total_output}

    def calculate_cost(self) -> Dict[str, float]:
        """Calculate total cost based on actual tokens, per-call pricing, and model rates"""
        tokens = self.get_total_tokens()
        total_cost = 0.0
        cost_by_model = {}

        for call in self.calls:
            model = call["model"]
            if model not in MODEL_COSTS:
                continue

            costs = MODEL_COSTS[model]
            if "per_call" in costs:
                call_cost = costs["per_call"]
            else:
                input_cost = (call["input_tokens"] * costs.get("input", 0.0)) / 1_000_000
                output_cost = (call["output_tokens"] * costs.get("output", 0.0)) / 1_000_000
                call_cost = input_cost + output_cost

            total_cost += call_cost
            if model not in cost_by_model:
                cost_by_model[model] = 0.0
            cost_by_model[model] += call_cost

        return {
            "total_cost_usd": round(total_cost, 6),
            "cost_by_model": {k: round(v, 6) for k, v in cost_by_model.items()},
            "total_input_tokens": tokens["input"],
            "total_output_tokens": tokens["output"],
        }

    def reset(self):
        """Reset tracker"""
        self.call_count = 0
        self.calls = []

    def __str__(self) -> str:
        """String representation with cost breakdown"""
        cost_info = self.calculate_cost()
        lines = [
            f"LLM Calls: {self.call_count}",
            f"Total Cost: ${cost_info['total_cost_usd']:.6f}",
            f"Tokens: {cost_info['total_input_tokens']} input + {cost_info['total_output_tokens']} output",
        ]
        for model, cost in cost_info["cost_by_model"].items():
            lines.append(f"  {model}: ${cost:.6f}")
        return "\n".join(lines)


# Global tracker instance
_tracker: Optional[LLMCallTracker] = None


def get_tracker() -> LLMCallTracker:
    """Get or create global tracker"""
    global _tracker
    if _tracker is None:
        _tracker = LLMCallTracker()
    return _tracker


def reset_tracker():
    """Reset global tracker"""
    global _tracker
    _tracker = None


def increment_llm_calls(
    operation: str = "api_call",
    model: str = "gpt-5.4-nano",
    input_tokens: int = 0,
    output_tokens: int = 0,
):
    """Increment global LLM call counter with actual token counts"""
    tracker = get_tracker()
    tracker.increment(operation, model, input_tokens, output_tokens)


def get_llm_call_count() -> int:
    """Get current LLM call count"""
    tracker = get_tracker()
    return tracker.get_count()


def get_llm_cost() -> Dict[str, float]:
    """Get current LLM cost"""
    tracker = get_tracker()
    return tracker.calculate_cost()


def get_llm_summary() -> str:
    """Get formatted LLM tracking summary"""
    tracker = get_tracker()
    return str(tracker)


def get_operation_averages() -> Dict[str, Dict[str, float]]:
    """Calculate average tokens per operation from recorded calls"""
    tracker = get_tracker()
    operations = {}

    for call in tracker.get_calls():
        op = call["operation"]
        if op not in operations:
            operations[op] = {"calls": 0, "total_input": 0, "total_output": 0}

        operations[op]["calls"] += 1
        operations[op]["total_input"] += call["input_tokens"]
        operations[op]["total_output"] += call["output_tokens"]

    # Calculate averages
    averages = {}
    for op, data in operations.items():
        if data["calls"] > 0:
            averages[op] = {
                "avg_input": round(data["total_input"] / data["calls"]),
                "avg_output": round(data["total_output"] / data["calls"]),
                "call_count": data["calls"],
            }

    return averages


def print_operation_averages():
    """Print formatted operation averages"""
    averages = get_operation_averages()
    if not averages:
        print("No LLM calls recorded yet.")
        return

    print("\nLLM Operation Averages:")
    for op, data in averages.items():
        print(
            f"  {op}: {data['avg_input']} input + {data['avg_output']} output tokens (from {data['call_count']} calls)"
        )
