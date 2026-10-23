from .schema import expand_instances, parse_plan


def run_planner(original_rgb, cleaned_rgb, regions, backend="gemini", plan_path=None):
    """planner 统一入口：backend=gemini | mock（需 plan_path）。"""
    if backend == "mock":
        if not plan_path:
            raise ValueError("mock planner 需要 plan_path")
        from .mock import load_plan

        return load_plan(plan_path, regions)
    if backend == "gemini":
        from .gemini import run_planner as _run

        return _run(original_rgb, cleaned_rgb, regions)
    raise ValueError(f"未知 planner backend: {backend}")
