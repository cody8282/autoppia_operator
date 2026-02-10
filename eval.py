#!/usr/bin/env python3
"""
Simple evaluation script for the LLM web agent.

Loads 20 tasks from the autoppia_rl task cache, runs each through the
LLMWebAgent.act() loop with the IWA StatefulEvaluator, and prints results.

Usage:
    python eval.py
    python eval.py --num-tasks 5 --max-steps 10
    python eval.py --model gpt-4o-mini --use-case SEARCH_FILM
"""

import asyncio
import json
import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

# ── Ensure the operator repo is on sys.path ─────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
OPERATOR_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(OPERATOR_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

# ── Load .env from autoppia_rl ──────────────────────────────────
from dotenv import load_dotenv

rl_env = OPERATOR_ROOT / "autoppia_rl" / ".env"
if rl_env.exists():
    load_dotenv(rl_env, override=True)

# ── Imports ──────────────────────────────────────────────────────
from loguru import logger

from autoppia_iwa.src.data_generation.tasks.classes import Task
from autoppia_iwa.src.evaluation.stateful_evaluator import AsyncStatefulEvaluator
from autoppia_rl.src.operator.agent import LLMWebAgent
from autoppia_rl.src.operator.runner import EpisodeRunner

# Default task cache path
TASK_CACHE = OPERATOR_ROOT / "autoppia_rl" / "data" / "task_cache" / "autoppia_cinema_tasks.json"

random.seed(time.time())


# ── Task loading ─────────────────────────────────────────────────

def load_tasks(
    cache_path: Path = TASK_CACHE,
    use_case: str | None = None,
    limit: int = 20,
) -> list[Task]:
    """Load tasks from the JSON cache, optionally filtered by use case."""
    with open(cache_path) as f:
        data = json.load(f)

    raw_tasks = data["tasks"] if isinstance(data, dict) and "tasks" in data else data

    tasks: list[Task] = []
    for td in raw_tasks:
        # Optional use-case filter
        if use_case:
            uc = td.get("use_case", {})
            uc_name = uc.get("name", "") if isinstance(uc, dict) else ""
            if use_case.upper() not in uc_name.upper():
                continue

        try:
            task = Task(**td)
            tasks.append(task)
        except Exception as e:
            logger.debug(f"Skipping task {td.get('id', '?')}: {e}")

        if len(tasks) >= limit:
            break

    return tasks


def inject_seed(task: Task) -> tuple[Task, int]:
    """Inject a random seed into the task URL for variation."""
    t = deepcopy(task)
    seed = random.randint(1, 100_000)
    base_url = t.url.split("?")[0] if "?" in t.url else t.url
    t.url = f"{base_url}?seed={seed}"
    return t, seed


# ── Main evaluation loop ────────────────────────────────────────

async def run_evaluation(
    model: str = "gpt-4o-mini",
    num_tasks: int = 20,
    max_steps: int = 15,
    use_case: str | None = None,
    temperature: float = 0.2,
):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not set. Check .env file.")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info(f"  Autoppia Operator – LLM Agent Evaluation")
    logger.info(f"  Model:      {model}")
    logger.info(f"  Tasks:      {num_tasks}")
    logger.info(f"  Max steps:  {max_steps}")
    logger.info(f"  Use case:   {use_case or 'all'}")
    logger.info("=" * 60)

    # Load tasks
    tasks = load_tasks(use_case=use_case, limit=num_tasks)
    logger.info(f"Loaded {len(tasks)} tasks")

    if not tasks:
        logger.error("No tasks found. Check task cache path and use_case filter.")
        return

    # Create agent and runner
    agent = LLMWebAgent(
        model=model,
        temperature=temperature,
        api_key=api_key,
        max_candidates=30,
    )
    runner = EpisodeRunner(agent=agent, max_steps=max_steps)

    # Results tracking
    results = {
        "model": model,
        "num_tasks": 0,
        "successes": 0,
        "failures": 0,
        "errors": 0,
        "timing": {
            "total_seconds": 0.0,
            "avg_task_seconds": 0.0,
            "avg_step_seconds": 0.0,
        },
        "episodes": [],
    }

    t_start = time.time()

    for i, base_task in enumerate(tasks):
        task, seed = inject_seed(base_task)
        uc_name = ""
        if hasattr(task, "use_case") and task.use_case:
            uc = task.use_case
            if isinstance(uc, dict):
                uc_name = uc.get("name", "unknown")
            elif hasattr(uc, "name"):
                uc_name = uc.name
            else:
                uc_name = str(uc)

        logger.info(
            f"[{i + 1}/{len(tasks)}] seed={seed} | {uc_name} | {task.prompt[:50]}..."
        )

        task_start = time.time()
        try:
            episode = await runner.run_episode(task)
            task_elapsed = time.time() - task_start

            results["num_tasks"] += 1
            steps_count = episode.total_steps or 0
            avg_step_seconds = (task_elapsed / steps_count) if steps_count > 0 else 0.0
            ep_data = {
                "task_id": str(task.id),
                "use_case": uc_name,
                "seed": seed,
                "success": episode.final_success,
                "score": episode.final_score,
                "steps": episode.total_steps,
                "task_seconds": round(task_elapsed, 4),
                "avg_step_seconds": round(avg_step_seconds, 4),
            }
            results["episodes"].append(ep_data)

            if episode.final_success:
                results["successes"] += 1
                logger.info(f"  -> SUCCESS (score={episode.final_score:.2f}, steps={episode.total_steps})")
            else:
                results["failures"] += 1
                logger.info(f"  -> FAILED  (score={episode.final_score:.2f}, steps={episode.total_steps})")

        except Exception as e:
            task_elapsed = time.time() - task_start
            results["num_tasks"] += 1
            results["errors"] += 1
            results["episodes"].append({
                "task_id": str(task.id),
                "use_case": uc_name,
                "seed": seed,
                "success": False,
                "score": 0.0,
                "steps": 0,
                "task_seconds": round(task_elapsed, 4),
                "avg_step_seconds": 0.0,
                "error": str(e),
            })
            logger.error(f"  -> ERROR: {e}")

    elapsed = time.time() - t_start

    # ── Summary ──────────────────────────────────────────────────
    total = results["num_tasks"]
    succ = results["successes"]
    rate = succ / total if total > 0 else 0
    avg_score = (
        sum(ep["score"] for ep in results["episodes"]) / total if total > 0 else 0
    )
    avg_steps = (
        sum(ep["steps"] for ep in results["episodes"]) / total if total > 0 else 0
    )
    avg_task_seconds = (
        sum(ep.get("task_seconds", 0.0) for ep in results["episodes"]) / total if total > 0 else 0
    )
    total_steps = sum(ep["steps"] for ep in results["episodes"])
    avg_step_seconds = (
        sum(ep.get("task_seconds", 0.0) for ep in results["episodes"]) / total_steps
        if total_steps > 0
        else 0
    )

    results["timing"]["total_seconds"] = round(elapsed, 4)
    results["timing"]["avg_task_seconds"] = round(avg_task_seconds, 4)
    results["timing"]["avg_step_seconds"] = round(avg_step_seconds, 4)

    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Model:          {model}")
    print(f"  Tasks run:      {total}")
    print(f"  Successes:      {succ}")
    print(f"  Failures:       {results['failures']}")
    print(f"  Errors:         {results['errors']}")
    print(f"  Success rate:   {rate:.1%}")
    print(f"  Avg score:      {avg_score:.3f}")
    print(f"  Avg steps:      {avg_steps:.1f}")
    print(f"  Avg task time:  {avg_task_seconds:.2f}s")
    print(f"  Avg step time:  {avg_step_seconds:.2f}s")
    print(f"  Total time:     {elapsed:.1f}s")
    print("=" * 60)

    # Per-use-case breakdown
    uc_stats: dict[str, dict] = {}
    for ep in results["episodes"]:
        uc = ep.get("use_case", "unknown")
        if uc not in uc_stats:
            uc_stats[uc] = {"total": 0, "success": 0}
        uc_stats[uc]["total"] += 1
        if ep["success"]:
            uc_stats[uc]["success"] += 1

    if len(uc_stats) > 1:
        print("\n  Per use-case breakdown:")
        for uc, st in sorted(uc_stats.items()):
            uc_rate = st["success"] / st["total"] if st["total"] > 0 else 0
            print(f"    {uc:30s}  {st['success']}/{st['total']}  ({uc_rate:.0%})")
        print()

    # Save results
    out_path = SCRIPT_DIR / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to: {out_path}\n")

    return results


# ── CLI ──────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Autoppia Operator - LLM Agent Evaluation")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model name")
    parser.add_argument("--num-tasks", type=int, default=20, help="Number of tasks to evaluate")
    parser.add_argument("--max-steps", type=int, default=15, help="Max steps per episode")
    parser.add_argument("--use-case", default=None, help="Filter by use case (e.g. SEARCH_FILM)")
    parser.add_argument("--temperature", type=float, default=0.2, help="LLM temperature")
    args = parser.parse_args()

    asyncio.run(
        run_evaluation(
            model=args.model,
            num_tasks=args.num_tasks,
            max_steps=args.max_steps,
            use_case=args.use_case,
            temperature=args.temperature,
        )
    )


if __name__ == "__main__":
    main()
