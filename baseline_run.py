"""Command-line runner for the Bagloee--Sarvi--Wallace inspired BB baseline.

Examples
--------
Run the Bagloee--Sarvi--Wallace inspired branch-and-bound adaptation on the
three default networks (SF, EMA and Anaheim)::

    /opt/anaconda3/bin/python baseline_run.py

Run the exact same candidate corridors used in an experiment::

    /opt/anaconda3/bin/python baseline_run.py --groups-json my_groups.json \
        --budget 120 --network my_net.tntp --trips my_trips.tntp

``groups-json`` must contain a JSON list of groups, where each group is a list
of ``[tail, head]`` pairs using zero-based node ids (the repository's internal
convention).  Add ``--groups-one-based`` when the file uses TNTP labels.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
from typing import Sequence

from baseline_common import (
    generate_disjoint_bidirectional_groups,
    generate_legacy_connected_fixed_groups,
    make_instance,
    read_tntp_demand,
    read_tntp_network,
    SolveResult,
    exact_optimistic_follower,
)
from baseline_bagloee_bb import solve_bagloee2016_inspired_branch_and_bound


METHODS = {
    "bagloee_bb": solve_bagloee2016_inspired_branch_and_bound,
}


BUILTIN_NETWORKS = {
    "sf": ("SF", "SiouxFalls/SiouxFalls_net.tntp", "SiouxFalls/SiouxFalls_trips.tntp"),
    "ema": (
        "EMA",
        "Eastern-Massachusetts/EMA_net.tntp",
        "Eastern-Massachusetts/EMA_trips.tntp",
    ),
    "anaheim": ("Anaheim", "Anaheim/Anaheim_net.tntp", "Anaheim/Anaheim_trips.tntp"),
}


# The first fields deliberately follow results_with_relative.csv, so the
# baseline table can be compared and joined with the manuscript's BPC table.
# `method` distinguishes the active published bicycle-lane baseline adaptation.
BASELINE_CSV_HEADERS = [
    "method",
    "network", "B", "seed", "g", "groups", "length", "value_cut",
    "Opt", "iter", "T_all", "T_Hpr", "T_Short_path", "T_Update", "T_value_cut",
    "UB", "LB", "gap", "value_cut_number", "Opt_base", "Opt_relative",
    "status", "exact_certificate", "formulation_exact", "nodes",
    "selected_group_indices", "design_cost", "candidate_mode", "group_costs",
    "time_limit_seconds", "candidate_groups_zero_based", "scenario", "metadata", "message",
]


def _parse_csv_floats(text: str, option_name: str) -> tuple[float, ...]:
    try:
        values = tuple(float(piece.strip()) for piece in text.split(",") if piece.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{option_name} must be a comma-separated float list") from exc
    if not values:
        raise argparse.ArgumentTypeError(f"{option_name} cannot be empty")
    return values


def _load_groups(path: str, *, one_based: bool) -> tuple[tuple[tuple[int, int], ...], ...]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("groups JSON must be a list")
    groups = []
    for group_index, raw_group in enumerate(raw):
        if not isinstance(raw_group, list):
            raise ValueError(f"groups JSON group {group_index} is not a list")
        group = []
        for arc in raw_group:
            if not isinstance(arc, (list, tuple)) or len(arc) != 2:
                raise ValueError(f"invalid arc in groups JSON group {group_index}: {arc!r}")
            tail, head = int(arc[0]), int(arc[1])
            if one_based:
                tail -= 1
                head -= 1
            group.append((tail, head))
        groups.append(tuple(group))
    return tuple(groups)


def _append_csv_row(path: Path, row: dict[str, object]) -> None:
    """Append one completed baseline result and flush it to durable CSV output."""
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=BASELINE_CSV_HEADERS, extrasaction="ignore")
        if needs_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in BASELINE_CSV_HEADERS})
        handle.flush()


def _result_gap(result: SolveResult) -> float | str:
    """Return a mathematically meaningful gap only when both bounds exist."""
    if result.upper_bound is None or result.lower_bound is None:
        return ""
    lower = float(result.lower_bound)
    upper = float(result.upper_bound)
    if lower <= 1e-8:
        return 0.0 if upper <= 1e-8 else "inf"
    denominator = abs(lower)
    return max(0.0, (upper - lower) / denominator)


def _csv_row(
    *,
    method_key: str,
    result: SolveResult,
    instance,
    args,
    base_objective: float,
    candidate_mode: str,
    network_name: str,
    betas: Sequence[float],
    scenario: str,
) -> dict[str, object]:
    """Map a solver result to the results_with_relative-compatible CSV row."""
    relative = "" if result.objective is None else float(result.objective) / base_objective
    selected = [] if result.z is None else [index + 1 for index, value in enumerate(result.z) if value == 1]
    return {
        "method": method_key,
        "network": network_name,
        "B": args.budget if args.budget is not None else args.budget_share,
        "seed": args.seed,
        "g": len(betas),
        "groups": instance.n_groups,
        "length": args.group_links,
        "value_cut": False,
        "Opt": result.objective if result.objective is not None else "",
        "iter": result.iterations if result.iterations is not None else "",
        "T_all": result.runtime_seconds,
        # These are BPC-specific timing buckets.  The exact baseline methods
        # do not execute the same phases, so blanks are more honest than a
        # fabricated decomposition of total wall-clock time.
        "T_Hpr": "",
        "T_Short_path": "",
        "T_Update": "",
        "T_value_cut": 0.0,
        "UB": result.upper_bound if result.upper_bound is not None else "",
        "LB": result.lower_bound if result.lower_bound is not None else "",
        "gap": _result_gap(result),
        "value_cut_number": 0,
        "Opt_base": base_objective,
        "Opt_relative": relative,
        "status": result.status,
        "exact_certificate": result.certified_optimal,
        "formulation_exact": result.formulation_exact,
        "nodes": result.nodes if result.nodes is not None else "",
        "selected_group_indices": json.dumps(selected),
        "design_cost": "" if result.z is None else instance.design_cost(result.z),
        "candidate_mode": candidate_mode,
        "group_costs": json.dumps(list(instance.group_costs)),
        "time_limit_seconds": args.time_limit,
        "candidate_groups_zero_based": json.dumps(
            [[list(instance.arcs[a]) for a in group] for group in instance.groups]
        ),
        "scenario": scenario,
        "metadata": json.dumps(result.metadata, ensure_ascii=False),
        "message": result.message,
    }


def _solve_method(method_key: str, instance, args) -> SolveResult:
    """Dispatch the single active BB baseline."""
    if method_key == "bagloee_bb":
        return solve_bagloee2016_inspired_branch_and_bound(
            instance,
            time_limit=args.time_limit,
            verbose=args.verbose,
            node_time_limit=args.bb_node_time_limit,
            local_search_passes=args.bb_local_search_passes,
        )
    raise ValueError(f"unsupported baseline method: {method_key}")


def _write_json_snapshot(path: Path, cases: list[dict[str, object]]) -> None:
    """Keep the optional JSON snapshot consistent with all completed cases."""
    path.write_text(
        json.dumps({"cases": cases}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _resolve_networks(args) -> tuple[tuple[str, str, str], ...]:
    """Resolve built-in batch names or one explicitly supplied TNTP pair."""
    if args.network is not None or args.trips is not None:
        if args.network is None or args.trips is None:
            raise ValueError("--network and --trips must be supplied together")
        return ((Path(args.network).stem, args.network, args.trips),)
    selected = tuple(name.strip().lower() for name in args.networks.split(",") if name.strip())
    unknown = [name for name in selected if name not in BUILTIN_NETWORKS]
    if not selected or unknown:
        raise ValueError(
            f"--networks must be a non-empty subset of {', '.join(BUILTIN_NETWORKS)}; unknown={unknown}"
        )
    return tuple(BUILTIN_NETWORKS[name] for name in selected)


def _build_instance_for_network(args, betas, shares, network_file: str, trips_file: str):
    """Build one network instance using the shared experiment parameters."""
    edges, travel_times = read_tntp_network(network_file, use_free_flow_time=args.use_free_flow_time)
    n_nodes = max(max(u, v) for u, v, _ in edges) + 1
    demand_matrix = read_tntp_demand(trips_file, n_nodes)
    if args.groups_json:
        groups = _load_groups(args.groups_json, one_based=args.groups_one_based)
        candidate_mode = "json"
    elif args.candidate_mode == "legacy":
        groups = generate_legacy_connected_fixed_groups(
            edges,
            group_count=args.groups,
            links_per_group=args.group_links,
            seed=args.seed,
        )
        candidate_mode = "legacy"
    else:
        groups = generate_disjoint_bidirectional_groups(
            edges,
            group_count=args.groups,
            links_per_group=args.group_links,
            seed=args.seed,
        )
        candidate_mode = "corridor"

    # Build once with zero budget solely to obtain the exact directed-arc
    # construction costs.  The second instance receives B=0.5 (or --budget).
    provisional = make_instance(
        edges=edges,
        demand_matrix=demand_matrix,
        groups=groups,
        budget=0.0,
        betas=betas,
        class_shares=shares,
        travel_times=travel_times,
        n_nodes=n_nodes,
    )
    budget = float(args.budget) if args.budget is not None else args.budget_share * sum(provisional.group_costs)
    instance = make_instance(
        edges=edges,
        demand_matrix=demand_matrix,
        groups=groups,
        budget=budget,
        betas=betas,
        class_shares=shares,
        travel_times=travel_times,
        n_nodes=n_nodes,
    )
    return instance, candidate_mode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Bagloee et al. (2016) inspired BB baseline on the manuscript model."
    )
    parser.add_argument(
        "--networks",
        default="sf,ema,anaheim",
        help="comma-separated built-in batch: sf,ema,anaheim (default: all three)",
    )
    parser.add_argument("--network", help="custom TNTP network file; requires --trips and overrides --networks")
    parser.add_argument("--trips", help="custom TNTP OD-demand file; requires --network")
    candidates = parser.add_argument_group("candidate corridors")
    candidates.add_argument("--groups-json", help="JSON candidate groups; overrides random generation")
    candidates.add_argument("--groups-one-based", action="store_true", help="JSON candidate nodes use TNTP's 1-based labels")
    candidates.add_argument("--groups", type=int, default=20, help="number of generated candidate corridors (default: 20)")
    candidates.add_argument("--group-links", type=int, default=1, help="adjacent physical links per generated corridor (default: 1)")
    candidates.add_argument("--seed", type=int, default=10100, help="seed for generated candidate corridors")
    candidates.add_argument(
        "--candidate-mode",
        choices=("legacy", "corridor"),
        default="legacy",
        help=(
            "legacy reproduces main_run.py grouping (default, for comparison with results.csv); "
            "corridor uses the newer random physical-corridor generator"
        ),
    )
    parser.add_argument("--budget", type=float, help="absolute construction budget; takes precedence over --budget-share")
    parser.add_argument("--budget-share", type=float, default=0.5, help="fraction of all candidate group cost used as budget")
    parser.add_argument("--betas", default="0.5", help="comma-separated beta_g values, e.g. 0.5,0.7")
    parser.add_argument("--class-shares", help="comma-separated class demand shares; defaults to equal shares")
    parser.add_argument(
        "--use-free-flow-time",
        action="store_true",
        help="use TNTP free-flow-time as T; default keeps the current manuscript code's T=L convention",
    )
    parser.add_argument(
        "--methods",
        default="bagloee_bb",
        help="kept for script compatibility; only bagloee_bb is currently supported",
    )
    bb = parser.add_argument_group("Bagloee 2016 inspired branch-and-bound")
    bb.add_argument(
        "--bb-node-time-limit",
        type=float,
        default=2.0,
        help="seconds allowed for each node SO/HPR relaxation in bagloee_bb (default: 2)",
    )
    bb.add_argument(
        "--bb-local-search-passes",
        type=int,
        default=3,
        help="deterministic add/drop/swap local-search passes for initial UB in bagloee_bb (default: 3)",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=36_000.0,
        help="per-method wall-clock limit in seconds (default: 36000 = 10 hours)",
    )
    parser.add_argument("--verbose", action="store_true", help="show Gurobi output for BB node relaxations")
    parser.add_argument(
        "--csv-output",
        default="bagloee_bb_baseline_results_with_relative.csv",
        help="append each completed baseline method to this CSV immediately",
    )
    parser.add_argument(
        "--output",
        help="optional JSON snapshot path; refreshed after each completed baseline method",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    betas = _parse_csv_floats(args.betas, "--betas")
    shares = _parse_csv_floats(args.class_shares, "--class-shares") if args.class_shares else None
    requested = tuple(item.strip().lower() for item in args.methods.split(",") if item.strip())
    unknown = [name for name in requested if name not in METHODS]
    if not requested or unknown:
        raise ValueError(f"--methods must be a non-empty subset of {', '.join(METHODS)}; unknown={unknown}")
    if args.time_limit is not None and args.time_limit <= 0:
        raise ValueError("--time-limit must be positive")
    if args.bb_node_time_limit is not None and args.bb_node_time_limit <= 0:
        raise ValueError("--bb-node-time-limit must be positive")
    if args.bb_local_search_passes < 0:
        raise ValueError("--bb-local-search-passes must be non-negative")
    if args.budget is None and not (0.0 <= args.budget_share <= 1.0):
        raise ValueError("--budget-share must lie in [0, 1]")
    network_cases = _resolve_networks(args)
    if args.groups_json and len(network_cases) != 1:
        raise ValueError("--groups-json is network-specific; select exactly one network with --networks")
    csv_path = Path(args.csv_output)
    print(f"CSV results: {csv_path} (append after every completed method)")
    completed_cases: list[dict[str, object]] = []
    had_error = False

    execution_cases: list[tuple[str, str, str, tuple[float, ...], tuple[float, ...] | None, str]] = [
        (network_name, network_file, trips_file, betas, shares, f"{network_name}_g{len(betas)}")
        for network_name, network_file, trips_file in network_cases
    ]

    for network_name, network_file, trips_file, case_betas, case_shares, scenario in execution_cases:
        try:
            instance, candidate_mode = _build_instance_for_network(
                args, case_betas, case_shares, network_file, trips_file
            )
        except Exception as exc:
            had_error = True
            print(f"{network_name} | setup ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
            completed_cases.append(
                {
                    "model": {"network": network_name, "network_file": network_file, "trips": trips_file},
                    "results": [{"status": "SETUP_ERROR", "message": f"{type(exc).__name__}: {exc}"}],
                }
            )
            if args.output:
                _write_json_snapshot(Path(args.output), completed_cases)
            continue

        print(f"\n=== {network_name} ({scenario}) ===")
        print(
            f"Instance: nodes={instance.n_nodes}, arcs={instance.n_arcs}, "
            f"commodities={instance.n_commodities}, groups={instance.n_groups}, budget={instance.budget:.6g}"
        )
        print(f"Group costs (directed-arc convention): {list(instance.group_costs)}")
        base_objective, _ = exact_optimistic_follower(
            instance, tuple(0 for _ in range(instance.n_groups))
        )
        print(f"No-build baseline objective (Opt_base): {base_objective}")
        model_payload = {
            "definition": "manuscript bicycle-lane optimistic shortest-path bilevel model",
            "network": network_name,
            "B": args.budget if args.budget is not None else args.budget_share,
            "seed": args.seed,
            "g": len(case_betas),
            "groups": instance.n_groups,
            "length": args.group_links,
            "candidate_mode": candidate_mode,
            "value_cut": False,
            "T_equals_L": not args.use_free_flow_time,
            "network_file": network_file,
            "trips": trips_file,
            "betas": list(case_betas),
            "class_shares": None if case_shares is None else list(case_shares),
            "scenario": scenario,
            "budget": instance.budget,
            "time_limit_seconds": args.time_limit,
            "Opt_base": base_objective,
            "group_costs": list(instance.group_costs),
            "groups_zero_based": [[list(instance.arcs[a]) for a in group] for group in instance.groups],
        }
        case_payload: dict[str, object] = {"model": model_payload, "results": []}
        completed_cases.append(case_payload)

        for name in requested:
            method_started = time.perf_counter()
            try:
                result = _solve_method(name, instance, args)
            except Exception as exc:  # Preserve prior rows and continue to the next baseline/network.
                result = SolveResult(
                    method=name,
                    status="ERROR",
                    runtime_seconds=time.perf_counter() - method_started,
                    certified_optimal=False,
                    formulation_exact=False,
                    message=f"{type(exc).__name__}: {exc}",
                )
            if result.status == "ERROR":
                had_error = True
            _append_csv_row(
                csv_path,
                _csv_row(
                    method_key=name,
                    result=result,
                    instance=instance,
                    args=args,
                    base_objective=base_objective,
                    candidate_mode=candidate_mode,
                    network_name=network_name,
                    betas=case_betas,
                    scenario=scenario,
                ),
            )
            case_payload["results"].append(asdict(result))
            if args.output:
                _write_json_snapshot(Path(args.output), completed_cases)
            print(
                f"{name:>3} | status={result.status:<18} exact_certificate={result.certified_optimal!s:<5} "
                f"objective={result.objective} z={result.z} runtime={result.runtime_seconds:.3f}s"
            )
            print(f"     appended to {csv_path}")
            if result.message:
                print(f"      {result.message}")

    if args.output:
        print(f"JSON snapshot: {args.output}")
    return 0 if not had_error else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"baseline_run: {exc}", file=sys.stderr)
        raise SystemExit(2)
