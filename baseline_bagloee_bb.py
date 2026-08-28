"""Bagloee--Sarvi--Wallace 2016 inspired branch-and-bound baseline.

This is a *model adaptation* of the solution logic in Bagloee, Sarvi and
Wallace (2016), "Bicycle lane priority: Promoting bicycle as a green mode even
in congested urban area", to the manuscript's shortest-path bilevel model.

The original paper solves a bicycle-priority-lane bilevel problem whose lower
level is a nonlinear multiclass user-equilibrium traffic assignment.  The
available TNTP benchmark instances in this repository do not contain the
paper's lane counts, capacity transformations, class-specific bias functions
and MUE/SO assignment implementation.  Therefore this file does not claim to
reproduce the original MUE model.  It keeps the paper's algorithmic structure:

* branch on link/corridor design variables;
* use a system-optimal/HPR relaxation as a valid lower bound at partial nodes;
* evaluate complete binary designs with the true lower-level response to get
  upper bounds; and
* prune nodes whose lower bound cannot improve the incumbent.

For the manuscript model, the true lower-level response is the exact
optimistic shortest-path evaluator in ``baseline_common.exact_optimistic_follower``.
The HPR/SO node relaxation drops only the follower-optimality condition while
preserving the binary project decisions that remain free at the node.  This is
a stronger system-optimal design lower-bound problem than the continuous
relaxation and is closer to the paper's SO bounding idea.  The outer
branch-and-bound over the original bilevel design decisions is implemented
explicitly here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Mapping, Sequence

import gurobipy as gp
from gurobipy import GRB

from baseline_common import BaselineInstance, SolveResult, exact_optimistic_follower, remaining_time


@dataclass
class _Node:
    fixed: dict[int, int]
    inherited_lower_bound: float = 0.0


@dataclass
class _RelaxationResult:
    status: str
    objective: float | None = None
    lower_bound: float | None = None
    z_values: tuple[float, ...] | None = None
    message: str = ""


@dataclass
class _Incumbent:
    value: float
    z: tuple[int, ...]


def _status_name(status: int) -> str:
    names = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.NUMERIC: "NUMERIC",
    }
    return names.get(status, f"GUROBI_STATUS_{status}")


def _fixed_cost(instance: BaselineInstance, fixed: Mapping[int, int]) -> float:
    return float(sum(instance.group_costs[j] for j, value in fixed.items() if value == 1))


def _is_integral(values: Sequence[float], tolerance: float) -> bool:
    return all(abs(value - round(value)) <= tolerance for value in values)


def _as_binary(values: Sequence[float]) -> tuple[int, ...]:
    return tuple(1 if float(value) >= 0.5 else 0 for value in values)


class _HprSoWorkspace:
    """Persistent HPR/SO relaxation for B&B node lower bounds."""

    def __init__(
        self,
        instance: BaselineInstance,
        *,
        verbose: bool = False,
        binary_design: bool = True,
    ) -> None:
        self.instance = instance
        self.binary_design = bool(binary_design)
        self.model_kind = "MIP" if self.binary_design else "LP"
        self.model = gp.Model(f"Bagloee2016_inspired_HPR_SO_{self.model_kind}")
        self.model.Params.OutputFlag = 1 if verbose else 0
        self.model.Params.FeasibilityTol = 1e-8
        self.model.Params.OptimalityTol = 1e-8
        self.model.Params.NumericFocus = 1
        self.model.Params.IntFeasTol = 1e-9
        self.model.Params.MIPGap = 0.0
        self.model.Params.MIPGapAbs = 0.0
        # Repeated node relaxations differ mainly by variable bounds.  Dual
        # simplex is usually the most stable warm-start method for this pattern.
        self.model.Params.Method = 1

        K, A, N, J = (
            instance.n_commodities,
            instance.n_arcs,
            instance.n_nodes,
            instance.n_groups,
        )
        z_type = GRB.BINARY if self.binary_design else GRB.CONTINUOUS
        self.z = self.model.addVars(J, lb=0.0, ub=1.0, vtype=z_type, name="z")
        self.f = self.model.addVars(
            [(k, a) for k in range(K) for a in range(A)],
            lb=0.0,
            ub={(k, a): instance.commodities[k].demand for k in range(K) for a in range(A)},
            name="f",
        )
        self.candidate_arcs = tuple(a for a, group in enumerate(instance.arc_to_group) if group >= 0)
        self.w = self.model.addVars(
            [(k, a) for k in range(K) for a in self.candidate_arcs],
            lb=0.0,
            ub={(k, a): instance.commodities[k].demand for k in range(K) for a in self.candidate_arcs},
            name="w",
        )

        self.model.addConstr(
            gp.quicksum(instance.group_costs[j] * self.z[j] for j in range(J)) <= instance.budget,
            name="budget",
        )
        for k, commodity in enumerate(instance.commodities):
            demand = commodity.demand
            for node in range(N):
                rhs = demand if node == commodity.origin else (-demand if node == commodity.destination else 0.0)
                self.model.addConstr(
                    gp.quicksum(self.f[k, a] for a in instance.outgoing[node])
                    - gp.quicksum(self.f[k, a] for a in instance.incoming[node])
                    == rhs,
                    name=f"flow_balance[{k},{node}]",
                )
            for a in self.candidate_arcs:
                group = instance.arc_to_group[a]
                self.model.addConstr(self.w[k, a] <= self.f[k, a], name=f"w_le_f[{k},{a}]")
                self.model.addConstr(
                    self.w[k, a] <= demand * self.z[group],
                    name=f"w_le_dz[{k},{a}]",
                )
                self.model.addConstr(
                    self.f[k, a] - self.w[k, a] <= demand * (1.0 - self.z[group]),
                    name=f"f_minus_w[{k},{a}]",
                )

        self.model.setObjective(
            gp.quicksum(
                instance.lengths[a] * self.f[k, a]
                for k in range(K)
                for a in range(A)
            )
            - gp.quicksum(
                instance.lengths[a] * self.w[k, a]
                for k in range(K)
                for a in self.candidate_arcs
            ),
            GRB.MINIMIZE,
        )
        self.model.update()

    def solve(self, fixed: Mapping[int, int], *, time_limit: float | None) -> _RelaxationResult:
        instance = self.instance
        if _fixed_cost(instance, fixed) > instance.budget + 1e-9:
            return _RelaxationResult("INFEASIBLE", message="Fixed built projects exceed the budget.")
        for j in range(instance.n_groups):
            if j in fixed:
                self.z[j].LB = float(fixed[j])
                self.z[j].UB = float(fixed[j])
            else:
                self.z[j].LB = 0.0
                self.z[j].UB = 1.0
        self.model.Params.TimeLimit = GRB.INFINITY if time_limit is None else max(0.0, float(time_limit))
        self.model.update()
        self.model.optimize()
        status = _status_name(self.model.Status)
        lower_bound = None
        try:
            bound = float(self.model.ObjBound)
            lower_bound = max(0.0, bound) if math.isfinite(bound) else None
        except (AttributeError, gp.GurobiError):
            lower_bound = None
        z_values = None
        if self.model.SolCount > 0:
            z_values = tuple(float(self.z[j].X) for j in range(instance.n_groups))
        if self.model.Status != GRB.OPTIMAL:
            return _RelaxationResult(
                status,
                lower_bound=lower_bound,
                z_values=z_values,
                message=f"HPR/SO node {self.model_kind} did not solve to optimality.",
            )
        return _RelaxationResult(
            "OPTIMAL",
            objective=float(self.model.ObjVal),
            lower_bound=lower_bound if lower_bound is not None else float(self.model.ObjVal),
            z_values=z_values,
        )


def _paper_like_merit_order(instance: BaselineInstance) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Rank projects by all-built flow per construction cost.

    The 2016 paper ranks candidate projects by link/project merit before
    branching.  The original merit terms use traffic-assignment quantities not
    available in this shortest-path model.  The closest link-based analogue is
    all-built cyclist flow per unit construction cost.  This affects search
    order and incumbent quality only; it is not used as a bound.
    """
    all_built = tuple(1 for _ in range(instance.n_groups))
    _, flows = exact_optimistic_follower(instance, all_built, return_arc_flows=True)
    assert flows is not None
    scores: list[float] = []
    for j, group in enumerate(instance.groups):
        flow = sum(float(flows[k, a]) for k in range(instance.n_commodities) for a in group)
        scores.append(flow / max(instance.group_costs[j], 1e-12))
    order = tuple(sorted(range(instance.n_groups), key=lambda j: (-scores[j], j)))
    return order, tuple(scores)


def _greedy_initial_design(instance: BaselineInstance, order: Sequence[int]) -> tuple[int, ...]:
    z = [0] * instance.n_groups
    cost = 0.0
    for j in order:
        if cost + instance.group_costs[j] <= instance.budget + 1e-9:
            z[j] = 1
            cost += instance.group_costs[j]
    return tuple(z)


def _evaluate_design(
    instance: BaselineInstance,
    z: tuple[int, ...],
    cache: dict[tuple[int, ...], float],
) -> float:
    if z not in cache:
        value, _ = exact_optimistic_follower(instance, z)
        cache[z] = float(value)
    return cache[z]


def _local_search_incumbent(
    instance: BaselineInstance,
    starts: Sequence[tuple[int, ...]],
    *,
    max_passes: int,
    cache: dict[tuple[int, ...], float],
    started: float,
    time_limit: float | None,
    reserve_seconds: float = 1.0,
) -> _Incumbent:
    """Deterministic add/drop/swap improvement for a strong initial UB."""
    best_z = min(starts, key=lambda z: _evaluate_design(instance, z, cache))
    best_value = cache[best_z]
    for _ in range(max_passes):
        allowance = remaining_time(started, time_limit)
        if allowance is not None and allowance <= reserve_seconds:
            break
        improved = False
        selected = [j for j, value in enumerate(best_z) if value]
        unselected = [j for j, value in enumerate(best_z) if not value]
        neighbors: list[tuple[int, ...]] = []
        # Add or drop one project.
        for j in unselected:
            candidate = list(best_z)
            candidate[j] = 1
            z = tuple(candidate)
            if instance.is_budget_feasible(z):
                neighbors.append(z)
        for j in selected:
            candidate = list(best_z)
            candidate[j] = 0
            neighbors.append(tuple(candidate))
        # Swap one built and one unbuilt project.
        for drop in selected:
            for add in unselected:
                candidate = list(best_z)
                candidate[drop] = 0
                candidate[add] = 1
                z = tuple(candidate)
                if instance.is_budget_feasible(z):
                    neighbors.append(z)
        # Deterministic order keeps the baseline reproducible.
        for z in sorted(set(neighbors)):
            allowance = remaining_time(started, time_limit)
            if allowance is not None and allowance <= reserve_seconds:
                break
            value = _evaluate_design(instance, z, cache)
            if value < best_value - 1e-6:
                best_value, best_z = value, z
                improved = True
        if not improved:
            break
    return _Incumbent(best_value, best_z)


def _lp_guided_completion(
    instance: BaselineInstance,
    fixed: Mapping[int, int],
    z_relaxation: Sequence[float] | None,
    order: Sequence[int],
) -> tuple[int, ...] | None:
    if _fixed_cost(instance, fixed) > instance.budget + 1e-9:
        return None
    z = [0] * instance.n_groups
    cost = 0.0
    for j, value in fixed.items():
        z[j] = int(value)
        if value == 1:
            cost += instance.group_costs[j]
    undecided = [j for j in range(instance.n_groups) if j not in fixed]
    if z_relaxation is None:
        undecided.sort(key=lambda j: order.index(j))
    else:
        rank = {j: index for index, j in enumerate(order)}
        undecided.sort(key=lambda j: (-float(z_relaxation[j]), rank[j], j))
    for j in undecided:
        if cost + instance.group_costs[j] <= instance.budget + 1e-9:
            z[j] = 1
            cost += instance.group_costs[j]
    return tuple(z)


def _choose_branch_group(
    fixed: Mapping[int, int],
    z_relaxation: Sequence[float] | None,
    order: Sequence[int],
    *,
    tolerance: float,
) -> int | None:
    undecided = [j for j in order if j not in fixed]
    if not undecided:
        return None
    if z_relaxation is None:
        return undecided[0]
    fractional = [
        j for j in undecided
        if tolerance < float(z_relaxation[j]) < 1.0 - tolerance
    ]
    if fractional:
        # Branch first on the most fractional LP decision.  Ties follow the
        # paper-like merit order.
        rank = {j: index for index, j in enumerate(order)}
        return max(fractional, key=lambda j: (min(z_relaxation[j], 1.0 - z_relaxation[j]), -rank[j]))
    return undecided[0]


def _method_name() -> str:
    return "Bagloee-Sarvi-Wallace-2016-inspired branch-and-bound"


def solve_bagloee2016_inspired_branch_and_bound(
    instance: BaselineInstance,
    *,
    time_limit: float | None = None,
    verbose: bool = False,
    tolerance: float = 1e-6,
    node_time_limit: float | None = 2.0,
    local_search_passes: int = 3,
    record_history: bool = False,
) -> SolveResult:
    """Solve the manuscript model with an explicit Bagloee-style B&B.

    This is exact for the manuscript model if the B&B tree is exhausted.  It is
    not an exact reproduction of the 2016 paper's MUE/SO traffic-assignment
    model; see module docstring and result metadata.
    """
    started = time.perf_counter()
    workspace = _HprSoWorkspace(instance, verbose=verbose, binary_design=True)
    order, scores = _paper_like_merit_order(instance)

    exact_cache: dict[tuple[int, ...], float] = {}
    no_build = tuple(0 for _ in range(instance.n_groups))
    greedy = _greedy_initial_design(instance, order)
    starts = [no_build]
    if greedy != no_build and instance.is_budget_feasible(greedy):
        starts.append(greedy)
    incumbent = _local_search_incumbent(
        instance,
        starts,
        max_passes=max(0, int(local_search_passes)),
        cache=exact_cache,
        started=started,
        time_limit=time_limit,
    )

    nodes_processed = 0
    lp_solves = 0
    pruned_by_bound = 0
    pruned_by_integrality = 0
    stack: list[_Node] = [_Node({}, 0.0)]
    history: list[dict[str, object]] = []

    def active_lower_bound(current: float | None = None) -> float:
        bounds = [node.inherited_lower_bound for node in stack]
        if current is not None:
            bounds.append(current)
        if not bounds:
            return incumbent.value
        return min(bounds)

    def append_history(lb: float | None = None, event: str = "node") -> None:
        if not record_history:
            return
        if lb is None:
            lb = active_lower_bound()
        history.append({
            "iteration": int(lp_solves),
            "nodes": int(nodes_processed),
            "time": time.perf_counter() - started,
            "UB": float(incumbent.value),
            "LB": float(lb),
            "event": event,
        })

    def metadata(extra: Mapping[str, object] | None = None) -> dict[str, object]:
        payload: dict[str, object] = {
            "paper": "Bagloee, Sarvi & Wallace (2016), Bicycle lane priority",
            "adaptation": (
                "Outer branch-and-bound and SO/HPR lower-bound logic adapted to "
                "the manuscript's optimistic shortest-path bilevel model."
            ),
            "not_reproduced": (
                "Original multiclass UE/MUE and nonlinear SO traffic assignment are "
                "not implemented because the TNTP benchmark data do not contain the "
                "paper's lane/capacity/bias inputs."
            ),
            "node_lower_bound": (
                "binary HPR/SO design relaxation with fixed partial design; "
                "drops follower shortest-path optimality but preserves project integrality"
            ),
            "upper_bound": "exact optimistic shortest-path evaluation by lexicographic Dijkstra",
            "branch_order_zero_based": list(order),
            "merit_scores": list(scores),
            "lp_solves": lp_solves,
            "exact_evaluations": len(exact_cache),
            "node_time_limit": node_time_limit,
            "local_search_passes": local_search_passes,
            "pruned_by_bound": pruned_by_bound,
            "pruned_by_integrality": pruned_by_integrality,
            "tree_nodes_remaining": len(stack),
        }
        if record_history:
            payload["history"] = history
        if extra:
            payload.update(extra)
        return payload

    while stack:
        allowance = remaining_time(started, time_limit)
        if allowance is not None and allowance <= 0.0:
            lb = active_lower_bound()
            append_history(lb, "time_limit")
            return SolveResult(
                method=_method_name(),
                status="TIME_LIMIT",
                objective=incumbent.value,
                z=incumbent.z,
                y=instance.lane_values(incumbent.z),
                runtime_seconds=time.perf_counter() - started,
                certified_optimal=False,
                formulation_exact=True,
                lower_bound=lb,
                upper_bound=incumbent.value,
                iterations=lp_solves,
                nodes=nodes_processed,
                message="Wall-clock limit reached before the B&B tree was exhausted.",
                metadata=metadata(),
            )

        node = stack.pop()
        nodes_processed += 1
        node_allowance = allowance
        if node_time_limit is not None:
            node_allowance = min(float(node_time_limit), allowance) if allowance is not None else float(node_time_limit)
        relaxation = workspace.solve(node.fixed, time_limit=node_allowance)
        lp_solves += 1
        if relaxation.status == "INFEASIBLE":
            append_history(active_lower_bound(), "infeasible")
            continue
        if relaxation.status not in {"OPTIMAL", "TIME_LIMIT"}:
            node_bound = node.inherited_lower_bound
            if relaxation.lower_bound is not None:
                node_bound = max(node_bound, float(relaxation.lower_bound))
            lb = active_lower_bound(node_bound)
            append_history(lb, relaxation.status.lower())
            return SolveResult(
                method=_method_name(),
                status=relaxation.status,
                objective=incumbent.value,
                z=incumbent.z,
                y=instance.lane_values(incumbent.z),
                runtime_seconds=time.perf_counter() - started,
                certified_optimal=False,
                formulation_exact=True,
                lower_bound=lb,
                upper_bound=incumbent.value,
                iterations=lp_solves,
                nodes=nodes_processed,
                message=relaxation.message,
                metadata=metadata(),
            )

        if relaxation.status == "OPTIMAL" and relaxation.objective is not None:
            node_lower_bound = max(node.inherited_lower_bound, float(relaxation.objective))
        else:
            node_lower_bound = node.inherited_lower_bound
            if relaxation.lower_bound is not None:
                node_lower_bound = max(node_lower_bound, float(relaxation.lower_bound))
            # If the global wall-clock is exhausted, return cleanly instead of
            # branching on a node whose relaxation stopped only because the
            # whole run is out of time.
            global_left = remaining_time(started, time_limit)
            if global_left is not None and global_left <= 0.0:
                lb = active_lower_bound(node_lower_bound)
                append_history(lb, "time_limit")
                return SolveResult(
                    method=_method_name(),
                    status="TIME_LIMIT",
                    objective=incumbent.value,
                    z=incumbent.z,
                    y=instance.lane_values(incumbent.z),
                    runtime_seconds=time.perf_counter() - started,
                    certified_optimal=False,
                    formulation_exact=True,
                    lower_bound=lb,
                    upper_bound=incumbent.value,
                    iterations=lp_solves,
                    nodes=nodes_processed,
                    message="Wall-clock limit reached during a node HPR/SO relaxation.",
                    metadata=metadata(),
                )
        if node_lower_bound >= incumbent.value - tolerance:
            pruned_by_bound += 1
            append_history(active_lower_bound(), "pruned_by_bound")
            continue

        completion = _lp_guided_completion(instance, node.fixed, relaxation.z_values, order)
        if completion is not None and instance.is_budget_feasible(completion):
            value = _evaluate_design(instance, completion, exact_cache)
            if value < incumbent.value - tolerance:
                incumbent = _Incumbent(value, completion)
            if node_lower_bound >= incumbent.value - tolerance:
                pruned_by_bound += 1
                append_history(active_lower_bound(), "pruned_by_bound")
                continue

        if relaxation.z_values is not None and _is_integral(relaxation.z_values, tolerance):
            candidate = _as_binary(relaxation.z_values)
            if instance.is_budget_feasible(candidate):
                value = _evaluate_design(instance, candidate, exact_cache)
                if value < incumbent.value - tolerance:
                    incumbent = _Incumbent(value, candidate)
                if value <= node_lower_bound + tolerance:
                    pruned_by_integrality += 1
                    append_history(active_lower_bound(), "pruned_by_integrality")
                    continue

        branch_group = _choose_branch_group(
            node.fixed,
            relaxation.z_values,
            order,
            tolerance=tolerance,
        )
        if branch_group is None:
            # Every design variable is fixed but the LP lower bound was not
            # tight enough.  Evaluate the leaf exactly and close it.
            leaf = tuple(int(node.fixed.get(j, 0)) for j in range(instance.n_groups))
            if instance.is_budget_feasible(leaf):
                value = _evaluate_design(instance, leaf, exact_cache)
                if value < incumbent.value - tolerance:
                    incumbent = _Incumbent(value, leaf)
            append_history(active_lower_bound(), "leaf")
            continue

        child_zero = dict(node.fixed)
        child_zero[branch_group] = 0
        child_one = dict(node.fixed)
        child_one[branch_group] = 1
        # LIFO stack: append zero first so the built branch is explored first,
        # matching the bicycle-priority intuition in the paper.
        stack.append(_Node(child_zero, node_lower_bound))
        if _fixed_cost(instance, child_one) <= instance.budget + 1e-9:
            stack.append(_Node(child_one, node_lower_bound))
        append_history(active_lower_bound(), "branch")

    elapsed = time.perf_counter() - started
    append_history(incumbent.value, "optimal")
    return SolveResult(
        method=_method_name(),
        status="OPTIMAL",
        objective=incumbent.value,
        z=incumbent.z,
        y=instance.lane_values(incumbent.z),
        runtime_seconds=elapsed,
        certified_optimal=True,
        formulation_exact=True,
        lower_bound=incumbent.value,
        upper_bound=incumbent.value,
        iterations=lp_solves,
        nodes=nodes_processed,
        metadata=metadata({"tree_exhausted": True}),
    )


__all__ = ["solve_bagloee2016_inspired_branch_and_bound"]
