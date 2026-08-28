"""Common data structures and exact evaluators for the baseline solvers.

The baseline implementations in this directory deliberately do not reuse the
legacy ``OPT_HPR`` model.  That model aggregates OD flows and is designed for
the paper's BPC algorithm.  The three baselines need a clean, auditable
link-based formulation with one flow vector per OD--class commodity.

Model convention (identical to the paper and to ``case_run.py``):

* one binary variable ``z_j`` builds all directed arcs in candidate group j;
* arcs outside a candidate group are fixed to zero;
* construction cost is the sum of the ``length`` values of the directed arcs
  in the group (therefore it exactly matches the current HPR code);
* a class-g cyclist sees ``T_a`` on an unbuilt arc and ``beta_g*T_a`` on a
  built arc;
* the leader minimizes unprotected exposure
  ``sum L_a f_ka (1-y_a)``;
* when several shortest paths have identical perceived time, the follower
  selects the least-exposure one (optimistic convention).

All arc travel times and all beta values must be strictly positive.  This is
important: it makes a shortest-path follower cycle-free, hence each
commodity's flow on every arc is bounded by its OD demand.  The bound is used
in the exact McCormick formulations in the other modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import math
import time
from pathlib import Path
from collections import defaultdict, deque
from typing import Iterable, Mapping, Sequence

import numpy as np


Arc = tuple[int, int]


@dataclass(frozen=True)
class Commodity:
    """One OD--user-class demand in the lower-level shortest-path problem."""

    origin: int
    destination: int
    demand: float
    beta: float
    class_id: int = 0


@dataclass(frozen=True)
class BaselineInstance:
    """Immutable input used by all three baseline solvers."""

    n_nodes: int
    arcs: tuple[Arc, ...]
    lengths: tuple[float, ...]
    travel_times: tuple[float, ...]
    groups: tuple[tuple[int, ...], ...]
    group_costs: tuple[float, ...]
    budget: float
    commodities: tuple[Commodity, ...]
    arc_to_group: tuple[int, ...]
    arc_index: Mapping[Arc, int]
    outgoing: tuple[tuple[int, ...], ...]
    incoming: tuple[tuple[int, ...], ...]

    @property
    def n_arcs(self) -> int:
        return len(self.arcs)

    @property
    def n_groups(self) -> int:
        return len(self.groups)

    @property
    def n_commodities(self) -> int:
        return len(self.commodities)

    def lane_values(self, z: Sequence[float], tolerance: float = 1e-8) -> tuple[int, ...]:
        """Return the y-vector induced by a binary group design."""
        if len(z) != self.n_groups:
            raise ValueError("z has the wrong number of candidate groups")
        values = []
        for group in self.arc_to_group:
            if group < 0:
                values.append(0)
            else:
                value = float(z[group])
                if abs(value) <= tolerance:
                    values.append(0)
                elif abs(value - 1.0) <= tolerance:
                    values.append(1)
                else:
                    raise ValueError("lane_values requires a binary group design")
        return tuple(values)

    def design_cost(self, z: Sequence[float]) -> float:
        return float(sum(c * float(v) for c, v in zip(self.group_costs, z)))

    def is_budget_feasible(self, z: Sequence[float], tolerance: float = 1e-8) -> bool:
        return self.design_cost(z) <= self.budget + tolerance


@dataclass
class SolveResult:
    """A solver result with a clear separation of formulation and certificate."""

    method: str
    status: str
    objective: float | None = None
    z: tuple[int, ...] | None = None
    y: tuple[int, ...] | None = None
    runtime_seconds: float = 0.0
    certified_optimal: bool = False
    formulation_exact: bool = True
    lower_bound: float | None = None
    upper_bound: float | None = None
    iterations: int | None = None
    nodes: int | None = None
    message: str = ""
    metadata: dict[str, object] = field(default_factory=dict)


def _validate_edges(edges: Iterable[tuple[int, int, float]], n_nodes: int | None = None):
    parsed = [(int(u), int(v), float(w)) for u, v, w in edges]
    if not parsed:
        raise ValueError("the network has no arcs")
    if any(w <= 0 for _, _, w in parsed):
        raise ValueError("all arc weights must be strictly positive")
    seen: set[Arc] = set()
    for u, v, _ in parsed:
        if u < 0 or v < 0:
            raise ValueError("node indices must be non-negative")
        if (u, v) in seen:
            raise ValueError(
                "parallel/duplicate arcs are not supported by this auditable baseline interface"
            )
        seen.add((u, v))
    inferred_n = max(max(u, v) for u, v, _ in parsed) + 1
    if n_nodes is None:
        n_nodes = inferred_n
    if n_nodes < inferred_n:
        raise ValueError("n_nodes is smaller than an arc endpoint")
    return parsed, int(n_nodes)


def make_instance(
    *,
    edges: Iterable[tuple[int, int, float]],
    demand_matrix: np.ndarray,
    groups: Sequence[Sequence[Arc | int]],
    budget: float,
    betas: Sequence[float] = (0.5,),
    class_shares: Sequence[float] | None = None,
    travel_times: Sequence[float] | None = None,
    n_nodes: int | None = None,
    demand_scale: float = 1.0,
) -> BaselineInstance:
    """Build a baseline instance from the repository's network convention.

    ``edges`` contains ``(tail, head, L_a)``.  By default ``T_a=L_a``, which
    is the convention used by the current HPR implementation.  Pass an
    explicit ``travel_times`` vector only when the manuscript model itself
    uses a different T field.

    Candidate groups may use node-pair arcs, e.g. ``[(0, 1), (1, 0)]``, or
    already-resolved arc indices.  Groups must be disjoint.  This is required
    by the manuscript's one-y-per-arc construction and avoids silently
    changing the meaning of an overlapping candidate corridor.
    """
    parsed_edges, n_nodes = _validate_edges(edges, n_nodes)
    arcs = tuple((u, v) for u, v, _ in parsed_edges)
    lengths = tuple(w for _, _, w in parsed_edges)
    if travel_times is None:
        travel_times_tuple = lengths
    else:
        travel_times_tuple = tuple(float(v) for v in travel_times)
        if len(travel_times_tuple) != len(arcs):
            raise ValueError("travel_times must have one value per edge")
        if any(v <= 0 for v in travel_times_tuple):
            raise ValueError("all travel times must be strictly positive")

    demand_matrix = np.asarray(demand_matrix, dtype=float)
    if demand_matrix.shape != (n_nodes, n_nodes):
        raise ValueError("demand_matrix shape must be (n_nodes, n_nodes)")
    if np.any(demand_matrix < 0):
        raise ValueError("OD demands must be non-negative")
    if demand_scale <= 0:
        raise ValueError("demand_scale must be strictly positive")

    betas = tuple(float(beta) for beta in betas)
    if not betas or any(beta <= 0 or beta > 1 for beta in betas):
        raise ValueError("every beta must satisfy 0 < beta <= 1")
    if class_shares is None:
        class_shares = tuple(1.0 / len(betas) for _ in betas)
    class_shares = tuple(float(share) for share in class_shares)
    if len(class_shares) != len(betas) or any(share < 0 for share in class_shares):
        raise ValueError("class_shares must be non-negative and match betas")
    if not math.isclose(sum(class_shares), 1.0, rel_tol=0.0, abs_tol=1e-10):
        raise ValueError("class_shares must sum to one")

    arc_index = {arc: a for a, arc in enumerate(arcs)}
    normalized_groups: list[tuple[int, ...]] = []
    used_arcs: set[int] = set()
    for j, raw_group in enumerate(groups):
        resolved: list[int] = []
        for entry in raw_group:
            if isinstance(entry, int):
                a = int(entry)
                if a < 0 or a >= len(arcs):
                    raise ValueError(f"group {j} contains invalid arc index {a}")
            else:
                arc = (int(entry[0]), int(entry[1]))
                if arc not in arc_index:
                    raise ValueError(f"group {j} contains an arc absent from the network: {arc}")
                a = arc_index[arc]
            resolved.append(a)
        if not resolved:
            raise ValueError(f"candidate group {j} is empty")
        if len(set(resolved)) != len(resolved):
            raise ValueError(f"candidate group {j} repeats an arc")
        overlap = used_arcs.intersection(resolved)
        if overlap:
            raise ValueError(
                f"candidate groups must be arc-disjoint; group {j} overlaps on {sorted(overlap)}"
            )
        used_arcs.update(resolved)
        normalized_groups.append(tuple(resolved))

    if budget < 0:
        raise ValueError("budget must be non-negative")
    arc_to_group = [-1] * len(arcs)
    group_costs: list[float] = []
    for j, group in enumerate(normalized_groups):
        for a in group:
            arc_to_group[a] = j
        group_costs.append(float(sum(lengths[a] for a in group)))

    outgoing: list[list[int]] = [[] for _ in range(n_nodes)]
    incoming: list[list[int]] = [[] for _ in range(n_nodes)]
    for a, (u, v) in enumerate(arcs):
        outgoing[u].append(a)
        incoming[v].append(a)

    commodities: list[Commodity] = []
    for r in range(n_nodes):
        for s in range(n_nodes):
            total = float(demand_matrix[r, s]) * float(demand_scale)
            if r == s or total <= 0:
                continue
            for class_id, (beta, share) in enumerate(zip(betas, class_shares)):
                d = total * share
                if d > 0:
                    commodities.append(Commodity(r, s, d, beta, class_id))
    if not commodities:
        raise ValueError("at least one positive OD demand is required")

    instance = BaselineInstance(
        n_nodes=n_nodes,
        arcs=arcs,
        lengths=lengths,
        travel_times=travel_times_tuple,
        groups=tuple(normalized_groups),
        group_costs=tuple(group_costs),
        budget=float(budget),
        commodities=tuple(commodities),
        arc_to_group=tuple(arc_to_group),
        arc_index=arc_index,
        outgoing=tuple(tuple(items) for items in outgoing),
        incoming=tuple(tuple(items) for items in incoming),
    )
    validate_reachability(instance)
    return instance


def validate_reachability(instance: BaselineInstance) -> None:
    """Fail early if an OD has no path in the underlying network."""
    reachable_cache: dict[int, set[int]] = {}
    for commodity in instance.commodities:
        if commodity.origin not in reachable_cache:
            reached = {commodity.origin}
            frontier = [commodity.origin]
            while frontier:
                node = frontier.pop()
                for a in instance.outgoing[node]:
                    nxt = instance.arcs[a][1]
                    if nxt not in reached:
                        reached.add(nxt)
                        frontier.append(nxt)
            reachable_cache[commodity.origin] = reached
        if commodity.destination not in reachable_cache[commodity.origin]:
            raise ValueError(
                f"commodity {commodity.origin}->{commodity.destination} has no network path"
            )


def exact_optimistic_follower(
    instance: BaselineInstance,
    z: Sequence[int | float],
    *,
    tolerance: float = 1e-9,
    return_arc_flows: bool = False,
) -> tuple[float, np.ndarray | None]:
    """Evaluate the follower exactly for a fixed binary design.

    It runs Dijkstra lexicographically on ``(perceived time, exposure)``.
    Thus every commodity selects a shortest perceived-time path and, among
    equal-time shortest paths, the one with minimum unprotected exposure.  The
    result is exactly the optimistic lower-level reaction required by the
    manuscript, up to floating-point comparison tolerance.
    """
    y = instance.lane_values(z, tolerance=tolerance)
    K, A = instance.n_commodities, instance.n_arcs
    flows = np.zeros((K, A), dtype=float) if return_arc_flows else None
    total_exposure = 0.0

    for k, commodity in enumerate(instance.commodities):
        dist_time = [math.inf] * instance.n_nodes
        dist_exposure = [math.inf] * instance.n_nodes
        prev_arc = [-1] * instance.n_nodes
        origin, destination = commodity.origin, commodity.destination
        dist_time[origin] = 0.0
        dist_exposure[origin] = 0.0
        queue: list[tuple[float, float, int]] = [(0.0, 0.0, origin)]

        while queue:
            current_time, current_exposure, node = heapq.heappop(queue)
            worse_time = current_time > dist_time[node] + tolerance
            same_time_worse_exposure = (
                abs(current_time - dist_time[node]) <= tolerance
                and current_exposure > dist_exposure[node] + tolerance
            )
            if worse_time or same_time_worse_exposure:
                continue
            if node == destination:
                break
            for a in instance.outgoing[node]:
                _, head = instance.arcs[a]
                arc_time = instance.travel_times[a] * (commodity.beta if y[a] else 1.0)
                arc_exposure = instance.lengths[a] * (1 - y[a])
                candidate_time = current_time + arc_time
                candidate_exposure = current_exposure + arc_exposure
                improves_time = candidate_time < dist_time[head] - tolerance
                same_time_better_exposure = (
                    abs(candidate_time - dist_time[head]) <= tolerance
                    and candidate_exposure < dist_exposure[head] - tolerance
                )
                if improves_time or same_time_better_exposure:
                    dist_time[head] = candidate_time
                    dist_exposure[head] = candidate_exposure
                    prev_arc[head] = a
                    heapq.heappush(queue, (candidate_time, candidate_exposure, head))

        if not math.isfinite(dist_time[destination]):
            raise RuntimeError(
                f"design unexpectedly disconnects {origin}->{destination}; no arc is removed in this model"
            )
        total_exposure += commodity.demand * dist_exposure[destination]
        if flows is not None:
            node = destination
            while node != origin:
                a = prev_arc[node]
                if a < 0:
                    raise RuntimeError("failed to reconstruct a shortest path")
                flows[k, a] += commodity.demand
                node = instance.arcs[a][0]

    return float(total_exposure), flows


def read_tntp_network(net_file: str | Path, *, use_free_flow_time: bool = False):
    """Read a standard TNTP network, preserving length and free-flow time.

    The current manuscript code uses the TNTP ``length`` column for both L
    and T.  Therefore callers should leave ``use_free_flow_time=False`` unless
    the manuscript and all compared methods are intentionally redefined.
    """
    parsed: list[tuple[int, int, float, float | None]] = []
    with Path(net_file).open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("~") or line.startswith("<"):
                continue
            fields = line.replace(";", " ").split()
            if len(fields) < 5:
                continue
            try:
                u, v = int(fields[0]) - 1, int(fields[1]) - 1
                length = float(fields[3])
            except ValueError:
                continue
            try:
                fft: float | None = float(fields[4])
            except ValueError:
                fft = None
            if length <= 0 or (fft is not None and fft <= 0):
                raise ValueError(f"TNTP file contains a non-positive length/time on {u + 1}->{v + 1}")
            if use_free_flow_time and fft is None:
                raise ValueError(
                    f"TNTP file has no numeric free-flow time on {u + 1}->{v + 1}; "
                    "cannot use --use-free-flow-time for this file"
                )
            parsed.append((u, v, length, fft))
    if not parsed:
        raise ValueError(f"no TNTP arcs found in {net_file}")
    edges = [(u, v, length) for u, v, length, _ in parsed]
    # The default T=L must reproduce read_.py.  In particular, Anaheim's
    # supplied file includes one valid tail/head/length row whose fifth field
    # is the text "free_flow_time"; legacy experiments retain that 914th arc.
    times = [float(fft) if use_free_flow_time else length for _, _, length, fft in parsed]
    return edges, times


def read_tntp_demand(trips_file: str | Path, n_nodes: int) -> np.ndarray:
    """Read a standard TNTP OD-demand file."""
    demand = np.zeros((n_nodes, n_nodes), dtype=float)
    current_origin: int | None = None
    with Path(trips_file).open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("~") or line.startswith("<"):
                continue
            lower = line.lower()
            if lower.startswith("origin"):
                fields = line.split()
                if len(fields) >= 2:
                    current_origin = int(fields[1]) - 1
                continue
            if current_origin is None:
                continue
            for segment in line.split(";"):
                if ":" not in segment:
                    continue
                destination_text, amount_text = segment.split(":", 1)
                destination = int(destination_text.strip()) - 1
                amount = float(amount_text.strip())
                if not (0 <= current_origin < n_nodes and 0 <= destination < n_nodes):
                    raise ValueError("TNTP demand endpoint lies outside the network")
                demand[current_origin, destination] = amount
    return demand


def generate_disjoint_bidirectional_groups(
    edges: Iterable[tuple[int, int, float]],
    *,
    group_count: int,
    links_per_group: int,
    seed: int,
) -> tuple[tuple[Arc, ...], ...]:
    """Generate reproducible, arc-disjoint physical candidate corridors.

    A corridor contains ``links_per_group`` adjacent undirected physical links
    and includes both directions whenever both directions exist.  It mirrors
    the group-generation rule in ``case_run.py`` but has no hidden global
    constants and supports any requested number of groups.
    """
    if group_count <= 0 or links_per_group <= 0:
        raise ValueError("group_count and links_per_group must be positive")
    edge_set = {(int(u), int(v)) for u, v, _ in edges}
    physical = sorted({tuple(sorted((u, v))) for u, v in edge_set if u != v and (v, u) in edge_set})
    adjacency: dict[int, list[tuple[int, int]]] = {}
    for link in physical:
        adjacency.setdefault(link[0], []).append(link)
        adjacency.setdefault(link[1], []).append(link)
    rng = np.random.default_rng(seed)

    # Randomized depth-first extension is adequate for the small candidate
    # sets used by baseline_run and remains reproducible for a given seed.
    for _attempt in range(1_000):
        used: set[tuple[int, int]] = set()
        groups: list[tuple[Arc, ...]] = []
        success = True
        for _ in range(group_count):
            starts = [link for link in physical if link not in used]
            rng.shuffle(starts)
            chosen_path: list[int] | None = None
            chosen_links: list[tuple[int, int]] | None = None
            for start in starts:
                orientations = [(start[0], start[1]), (start[1], start[0])]
                rng.shuffle(orientations)
                for first, second in orientations:
                    path = [first, second]
                    corridor_links = [start]
                    while len(corridor_links) < links_per_group:
                        candidates = [
                            link for link in adjacency.get(path[-1], [])
                            if link not in used and link not in corridor_links
                        ]
                        if not candidates:
                            break
                        link = candidates[int(rng.integers(len(candidates)))]
                        next_node = link[0] if link[1] == path[-1] else link[1]
                        corridor_links.append(link)
                        path.append(next_node)
                    if len(corridor_links) == links_per_group:
                        chosen_path, chosen_links = path, corridor_links
                        break
                if chosen_path is not None:
                    break
            if chosen_path is None or chosen_links is None:
                success = False
                break
            directed: list[Arc] = []
            for u, v in zip(chosen_path[:-1], chosen_path[1:]):
                directed.extend(((u, v), (v, u)))
            groups.append(tuple(directed))
            used.update(chosen_links)
        if success:
            return tuple(groups)
    raise RuntimeError(
        f"could not generate {group_count} disjoint {links_per_group}-link candidate groups for seed {seed}"
    )


def generate_legacy_connected_fixed_groups(
    edges: Iterable[tuple[int, int, float]],
    *,
    group_count: int,
    links_per_group: int,
    seed: int,
) -> tuple[tuple[Arc, ...], ...]:
    """Reproduce ``main_run.group_edges_connected_fixed`` exactly.

    This is the candidate-set generator that produced the legacy rows in
    ``results.csv`` such as ``SF,0.5,10100,1,20,1,...``.  It groups undirected
    arc pairs into connected groups, starting from the currently highest-cost
    unused unit and randomising BFS neighbours with NumPy's legacy global RNG.

    It exists separately from ``generate_disjoint_bidirectional_groups`` so a
    baseline comparison can deliberately use *the same candidate set* as the
    manuscript's BPC experiment, rather than merely a matching random seed.
    """
    if group_count <= 0 or links_per_group <= 0:
        raise ValueError("group_count and links_per_group must be positive")
    parsed_edges, _ = _validate_edges(edges)
    # Keep the insertion order of the input edges and of pair_map values.  It
    # matches SparseGraph.g.es in main_run.group_edges_connected_fixed.
    pair_map: dict[tuple[int, int], list[tuple[Arc, float]]] = defaultdict(list)
    for u, v, weight in parsed_edges:
        pair_map[(min(u, v), max(u, v))].append(((u, v), weight))
    units = []
    for arcs_with_weights in pair_map.values():
        arcs = [arc for arc, _ in arcs_with_weights]
        nodes = set(node for arc in arcs for node in arc)
        units.append({"arcs": arcs, "cost": sum(weight for _, weight in arcs_with_weights), "nodes": nodes})
    if len(units) < group_count * links_per_group:
        raise ValueError(
            f"cannot form {group_count} groups with {links_per_group} physical links each; "
            f"only {len(units)} physical-link units exist"
        )
    node_to_units: dict[int, list[int]] = defaultdict(list)
    for index, unit in enumerate(units):
        for node in unit["nodes"]:
            node_to_units[node].append(index)
    adjacency: dict[int, set[int]] = {index: set() for index in range(len(units))}
    for indexes in node_to_units.values():
        for first in indexes:
            for second in indexes:
                if first != second:
                    adjacency[first].add(second)

    # Deliberately use the legacy global NumPy RandomState API, exactly as the
    # old experiment does.  This makes its seed=10100 candidate groups
    # reproducible for an apples-to-apples baseline run.
    np.random.seed(seed)
    unassigned = set(range(len(units)))
    groups: list[tuple[Arc, ...]] = []
    for group_index in range(group_count):
        if len(unassigned) < links_per_group:
            raise ValueError(f"insufficient unassigned units for group {group_index + 1}")
        start = max(unassigned, key=lambda index: units[index]["cost"])
        queue: deque[int] = deque([start])
        visited = {start}
        while queue and len(visited) < links_per_group:
            current = queue.popleft()
            neighbours = list(adjacency[current])
            np.random.shuffle(neighbours)
            for neighbour in neighbours:
                if neighbour in unassigned and neighbour not in visited:
                    visited.add(neighbour)
                    queue.append(neighbour)
                    if len(visited) == links_per_group:
                        break
        if len(visited) != links_per_group:
            raise ValueError(
                f"legacy grouping failed at group {group_index + 1}: "
                f"found {len(visited)} of {links_per_group} connected units"
            )
        for index in visited:
            unassigned.discard(index)
        groups.append(tuple(arc for index in visited for arc in units[index]["arcs"]))
    return tuple(groups)


def remaining_time(start: float, time_limit: float | None) -> float | None:
    """Return the remaining wall-clock allowance, or None for an unlimited run."""
    if time_limit is None:
        return None
    return max(0.0, float(time_limit) - (time.perf_counter() - start))
