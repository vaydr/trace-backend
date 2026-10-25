"""Algorithm 1: simple-path BFS with clause-prefix ranking."""

from collections import defaultdict, deque

from .schema import EvidencePath, PathStep, SearchResult, Triple


def clause_agreement(predicates: list[str], clauses: list[list[str]]) -> float:
    """Return max_i LCP(predicates, clauses[i]) / (i + 1)."""
    score = 0.0
    for rank, clause in enumerate(clauses, 1):
        prefix = 0
        for actual, hinted in zip(predicates, clause, strict=False):
            if actual != hinted:
                break
            prefix += 1
        score = max(score, prefix / rank)
    return score


def search_paths(
    triples: list[Triple],
    topic: str,
    answer: str,
    clauses: list[list[str]],
    max_hops: int = 8,
    max_states: int = 200000,
) -> SearchResult:
    """Enumerate paths in both edge directions; mark capped searches incomplete."""
    if max_hops < 0 or max_states < 1:
        raise ValueError("Invalid search budget")
    adjacent_pairs = defaultdict(dict)
    for t in sorted({t.id: t for t in triples}.values(), key=lambda t: t.key):
        adjacent_pairs[t.subject].setdefault((t.object, t.predicate), []).append((t.id, False))
        if t.subject != t.object:
            adjacent_pairs[t.object].setdefault((t.subject, t.predicate), []).append((t.id, True))
    adjacency = defaultdict(list)
    for vertex, pairs in adjacent_pairs.items():
        for (neighbor, predicate), origins in sorted(pairs.items()):
            # F contains distinct (predicate, neighbor) pairs. For reciprocal
            # duplicates, retain a source edge by forward orientation, then ID.
            triple_id, reversed_ = min(origins, key=lambda origin: (origin[1], origin[0]))
            adjacency[vertex].append((neighbor, predicate, triple_id, reversed_))

    # Target distances are an admissible lower bound on remaining path length.
    distances = {answer: 0}
    pending = deque([answer])
    while pending:
        v = pending.popleft()
        if distances[v] >= max_hops:
            continue
        for u, *_ in adjacency[v]:
            if u not in distances:
                distances[u] = distances[v] + 1
                pending.append(u)
    if topic not in distances:
        return SearchResult(paths=[], complete=True, states_examined=0)

    queue = deque([([topic], [])])
    paths = []
    states = 0
    enqueued = 1
    limited = False
    while queue:
        nodes, steps = queue.popleft()
        states += 1
        vertex = nodes[-1]
        if vertex == answer:
            paths.append(
                EvidencePath(
                    nodes=nodes,
                    steps=steps,
                    hops=len(steps),
                    agreement=clause_agreement([s.predicate for s in steps], clauses),
                )
            )
            continue
        if len(steps) >= max_hops:
            continue
        for neighbor, predicate, triple_id, reversed_ in adjacency[vertex]:
            if (
                neighbor in nodes
                or len(steps) + 1 + distances.get(neighbor, max_hops + 1) > max_hops
            ):
                continue
            if enqueued >= max_states:
                limited = True
                continue
            step = PathStep(
                triple_id=triple_id,
                source=vertex,
                target=neighbor,
                predicate=predicate,
                reversed=reversed_,
            )
            queue.append((nodes + [neighbor], steps + [step]))
            enqueued += 1
    paths.sort(
        key=lambda p: (-p.agreement, p.hops, tuple((s.triple_id, s.reversed) for s in p.steps))
    )
    return SearchResult(
        paths=paths,
        complete=not limited,
        states_examined=states,
        reason="state_limit" if limited else None,
    )
