# Sokoban Solver

A push-optimal Sokoban solver written for make.fun's
`sokoban-solver-01-run-20261121` competition (Node 22, no dependencies, no
network, deterministic).

## How it works

Every generated level is a 1-wide hallway with 1–4 rooms off it, 3–4 boxes
per room. A box pushed out of a room through its door can never be pushed
back, so the rooms are independent problems and the level's push-optimal
count is the sum of the rooms' optima. The solver exploits that:

1. **Room decomposition** — the dead squares of the player's (hallway) row
   are cut out of the floor; each remaining component holding boxes is a room.
   Other rooms' boxes stand as walls while one room is solved.
2. **Push-optimal BFS per room** — breadth-first search over pushes. A state
   is the boxes' squares plus the player's reachable region (named by its
   lowest square); states are expanded in order of pushes made, so the first
   solved state is push-optimal.
3. **Deadlock pruning** — dead squares (a box that can never be pushed onto
   any goal, found by pulling a box back from the goals) and frozen 2×2
   blocks of walls and boxes.
4. **A\* fallback ladder** — if a room's BFS exhausts its work budget, a
   push-based A\* with a min-cost box→goal assignment heuristic (exact by
   permutation up to 4 boxes) still returns a valid solution instead of an
   empty one; each A\* rung gets one second chance at twice the budget, and
   a level that does not split into rooms falls back to whole-level A\* the
   same way.

Budgets are state counts, never the clock, and the search order is fixed, so
two runs write byte-identical outputs. The whole practice set solves in
about 1 second, far inside the 60-second wall budget.

## Results (practice set, kit scorer)

| instance | P\* | pushes | points |
|----------|----|--------|--------|
| i001 | 13 | 13 | 1000 |
| i002 | 16 | 16 | 1000 |
| i003 | 29 | 29 | 1000 |
| i004 | 47 | 47 | 1000 |
| i005 | 60 | 60 | 1000 |
| **total** | | | **1,000,000** |

The house baseline (plain-astar 1.0.0) scores 200,000 on the same set — it
solves only i001. On 6 freshly drawn random sets (30 instances) the solver
also reached the optimum on every instance (30,000/30,000), worst set
1.1 s.

## Run locally

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=<kit>/practice/input MAKEFUN_OUTPUT_DIR=out node main.ts
```

`main.ts` reads `/input/<id>/level.txt` (or `$MAKEFUN_INPUT_DIR`) and writes
`/output/<id>/solution.txt` (or `$MAKEFUN_OUTPUT_DIR`) for every instance.

## License

MIT — see [LICENSE](LICENSE).
