// Sokoban solver for make.fun sokoban-solver-01.
//
// Strategy: push-based search, decomposed per room. Every generated level is a
// 1-wide hallway with 1-4 rooms off it (3-4 boxes each); a box pushed out of a
// room can never come back, so each room is independent and the level optimum
// is the sum of the rooms' optima. Each room is solved push-optimally with a
// breadth-first search over pushes (a state is the boxes' squares plus the
// player's reachable region), pruned by dead squares (a box that can never be
// pushed onto any goal) and frozen 2x2 blocks.
//
// Fallback ladder (each rung only runs when the one above it gives up, so the
// common case stays fast): room BFS -> room A* -> room A* again with a doubled
// work budget -> whole-level A* -> whole-level A* with a doubled budget. The
// A* rungs use a min-cost box-to-goal assignment heuristic, so a level still
// gets a valid (if not optimal) answer instead of an empty one when a room
// proves too big for the BFS.
//
// Deterministic by construction: fixed search order, work-based budgets only,
// no clock, no randomness, no environment reads. Two runs write the same bytes.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Work budgets (state counts, never the clock): a room the generator keeps is
// solvable within 250k states by its own push search, so 300k leaves margin.
// Each A* rung gets one second chance at twice the budget: a room that only
// just overflows the first budget still gets solved instead of abandoned.
const BFS_MAX_STATES = 300_000;
const ASTAR_MAX_STATES = 300_000;
const WHOLE_MAX_STATES = 400_000;

type Dir = [number, number, string, string];
const DIRS: Dir[] = [
  [0, -1, 'u', 'U'],
  [0, 1, 'd', 'D'],
  [-1, 0, 'l', 'L'],
  [1, 0, 'r', 'R'],
];

interface Level {
  w: number;
  h: number;
  wall: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
}

interface Solution {
  pushes: number;
  moves: string;
}

// ── Section 1: parsing and board analysis ──

/** Parse XSB text (# wall, space floor, . goal, $ box, * box on goal, @ player). */
function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  // The outer frame is wall whatever it shows: nothing leaves the board.
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
    }
  }
  return { w, h, wall, goals, boxes: boxes.sort((a, b) => a - b), player };
}

/**
 * Live squares: squares from which a box alone can still be pushed onto some
 * goal. Found by pulling a box back from the goals: a box on x could have come
 * from y = x - d with the player standing at y - d. Anything else is dead:
 * a box there is lost, so a search never pushes onto it.
 */
function liveSquares(level: Level): Uint8Array {
  const { w, wall, goals } = level;
  const live = new Uint8Array(wall.length);
  const queue = [...goals];
  for (const g of goals) live[g] = 1;
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi]!;
    for (const [dx, dy] of DIRS) {
      const d = dy * w + dx;
      const y = x - d;
      if (!wall[y] && !wall[y - d] && !live[y]) {
        live[y] = 1;
        queue.push(y);
      }
    }
  }
  return live;
}

/**
 * Flood-fill the squares the player reaches from `from` with `occupied` in
 * the way, marking them in `seen` with `stamp`; returns the lowest square of
 * the region, which names the region in a state's key.
 */
function reach(
  level: Level,
  occupied: Uint8Array,
  from: number,
  seen: Int32Array,
  stamp: number,
  stack: Int32Array,
): number {
  const { w, wall } = level;
  let low = from;
  let n = 0;
  stack[n++] = from;
  seen[from] = stamp;
  while (n > 0) {
    const p = stack[--n]!;
    if (p < low) low = p;
    // Neighbours: up, down, left, right (bounds are walls by construction).
    const qs = [p - w, p + w, p - 1, p + 1];
    for (const q of qs) {
      if (wall[q] === 0 && occupied[q] === 0 && seen[q] !== stamp) {
        seen[q] = stamp;
        stack[n++] = q;
      }
    }
  }
  return low;
}

/**
 * Whether the box just pushed to `t` sits in a frozen 2x2 block of walls and
 * boxes with a box off its goal: no push can ever move any of those boxes.
 */
function frozenBlock(level: Level, occupied: Uint8Array, goal: Uint8Array, t: number): boolean {
  const { w, wall } = level;
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    const b = a + 1;
    const c = a + w;
    const d = c + 1;
    if (!(wall[a] || occupied[a]) || !(wall[b] || occupied[b]) || !(wall[c] || occupied[c]) || !(wall[d] || occupied[d])) continue;
    if ((occupied[a] && !goal[a]) || (occupied[b] && !goal[b]) || (occupied[c] && !goal[c]) || (occupied[d] && !goal[d])) return true;
  }
  return false;
}

/** A state's key: the region's lowest square and the boxes' squares, base `size`. */
function keyOf(boxes: number[], low: number, size: number): number {
  let k = low;
  for (const b of boxes) k = k * size + b;
  return k;
}

/** The shortest walk from `from` to `to` around walls and boxes (lower LURD). */
function walkTo(level: Level, occupied: Uint8Array, from: number, to: number): string {
  const { w, wall } = level;
  if (from === to) return '';
  const prev = new Int32Array(wall.length).fill(-1);
  const how = new Array<string>(wall.length);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi]!;
    if (p === to) break;
    for (const [dx, dy, letter] of DIRS) {
      const q = p + dy * w + dx;
      if (wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = letter;
      queue.push(q);
    }
  }
  if (prev[to] === -1) throw new Error('walkTo: push square out of reach');
  let s = '';
  for (let p = to; p !== from; p = prev[p]!) s = how[p]! + s;
  return s;
}

/** The square the player stands on after `moves` (case aside) from `from`. */
function endOf(w: number, from: number, moves: string): number {
  let p = from;
  for (const ch of moves.toLowerCase()) p += ch === 'u' ? -w : ch === 'd' ? w : ch === 'l' ? -1 : 1;
  return p;
}

function solved(boxes: number[], goal: Uint8Array): boolean {
  return boxes.every((b) => goal[b]);
}

/**
 * Rebuild the player's moves from a search's parent chain: for each push in
 * order, walk to the square behind the pushed box, then push. Shared by the
 * BFS and the A* — both record the same parent/box arrays — so one
 * implementation serves both searches and a fix here fixes both.
 */
function rebuildMoves(level: Level, statesBoxes: number[][], parent: number[], found: number): Solution {
  const { w } = level;
  const occupied = new Uint8Array(level.wall.length);
  const path: number[] = [];
  for (let s = found; s > 0; s = parent[s]!) path.push(s);
  path.reverse();
  let moves = '';
  let player = level.player;
  let boxes = level.boxes;
  for (const s of path) {
    const nextBoxes = statesBoxes[s]!;
    const from = boxes.find((b) => !nextBoxes.includes(b))!;
    const to = nextBoxes.find((b) => !boxes.includes(b))!;
    const d = to - from;
    const dir = DIRS.find(([dx, dy]) => dy * w + dx === d)!;
    for (const b of boxes) occupied[b] = 1;
    moves += walkTo(level, occupied, player, from - d);
    for (const b of boxes) occupied[b] = 0;
    moves += dir[3];
    player = from;
    boxes = nextBoxes;
  }
  return { pushes: path.length, moves };
}

// ── Section 2: the two push searches (BFS, then A*) ──

/**
 * Push-optimal solution by breadth-first search over pushes, at most
 * `maxStates` states. States are visited in order of pushes made, so the
 * first solved state found is optimal. Null when the budget runs out or the
 * level has no solution.
 */
function solvePushBFS(level: Level, maxStates: number): Solution | null {
  const { w, wall, goals } = level;
  const size = wall.length;
  const goal = new Uint8Array(size);
  for (const g of goals) goal[g] = 1;
  const live = liveSquares(level);
  if (level.boxes.some((b) => !live[b])) return null;
  if (level.boxes.length > 4) return null; // numeric keys stay exact to 5 cells
  const occupied = new Uint8Array(size);
  const seen = new Int32Array(size);
  const mark = new Int32Array(size);
  const stack = new Int32Array(size);
  let stamp = 0;

  const statesBoxes: number[][] = [level.boxes];
  const statesPlayer: number[] = [level.player];
  const parent: number[] = [-1];
  const index = new Map<number, number>();
  for (const b of level.boxes) occupied[b] = 1;
  index.set(keyOf(level.boxes, reach(level, occupied, level.player, seen, ++stamp, stack), size), 0);
  for (const b of level.boxes) occupied[b] = 0;

  let found = solved(level.boxes, goal) ? 0 : -1;
  for (let head = 0; found < 0 && head < statesBoxes.length; head++) {
    const boxes = statesBoxes[head]!;
    for (const b of boxes) occupied[b] = 1;
    reach(level, occupied, statesPlayer[head]!, seen, ++stamp, stack);
    const here = stamp;
    for (let bi = 0; bi < boxes.length && found < 0; bi++) {
      const b = boxes[bi]!;
      for (const [dx, dy] of DIRS) {
        const d = dy * w + dx;
        const t = b + d;
        // The player must be able to stand behind the box, and the target
        // must be open floor a box can still be saved from.
        if (seen[b - d] !== here || wall[t] || occupied[t] || !live[t]) continue;
        occupied[b] = 0;
        occupied[t] = 1;
        const frozen = frozenBlock(level, occupied, goal, t);
        let key = -1;
        let next: number[] | null = null;
        if (!frozen) {
          next = boxes.slice();
          next[bi] = t;
          next.sort((x, y) => x - y);
          key = keyOf(next, reach(level, occupied, b, mark, ++stamp, stack), size);
        }
        occupied[t] = 0;
        occupied[b] = 1;
        if (frozen || index.has(key)) continue;
        if (statesBoxes.length >= maxStates) {
          for (const x of boxes) occupied[x] = 0;
          return null;
        }
        index.set(key, statesBoxes.length);
        statesBoxes.push(next!);
        statesPlayer.push(b);
        parent.push(head);
        if (solved(next!, goal)) {
          found = statesBoxes.length - 1;
          break;
        }
      }
    }
    for (const x of boxes) occupied[x] = 0;
  }
  if (found < 0) return null;

  return rebuildMoves(level, statesBoxes, parent, found);
}

/**
 * Distances over open floor from every square to its nearest goal, and from
 * each square to each goal: the A* heuristic's raw material.
 */
function goalDistances(level: Level): { nearest: Int32Array; toGoal: Int32Array[] } {
  const { w, wall, goals } = level;
  const size = wall.length;
  const nearest = new Int32Array(size).fill(-1);
  const toGoal: Int32Array[] = goals.map(() => new Int32Array(size).fill(-1));
  goals.forEach((g, gi) => {
    const dist = toGoal[gi]!;
    const queue = [g];
    dist[g] = 0;
    for (let qi = 0; qi < queue.length; qi++) {
      const p = queue[qi]!;
      for (const q of [p - w, p + w, p - 1, p + 1]) {
        if (!wall[q] && dist[q] === -1) {
          dist[q] = dist[p]! + 1;
          queue.push(q);
        }
      }
    }
  });
  for (let i = 0; i < size; i++) {
    let best = -1;
    for (const dist of toGoal) {
      const d = dist[i]!;
      if (d >= 0 && (best < 0 || d < best)) best = d;
    }
    nearest[i] = best;
  }
  return { nearest, toGoal };
}

/**
 * Min-cost assignment of boxes to goals by floor distance (exact Hungarian
 * by permutation for up to 4 boxes, greedy beyond). Admissible: each push
 * moves one box one square, other boxes ignored.
 */
function assignmentHeuristic(boxes: number[], goals: number[], toGoal: Int32Array[]): number {
  const n = boxes.length;
  if (n === 0) return 0;
  if (n > 4) {
    // Greedy: repeatedly take the cheapest remaining box-goal pair.
    const used = new Set<number>();
    let total = 0;
    for (const b of boxes) {
      let best = Infinity;
      let bestG = -1;
      goals.forEach((g, gi) => {
        if (used.has(gi)) return;
        const d = toGoal[gi]![b]!;
        if (d >= 0 && d < best) {
          best = d;
          bestG = gi;
        }
      });
      if (bestG < 0) return Infinity;
      used.add(bestG);
      total += best;
    }
    return total;
  }
  // Exact: try every permutation of goal indices.
  const perm = goals.map((_, i) => i);
  let best = Infinity;
  const rec = (k: number, total: number): void => {
    if (total >= best) return;
    if (k === n) {
      best = total;
      return;
    }
    for (let i = k; i < n; i++) {
      [perm[k], perm[i]] = [perm[i]!, perm[k]!];
      const d = toGoal[perm[k]!]![boxes[k]!]!;
      if (d >= 0) rec(k + 1, total + d);
      [perm[k], perm[i]] = [perm[i]!, perm[k]!];
    }
  };
  rec(0, 0);
  return best;
}

/**
 * Push-based A*: same states and pruning as the BFS, but ordered by pushes
 * made plus the assignment heuristic, so it finds a (usually good, not
 * provably optimal) solution with far fewer states when the BFS budget runs
 * out. Null when the budget runs out or the level has no solution.
 */
function solvePushAStar(level: Level, maxStates: number): Solution | null {
  const { w, wall, goals } = level;
  const size = wall.length;
  const goal = new Uint8Array(size);
  for (const g of goals) goal[g] = 1;
  const live = liveSquares(level);
  if (level.boxes.some((b) => !live[b])) return null;
  const { toGoal } = goalDistances(level);
  const occupied = new Uint8Array(size);
  const seen = new Int32Array(size);
  const mark = new Int32Array(size);
  const stack = new Int32Array(size);
  let stamp = 0;

  const statesBoxes: number[][] = [level.boxes];
  const statesPlayer: number[] = [level.player];
  const statesG: number[] = [0];
  const parent: number[] = [-1];
  const skey = (boxes: number[], low: number): string => low + ':' + boxes.join(',');
  const best = new Map<string, number>();
  for (const b of level.boxes) occupied[b] = 1;
  best.set(skey(level.boxes, reach(level, occupied, level.player, seen, ++stamp, stack)), 0);
  for (const b of level.boxes) occupied[b] = 0;

  // Binary heap on [f, h, sequence, state]: lowest f first, ties by lower h,
  // then by insertion order, so the run is deterministic.
  const heap: [number, number, number, number][] = [];
  let seq = 0;
  const h0 = assignmentHeuristic(level.boxes, goals, toGoal);
  const less = (a: [number, number, number, number], b: [number, number, number, number]): boolean =>
    a[0] < b[0] || (a[0] === b[0] && (a[1] < b[1] || (a[1] === b[1] && a[2] < b[2])));
  const heapPush = (e: [number, number, number, number]): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i]!, heap[p]!)) break;
      [heap[i], heap[p]] = [heap[p]!, heap[i]!];
      i = p;
    }
  };
  const heapPop = (): [number, number, number, number] => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1]!, heap[c]!)) c++;
        if (!less(heap[c]!, heap[i]!)) break;
        [heap[i], heap[c]] = [heap[c]!, heap[i]!];
        i = c;
      }
    }
    return top;
  };
  heapPush([h0, h0, seq++, 0]);

  let found = solved(level.boxes, goal) ? 0 : -1;
  while (heap.length > 0 && found < 0) {
    const [, , , s] = heapPop();
    const boxes = statesBoxes[s]!;
    const g = statesG[s]!;
    for (const b of boxes) occupied[b] = 1;
    const low = reach(level, occupied, statesPlayer[s]!, seen, ++stamp, stack);
    const here = stamp;
    if (best.get(skey(boxes, low)) !== s) {
      for (const x of boxes) occupied[x] = 0;
      continue;
    }
    for (let bi = 0; bi < boxes.length && found < 0; bi++) {
      const b = boxes[bi]!;
      for (const [dx, dy] of DIRS) {
        const d = dy * w + dx;
        const t = b + d;
        if (seen[b - d] !== here || wall[t] || occupied[t] || !live[t]) continue;
        occupied[b] = 0;
        occupied[t] = 1;
        const frozen = frozenBlock(level, occupied, goal, t);
        let key = '';
        let next: number[] | null = null;
        let h = 0;
        if (!frozen) {
          next = boxes.slice();
          next[bi] = t;
          next.sort((x, y) => x - y);
          key = skey(next, reach(level, occupied, b, mark, ++stamp, stack));
          h = assignmentHeuristic(next, goals, toGoal);
        }
        occupied[t] = 0;
        occupied[b] = 1;
        if (frozen || h === Infinity) continue;
        const prev = best.get(key);
        if (prev !== undefined && statesG[prev]! <= g + 1) continue;
        if (statesBoxes.length >= maxStates) {
          for (const x of boxes) occupied[x] = 0;
          return null;
        }
        const ns = statesBoxes.length;
        best.set(key, ns);
        statesBoxes.push(next!);
        statesPlayer.push(b);
        statesG.push(g + 1);
        parent.push(s);
        if (solved(next!, goal)) {
          found = ns;
          break;
        }
        heapPush([g + 1 + h, h, seq++, ns]);
      }
    }
    for (const x of boxes) occupied[x] = 0;
  }
  if (found < 0) return null;

  return rebuildMoves(level, statesBoxes, parent, found);
}

// ── Section 3: room decomposition and the fallback ladder ──

interface Room {
  squares: number[];
  boxes: number[];
  goals: number[];
  done: boolean;
}

/**
 * The level's rooms: cut the dead squares of the player's row (the hallway:
 * a box there reaches no goal) out of the floor; each remaining part holding
 * boxes is a room. Null when a box starts in the hallway, a goal falls
 * outside every room, or a part holds a different count of boxes and goals.
 */
function roomsOf(level: Level): Room[] | null {
  const { w, wall, goals, boxes, player } = level;
  const live = liveSquares(level);
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let i = row * w; i < (row + 1) * w; i++) if (!wall[i] && !live[i]) cut[i] = 1;
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const rooms: Room[] = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      for (const [dx, dy] of DIRS) {
        const q = squares[i]! + dy * w + dx;
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
    }
    rooms.push({ squares, boxes: [], goals: [], done: false });
  }
  for (const b of boxes) rooms[part[b]!]!.boxes.push(b);
  for (const g of goals) if (part[g]! >= 0) rooms[part[g]!]!.goals.push(g);
  if (goals.some((g) => part[g]! < 0) || rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

/**
 * Solve room by room: each room's push-optimal solution in turn, the other
 * rooms' boxes standing as walls (on their goals once their room is done),
 * the player walking on from where the last room left it. Null when a room
 * gives up even to the A* fallback.
 */
function solveByRooms(level: Level): Solution | null {
  const rooms = roomsOf(level);
  if (rooms === null) return null;
  if (rooms.some((r) => r.boxes.length > 4)) return null;
  let player = level.player;
  let pushes = 0;
  let moves = '';
  for (const room of rooms) {
    const wall = level.wall.slice();
    for (const other of rooms) {
      if (other === room) continue;
      for (const b of other.done ? other.goals : other.boxes) wall[b] = 1;
    }
    const sub: Level = {
      w: level.w,
      h: level.h,
      wall,
      goals: room.goals.slice().sort((a, b) => a - b),
      boxes: room.boxes.slice().sort((a, b) => a - b),
      player,
    };
    // Fallback ladder for one room: push-optimal BFS, then A*, then A* once
    // more with a doubled work budget (a room that only just overflows the
    // first A* budget still gets solved instead of failing the whole level).
    const solvedRoom =
      solvePushBFS(sub, BFS_MAX_STATES) ??
      solvePushAStar(sub, ASTAR_MAX_STATES) ??
      solvePushAStar(sub, ASTAR_MAX_STATES * 2);
    if (solvedRoom === null) return null;
    pushes += solvedRoom.pushes;
    moves += solvedRoom.moves;
    player = endOf(level.w, player, solvedRoom.moves);
    room.done = true;
  }
  return { pushes, moves };
}

/** Solve one level's text: room by room, else whole-level A*, else empty. */
// ── Section 4: driver ──
function solve(text: string): string {
  const level = parseLevel(text);
  if (level.player < 0 || level.boxes.length === 0) return '';
  const goal = new Uint8Array(level.wall.length);
  for (const g of level.goals) goal[g] = 1;
  if (solved(level.boxes, goal)) return '';
  const byRooms = solveByRooms(level);
  if (byRooms !== null) return byRooms.moves;
  // Last resort, same second-chance rule as the per-room ladder: whole-level
  // A*, then once more with a doubled work budget.
  const whole = solvePushAStar(level, WHOLE_MAX_STATES) ?? solvePushAStar(level, WHOLE_MAX_STATES * 2);
  return whole === null ? '' : whole.moves;
}

for (const id of readdirSync(I).sort()) {
  const moves = solve(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
