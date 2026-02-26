"""
logic_multi.py — Multi-bot swarm orchestrator
"""

from scipy.optimize import linear_sum_assignment
import numpy as np
import logic01
from logic01 import (
    _snap, _pos_eq, _close, _step,
    _push_plan, _best_route, _choose_order, _simulate_order,
    _emergency_step, SNAP,
)

MULTI_STATE = {
    'bot_states':    {},
    'assignments':   {},
    'task_pool':     [],
    'placed_blocks': set(),
    'initialised':   False,
}

STUCK_LIMIT    = 3    # ticks stuck before backoff
REASSIGN_LIMIT = 15   # ticks stuck before releasing task for reassignment


# ── push corridor helpers ─────────────────────────────────────────────────────

def _push_corridor_cells(b_x, b_y, g_x, g_y, grid, order):
    """All cells the block will pass through (excluding start) under given order."""
    cells = []
    cx, cy = b_x, b_y
    axes = ['X', 'Y'] if order == 'XY' else ['Y', 'X']
    for axis in axes:
        if axis == 'X':
            step = grid if g_x > cx else (-grid if g_x < cx else 0)
            while step and not _close(cx, g_x):
                cx += step
                cells.append((cx, cy))
        else:
            step = grid if g_y > cy else (-grid if g_y < cy else 0)
            while step and not _close(cy, g_y):
                cy += step
                cells.append((cx, cy))
    return cells


def _first_blocker_in_corridor(b_x, b_y, g_x, g_y, grid, order, block_positions, own_block_id):
    """
    Return (cell, blocking_block_id) for the FIRST cell in the push corridor
    occupied by a foreign block, or None if clear.
    """
    cells = _push_corridor_cells(b_x, b_y, g_x, g_y, grid, order)
    for cell in cells:
        for blk_id, (px, py) in block_positions.items():
            if blk_id == own_block_id:
                continue
            if abs(cell[0] - px) < SNAP and abs(cell[1] - py) < SNAP:
                return cell, blk_id
    return None


def _corridor_clear(b_x, b_y, g_x, g_y, grid, order, block_positions, own_block_id):
    return _first_blocker_in_corridor(
        b_x, b_y, g_x, g_y, grid, order, block_positions, own_block_id) is None


# ── obstacle-aware bot navigation ─────────────────────────────────────────────

def _seg_blocked(ax, ay, bx, by, obstacles, snap=5):
    for ox, oy in obstacles:
        if abs(oy - ay) < snap:
            lo, hi = min(ax, bx), max(ax, bx)
            if lo - snap < ox < hi + snap:
                return True
        if abs(ox - bx) < snap:
            lo, hi = min(ay, by), max(ay, by)
            if lo - snap < oy < hi + snap:
                return True
    return False

def _route_blocked(start, waypoints, obstacles, snap=5):
    prev = start
    for wp in waypoints:
        if _seg_blocked(prev[0], prev[1], wp[0], wp[1], obstacles, snap):
            return True
        prev = wp
    return False

def _best_route_obs(bx, by, sx, sy, b_x, b_y, push_axis, grid, facing, obstacles):
    from logic01 import (_direct_clear, _detour_wps, _score_waypoints, _cost)
    cands = []
    if _direct_clear(bx, by, sx, sy, b_x, b_y):
        wps = [(sx, sy)]
        blocked = _route_blocked((bx, by), wps, obstacles)
        s, t = _score_waypoints(wps, (bx, by), facing)
        cands.append((_cost(s, t), blocked, wps))
    for side in (+1, -1):
        wps = _detour_wps(bx, by, sx, sy, b_x, b_y, push_axis, grid, side)
        if wps:
            blocked = _route_blocked((bx, by), wps, obstacles)
            s, t = _score_waypoints(wps, (bx, by), facing)
            cands.append((_cost(s, t), blocked, wps))
    for ox, oy in obstacles:
        for side in (+1, -1):
            for axis in ('X', 'Y'):
                wps = _detour_wps(bx, by, sx, sy, ox, oy, axis, grid, side)
                if wps:
                    blocked = _route_blocked((bx, by), wps, obstacles)
                    s, t = _score_waypoints(wps, (bx, by), facing)
                    cands.append((_cost(s, t), blocked, wps))
    if not cands:
        return [(sx, sy)]
    cands.sort(key=lambda x: (x[1], x[0]))
    return cands[0][2]


def reset_logic():
    MULTI_STATE['bot_states'].clear()
    MULTI_STATE['assignments'].clear()
    MULTI_STATE['task_pool'].clear()
    MULTI_STATE['placed_blocks'].clear()
    MULTI_STATE['initialised'] = False


# ── helpers ───────────────────────────────────────────────────────────────────

def _manhattan(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def _task_cost(bot_pos, block_pos, goal_pos, facing, grid,
               other_block_positions=None):
    ok_x, sx, tx = _simulate_order(bot_pos, block_pos, goal_pos, "XY", facing, grid)
    ok_y, sy, ty = _simulate_order(bot_pos, block_pos, goal_pos, "YX", facing, grid)
    costs = []
    if ok_x: costs.append(sx + logic01.TURN_WEIGHT * tx)
    if ok_y: costs.append(sy + logic01.TURN_WEIGHT * ty)
    base = min(costs) if costs else 99999
    # Penalise if staging cell is occupied by another block
    if other_block_positions:
        for order in ('XY', 'YX'):
            _, _, stg_x, stg_y, _ = _push_plan(block_pos[0], block_pos[1],
                                                 goal_pos[0], goal_pos[1],
                                                 grid, order)
            for bpos in other_block_positions:
                if abs(stg_x - bpos[0]) < grid and abs(stg_y - bpos[1]) < grid:
                    base += 200   # staging penalty — prefer accessible blocks
                    break
    return base

def _is_real_task(assignment):
    return (assignment is not None
            and isinstance(assignment, dict)
            and 'block_id' in assignment)


# ── assignment: blocks → goals ────────────────────────────────────────────────

def _pair_blocks_to_goals(blocks, goals, grid):
    """
    Pair blocks to goals prioritizing farther goals first.
    Strategy: Use Hungarian algorithm but with modified costs that favor
    assigning blocks to farther goals.
    """
    if not blocks or not goals:
        return []
    
    n, m = len(blocks), len(goals)
    
    # Calculate all distances
    distances = [[_manhattan(blocks[i]['pos'], goals[j]['pos'])
                  for j in range(m)] for i in range(n)]
    
    # Find the maximum distance to use for inversion
    max_dist = max(max(row) for row in distances) if distances else 1
    
    # Create cost matrix that prioritizes farther goals
    # We invert the distances: farther goals get lower cost (higher priority)
    # Add small original distance to break ties sensibly
    cost = [[(max_dist - distances[i][j]) * 1000 + distances[i][j]
             for j in range(m)] for i in range(n)]
    
    row_ind, col_ind = linear_sum_assignment(np.array(cost))
    
    return [{
        'block_id':  blocks[r]['id'],
        'block_pos': blocks[r]['pos'],
        'goal_pos':  goals[c]['pos'],
    } for r, c in zip(row_ind, col_ind)]


# ── assignment: bots → tasks ──────────────────────────────────────────────────

def _assign_bots_to_tasks(free_bots, tasks, grid):
    if not free_bots or not tasks:
        return {}
    n, m = len(free_bots), len(tasks)
    cost = [[_task_cost(
                free_bots[i]['pos'],
                tasks[j]['block_pos'],
                tasks[j]['goal_pos'],
                tuple(free_bots[i]['facing']),
                grid)
             for j in range(m)] for i in range(n)]
    row_ind, col_ind = linear_sum_assignment(np.array(cost))
    return {free_bots[r]['id']: tasks[c] for r, c in zip(row_ind, col_ind)}


# ── per-bot state init ────────────────────────────────────────────────────────

def _init_bot_state(bot_pos, block_pos, goal_pos, facing, grid):
    order = _choose_order(bot_pos, block_pos, goal_pos, facing, grid)
    return {
        'phase':          'NAV',
        'facing':         list(facing),
        'waypoint_queue': [],
        'axis_order':     order,
        'last_goal':      goal_pos,
        'stuck_counter':  0,
        'last_pos':       bot_pos,
        # push detour: intermediate block-level sub-goals
        'push_subgoals':  [],   # [(gx,gy), ...] remaining block waypoints
    }

def _idle_state(facing=None):
    return {
        'phase':          'IDLE',
        'facing':         list(facing) if facing else [1, 0],
        'waypoint_queue': [],
        'axis_order':     'XY',
        'last_goal':      None,
        'stuck_counter':  0,
        'last_pos':       (0, 0),
        'push_subgoals':  [],
        'no_move_count':  0,
    }


# ── reassignment helper ───────────────────────────────────────────────────────

def _do_reassign(free_bot_ids, bot_pos_map, bot_states, task_pool, grid):
    if not free_bot_ids or not task_pool:
        return {}
    free_bots = []
    for bid in free_bot_ids:
        pos    = bot_pos_map.get(bid, (0, 0))
        facing = tuple(bot_states.get(bid, {}).get('facing', [1, 0]))
        free_bots.append({'id': bid, 'pos': pos, 'facing': facing})
    new_assigned = _assign_bots_to_tasks(free_bots, task_pool, grid)
    for bot_id, task in new_assigned.items():
        bot_p  = bot_pos_map.get(bot_id, (0, 0))
        facing = tuple(bot_states.get(bot_id, {}).get('facing', [1, 0]))
        bot_states[bot_id] = _init_bot_state(
            bot_p, task['block_pos'], task['goal_pos'], facing, grid)
    used_ids = {t['block_id'] for t in new_assigned.values()}
    to_remove = [t for t in task_pool if t['block_id'] in used_ids]
    for t in to_remove:
        task_pool.remove(t)
    return new_assigned


# ── push corridor rerouting ───────────────────────────────────────────────────

def _replan_push_if_blocked(s, b_x, b_y, g_x, g_y, grid,
                             block_positions, own_block_id):
    """
    Bot is at staging but cannot push (corridor or immediate cell blocked).
    Simple fix: switch to the OTHER axis so the bot walks around to the
    perpendicular side of the block and pushes from there.

    XY blocked → try YX (bot goes to top/bottom of block instead of left/right)
    YX blocked → try XY

    If the alt staging is also blocked by a block, try 4 escape sub-goals:
    push the block one step in each free direction to create space, then resume.
    """
    order = s.get('axis_order', 'XY')

    # Already on a sub-goal sequence — let it play out first
    if s.get('push_subgoals'):
        return False

    # ── Check current axis ────────────────────────────────────────────────────
    immediate_blocked = _first_blocker_in_corridor(
        b_x, b_y, g_x, g_y, grid, order, block_positions, own_block_id)
    if immediate_blocked is None:
        return False   # not actually blocked

    # ── Try the other axis ────────────────────────────────────────────────────
    alt = 'YX' if order == 'XY' else 'XY'
    _, _, alt_sx, alt_sy, _ = _push_plan(b_x, b_y, g_x, g_y, grid, alt)

    alt_stg_free = all(
        bid2 == own_block_id
        or abs(alt_sx - px) >= SNAP or abs(alt_sy - py) >= SNAP
        for bid2, (px, py) in block_positions.items()
    )
    alt_corr_clear = _corridor_clear(
        b_x, b_y, g_x, g_y, grid, alt, block_positions, own_block_id)

    if alt_stg_free and alt_corr_clear:
        # Switch axis: bot will now navigate to the perpendicular staging cell
        s['axis_order']     = alt
        s['waypoint_queue'] = []
        return True

    # ── Both axes blocked: push block one step in any free direction ──────────
    # Try all 4 cardinal directions; pick the first whose destination cell is free
    # and whose staging cell is also free.  That becomes a single sub-goal to
    # create space, after which normal planning resumes.
    for dx, dy in [(grid,0),(-grid,0),(0,grid),(0,-grid)]:
        dest_x, dest_y = b_x + dx, b_y + dy
        # destination cell must be free
        if any(bid2 != own_block_id
               and abs(dest_x - px) < SNAP and abs(dest_y - py) < SNAP
               for bid2, (px, py) in block_positions.items()):
            continue
        # staging for this micro-push must also be free
        micro_order = 'XY' if dx != 0 else 'YX'
        _, _, sx, sy, _ = _push_plan(b_x, b_y, dest_x, dest_y, grid, micro_order)
        if any(bid2 != own_block_id
               and abs(sx - px) < SNAP and abs(sy - py) < SNAP
               for bid2, (px, py) in block_positions.items()):
            continue
        # Good — push block one step in this direction, then resume real goal
        s['push_subgoals']  = [(dest_x, dest_y), (g_x, g_y)]
        s['waypoint_queue'] = []
        s['axis_order']     = micro_order
        return True

    return False   # fully surrounded — wait


def _compute_bot_move(bot_x, bot_y, b_x, b_y, g_x, g_y, s, grid,
                      obstacles=None, block_positions=None, own_block_id=None):
    """
    Returns (dx, dy, is_pushing).
    block_positions: dict block_id->(x,y) used for push corridor checks.
    """
    if obstacles       is None: obstacles       = set()
    if block_positions is None: block_positions = {}

    phase  = s['phase']
    facing = tuple(s['facing'])
    wq     = s['waypoint_queue']
    order  = s['axis_order']

    # Goal changed mid-task
    if s.get('last_goal') != (g_x, g_y):
        order = _choose_order((bot_x,bot_y),(b_x,b_y),(g_x,g_y), facing, grid)
        s['axis_order']    = order
        s['last_goal']     = (g_x, g_y)
        s['push_subgoals'] = []
        wq = []

    # ── Resolve effective sub-goal for this cycle ─────────────────────────────
    subgoals = s.get('push_subgoals', [])
    if subgoals:
        eff_gx, eff_gy = subgoals[0]
        # Advance if block has reached this sub-goal
        if _close(b_x, eff_gx) and _close(b_y, eff_gy):
            subgoals.pop(0)
            s['push_subgoals'] = subgoals
            wq = []
            # Rechoose axis order for next segment
            if subgoals:
                eff_gx, eff_gy = subgoals[0]
                order = _choose_order((bot_x,bot_y),(b_x,b_y),(eff_gx,eff_gy), facing, grid)
                s['axis_order'] = order
            else:
                eff_gx, eff_gy = g_x, g_y
                order = _choose_order((bot_x,bot_y),(b_x,b_y),(g_x,g_y), facing, grid)
                s['axis_order'] = order
    else:
        eff_gx, eff_gy = g_x, g_y

    push_dx, push_dy, stg_x, stg_y, push_axis = _push_plan(
        b_x, b_y, eff_gx, eff_gy, grid, order)

    move_dx = move_dy = 0
    pushing = False

    if phase == 'PUSH':
        phase = 'NAV'; wq = []

    if phase == 'NAV':
        # ── If staging cell is blocked by a foreign block, switch axis now ─────
        if own_block_id and block_positions and not subgoals:
            stg_blocked = any(
                bid2 != own_block_id
                and abs(stg_x - px) < SNAP and abs(stg_y - py) < SNAP
                for bid2, (px, py) in block_positions.items()
            )
            if stg_blocked:
                _replan_push_if_blocked(
                    s, b_x, b_y, eff_gx, eff_gy, grid, block_positions, own_block_id)
                order    = s['axis_order']
                wq       = s['waypoint_queue']
                subgoals = s.get('push_subgoals', [])
                eff_gx, eff_gy = subgoals[0] if subgoals else (g_x, g_y)
                push_dx, push_dy, stg_x, stg_y, push_axis = _push_plan(
                    b_x, b_y, eff_gx, eff_gy, grid, order)

        while wq and _pos_eq((bot_x, bot_y), wq[0]): wq.pop(0)

        if _pos_eq((bot_x, bot_y), (stg_x, stg_y)):
            phase = 'PUSH'
        else:
            if not wq:
                wq = _best_route_obs(bot_x, bot_y, stg_x, stg_y,
                                     b_x, b_y, push_axis, grid, facing, obstacles)
                while wq and _pos_eq((bot_x, bot_y), wq[0]): wq.pop(0)
            if wq:
                tgt_x, tgt_y = wq[0]
                dx = _step(bot_x, tgt_x, grid)
                dy = _step(bot_y, tgt_y, grid)
                if dx != 0:   move_dx = dx
                elif dy != 0: move_dy = dy
                if _close(bot_x+move_dx, b_x) and _close(bot_y+move_dy, b_y):
                    move_dx, move_dy = _emergency_step(
                        bot_x, bot_y, b_x, b_y, push_axis, grid)
                    wq = []

    if phase == 'PUSH':
        # Check the immediate next cell for the block
        next_cell = (b_x + push_dx, b_y + push_dy)
        immediate_blocked = any(
            abs(next_cell[0] - px) < SNAP and abs(next_cell[1] - py) < SNAP
            for bid2, (px, py) in block_positions.items()
            if bid2 != own_block_id
        )
        if immediate_blocked:
            # Go around: switch to perpendicular axis and re-navigate
            _replan_push_if_blocked(
                s, b_x, b_y, eff_gx, eff_gy, grid, block_positions, own_block_id)
            order    = s['axis_order']
            wq       = s['waypoint_queue']
            subgoals = s.get('push_subgoals', [])
            eff_gx, eff_gy = subgoals[0] if subgoals else (g_x, g_y)
            push_dx, push_dy, stg_x, stg_y, push_axis = _push_plan(
                b_x, b_y, eff_gx, eff_gy, grid, order)
            phase = 'NAV'; wq = []
        else:
            move_dx, move_dy = push_dx, push_dy
            pushing = True; phase = 'NAV'; wq = []

    if move_dx != 0 or move_dy != 0:
        facing = (move_dx // grid if move_dx else 0,
                  move_dy // grid if move_dy else 0)

    s['phase']          = phase
    s['facing']         = list(facing)
    s['waypoint_queue'] = wq
    s['axis_order']     = order
    return move_dx, move_dy, pushing


# ── collision resolution ──────────────────────────────────────────────────────

def _resolve_collisions(proposals, bot_id_order, all_bot_positions, all_block_positions, grid):
    bot_cells   = {}
    block_cells = {}
    for bid, (bx, by) in all_bot_positions.items():
        bot_cells[(bx, by)] = bid
    for blk_id, (bx, by) in all_block_positions.items():
        block_cells[(bx, by)] = blk_id

    pushing_bot_block = {}
    for bid, (cx, cy, dx, dy, pushing, blk_next) in proposals.items():
        if pushing:
            nx, ny = cx + dx, cy + dy
            blk_id = block_cells.get((nx, ny))
            if blk_id:
                pushing_bot_block[bid] = blk_id

    yield_moves = {}
    for bid, (cx, cy, dx, dy, pushing, blk_next) in proposals.items():
        if not pushing or not blk_next:
            continue
        bx2, by2 = blk_next
        blocker = bot_cells.get((bx2, by2))
        if blocker and blocker != bid:
            b_props = proposals.get(blocker, (0, 0, 0, 0, False, None))
            if b_props[2] == 0 and b_props[3] == 0:
                if dx != 0:
                    perp_options = [(0, grid), (0, -grid)]
                else:
                    perp_options = [(grid, 0), (-grid, 0)]
                for ydx, ydy in perp_options:
                    tnx, tny = bx2 + ydx, by2 + ydy
                    if (bot_cells.get((tnx, tny)) is None and
                            block_cells.get((tnx, tny)) is None):
                        yield_moves[blocker] = (ydx, ydy)
                        break

    priority = {bid: (0 if props[4] else 1) for bid, props in proposals.items()}
    ordered  = sorted(proposals.keys(),
                      key=lambda b: (priority[b],
                                     bot_id_order.index(b)
                                     if b in bot_id_order else 999))

    next_claims = {}
    allowed     = {bid: True for bid in proposals}
    overrides   = {}

    for bid in ordered:
        if bid in yield_moves and proposals[bid][2] == 0 and proposals[bid][3] == 0:
            ydx, ydy = yield_moves[bid]
            cx, cy   = proposals[bid][0], proposals[bid][1]
            tnx, tny = cx + ydx, cy + ydy
            if next_claims.get((tnx, tny)) is None:
                overrides[bid] = (ydx, ydy)
                next_claims[(tnx, tny)] = bid
            continue

        cx, cy, dx, dy, pushing, blk_next = proposals[bid]
        if dx == 0 and dy == 0:
            continue
        nx, ny = cx + dx, cy + dy

        bot_there = bot_cells.get((nx, ny))
        if bot_there is not None and bot_there != bid:
            allowed[bid] = False
            continue

        blk_there = block_cells.get((nx, ny))
        if blk_there is not None:
            own_block = pushing_bot_block.get(bid)
            if pushing and own_block == blk_there:
                if blk_next:
                    bx2, by2 = blk_next
                    if (bot_cells.get((bx2, by2)) is not None or
                            block_cells.get((bx2, by2)) is not None or
                            (bx2, by2) in next_claims):
                        allowed[bid] = False
                        continue
            else:
                allowed[bid] = False
                continue

        if (nx, ny) in next_claims:
            allowed[bid] = False
            continue

        next_claims[(nx, ny)] = bid
        if pushing and blk_next:
            bx2, by2 = blk_next
            next_claims[(bx2, by2)] = f"{bid}:block"

    for bid in proposals:
        if not allowed[bid] or bid in overrides:
            continue
        cx, cy, dx, dy, pushing, _ = proposals[bid]
        if pushing:
            continue
        nx, ny = cx + dx, cy + dy
        for other, (_, _, _, _, op, oblk) in proposals.items():
            if op and oblk and abs(nx - oblk[0]) < 5 and abs(ny - oblk[1]) < 5:
                allowed[bid] = False
                break

    result = {}
    for bid in proposals:
        if bid in overrides:
            result[bid] = (overrides[bid][0], overrides[bid][1], True)
        elif allowed[bid]:
            result[bid] = (proposals[bid][2], proposals[bid][3], True)
        else:
            result[bid] = (0, 0, False)
    return result


def compute_swarm_moves(state, grid_size, frozen_bots=None):
    if frozen_bots is None:
        frozen_bots = set()
    G  = grid_size
    ms = MULTI_STATE

    bots   = [{'id': b['id'],
               'pos': (_snap(b['pos'][0],G), _snap(b['pos'][1],G))}
              for b in state['bots']]
    blocks = [{'id': b['id'],
               'pos': (_snap(b['pos'][0],G), _snap(b['pos'][1],G))}
              for b in state['blocks']]
    goals  = [{'id': g['id'],
               'pos': (_snap(g['pos'][0],G), _snap(g['pos'][1],G))}
              for g in state['goals']]

    if not bots:
        return {'moves': {}, 'assignments': {}}

    block_by_id = {b['id']: b for b in blocks}
    bot_pos_map = {b['id']: b['pos'] for b in bots}
    bot_id_list = [b['id'] for b in bots]

    if not ms['initialised']:
        ms['initialised'] = True
        tasks = _pair_blocks_to_goals(blocks, goals, G)
        ms['task_pool'] = tasks[:]
        free_bots = [{'id': b['id'], 'pos': b['pos'], 'facing': (1,0)} for b in bots]
        new_assigned = _assign_bots_to_tasks(free_bots, ms['task_pool'], G)
        for bot_id, task in new_assigned.items():
            ms['assignments'][bot_id] = task
            ms['bot_states'][bot_id] = _init_bot_state(
                bot_pos_map[bot_id], task['block_pos'], task['goal_pos'], (1,0), G)
        used_ids = {t['block_id'] for t in new_assigned.values()}
        ms['task_pool'] = [t for t in ms['task_pool'] if t['block_id'] not in used_ids]
        for b in bots:
            if b['id'] not in ms['assignments']:
                ms['assignments'][b['id']] = None
                ms['bot_states'][b['id']] = _idle_state()

    for bot_id, task in ms['assignments'].items():
        if not _is_real_task(task): continue
        block = block_by_id.get(task['block_id'])
        if block:
            task['block_pos'] = block['pos']

    block_by_id = {b['id']: b for b in blocks}
    freed_this_tick = []
    for bot_id, task in list(ms['assignments'].items()):
        if not _is_real_task(task): continue
        block = block_by_id.get(task['block_id'])
        if block is None: continue
        if _pos_eq(block['pos'], task['goal_pos']):
            ms['placed_blocks'].add(task['block_id'])
            ms['assignments'][bot_id] = {'done': True}
            ms['bot_states'][bot_id]['phase'] = 'IDLE'
            freed_this_tick.append(bot_id)

    if freed_this_tick and ms['task_pool']:
        new_assigned = _do_reassign(
            freed_this_tick, bot_pos_map, ms['bot_states'], ms['task_pool'], G)
        for bot_id, task in new_assigned.items():
            ms['assignments'][bot_id] = task

    all_idle = [
        b['id'] for b in bots
        if ms['bot_states'].get(b['id'], {}).get('phase') == 'IDLE'
        and not _is_real_task(ms['assignments'].get(b['id']))
    ]
    if all_idle and ms['task_pool']:
        new_assigned = _do_reassign(
            all_idle, bot_pos_map, ms['bot_states'], ms['task_pool'], G)
        for bot_id, task in new_assigned.items():
            ms['assignments'][bot_id] = task

    all_block_positions = {b['id']: b['pos'] for b in blocks}
    proposals    = {}
    push_map     = {}
    state_before = {}
    state_after  = {}
    all_bot_positions = {b['id']: b['pos'] for b in bots}

    for bot in bots:
        bot_id       = bot['id']
        bot_x, bot_y = bot['pos']
        task         = ms['assignments'].get(bot_id)
        s            = ms['bot_states'].get(bot_id)

        if not _is_real_task(task) or s is None or s['phase'] == 'IDLE':
            proposals[bot_id] = (bot_x, bot_y, 0, 0, False, None)
            continue

        block = block_by_id.get(task['block_id'])
        if block is None:
            proposals[bot_id] = (bot_x, bot_y, 0, 0, False, None)
            continue

        b_x, b_y = block['pos']
        g_x, g_y = task['goal_pos']

        if bot_id in frozen_bots:
            snap_f = {k: (v[:] if isinstance(v, list) else v) for k, v in s.items()}
            snap_f['push_subgoals'] = list(s.get('push_subgoals', []))
            dx_f, dy_f, push_f = _compute_bot_move(
                bot_x, bot_y, b_x, b_y, g_x, g_y, s, G,
                obstacles=set(),
                block_positions=all_block_positions,
                own_block_id=task['block_id'])
            s.update(snap_f)
            s['waypoint_queue'] = snap_f['waypoint_queue'][:]
            s['push_subgoals']  = snap_f['push_subgoals'][:]
            blk_next_f = (b_x+dx_f, b_y+dy_f) if push_f else None
            proposals[bot_id] = (bot_x, bot_y, dx_f, dy_f, push_f, blk_next_f)
            if push_f: push_map[bot_id] = task['block_id']
            continue

        obstacles = set()
        for obid, (ox, oy) in all_bot_positions.items():
            if obid != bot_id:
                obstacles.add((ox, oy))
        for blk_id, (bx2, by2) in all_block_positions.items():
            if blk_id != task['block_id']:
                obstacles.add((bx2, by2))
        # Add own block so navigation paths never route through it
        own_blk_pos = all_block_positions.get(task['block_id'])
        if own_blk_pos:
            obstacles.add(own_blk_pos)

        snap = {k: (v[:] if isinstance(v, list) else v) for k, v in s.items()}
        snap['push_subgoals'] = list(s.get('push_subgoals', []))
        state_before[bot_id] = snap

        dx, dy, pushing = _compute_bot_move(
            bot_x, bot_y, b_x, b_y, g_x, g_y, s, G,
            obstacles=obstacles,
            block_positions=all_block_positions,
            own_block_id=task['block_id'])

        after = {k: (v[:] if isinstance(v, list) else v) for k, v in s.items()}
        after['push_subgoals'] = list(s.get('push_subgoals', []))
        state_after[bot_id] = after

        # Restore snapshot — committed only if collision resolver approves
        s.update(snap)
        s['waypoint_queue'] = snap['waypoint_queue'][:]
        s['push_subgoals']  = snap['push_subgoals'][:]

        blk_next = (b_x+dx, b_y+dy) if pushing else None
        proposals[bot_id] = (bot_x, bot_y, dx, dy, pushing, blk_next)
        if pushing: push_map[bot_id] = task['block_id']

    resolution = _resolve_collisions(
        proposals, bot_id_list, all_bot_positions, all_block_positions, G)

    moves           = {}
    assignments_out = {}

    for bot in bots:
        bot_id       = bot['id']
        bot_x, bot_y = bot['pos']
        task         = ms['assignments'].get(bot_id)
        s            = ms['bot_states'].get(bot_id, {})

        fin_dx, fin_dy, was_allowed = resolution.get(bot_id, (0, 0, True))

        blocked_by_block = False
        if not was_allowed and _is_real_task(task):
            block = block_by_id.get(task.get('block_id',''))
            if block:
                b_x2, b_y2 = block['pos']
                g_x2, g_y2 = task['goal_pos']
                plan = _push_plan(b_x2, b_y2, g_x2, g_y2, G,
                                  ms['bot_states'].get(bot_id,{}).get('axis_order','XY'))
                if plan[4]:
                    dest_x, dest_y = b_x2 + plan[0], b_y2 + plan[1]
                    if all_block_positions.get((dest_x, dest_y)) is not None:
                        blocked_by_block = True

        if was_allowed and bot_id in state_after:
            s.update(state_after[bot_id])
            s['waypoint_queue'] = state_after[bot_id]['waypoint_queue'][:]
            s['push_subgoals']  = state_after[bot_id].get('push_subgoals', [])
            s['stuck_counter']  = 0
            # Track zero-move ticks even when technically allowed
            if fin_dx == 0 and fin_dy == 0 and _is_real_task(task):
                s['no_move_count'] = s.get('no_move_count', 0) + 1
            else:
                s['no_move_count'] = 0
        elif bot_id in state_before:
            s.update(state_before[bot_id])
            s['waypoint_queue'] = state_before[bot_id]['waypoint_queue'][:]
            s['push_subgoals']  = state_before[bot_id].get('push_subgoals', [])
            if not blocked_by_block:
                s['stuck_counter'] = s.get('stuck_counter', 0) + 1
            s['no_move_count'] = s.get('no_move_count', 0) + 1

        if (s.get('stuck_counter', 0) >= STUCK_LIMIT
                and _is_real_task(task)
                and not blocked_by_block):
            b_x, b_y = task.get('block_pos', (bot_x, bot_y))
            # Try all 4 directions for backoff, prefer perpendicular to block
            if abs(b_x-bot_x) >= abs(b_y-bot_y):
                candidates = [(0, G), (0, -G), (G, 0), (-G, 0)]
            else:
                candidates = [(G, 0), (-G, 0), (0, G), (0, -G)]
            fin_dx, fin_dy = 0, 0
            for bdx, bdy in candidates:
                bnx, bny = bot_x + bdx, bot_y + bdy
                clear = all(abs(bnx-ox) >= 5 or abs(bny-oy) >= 5
                            for obid,(ox,oy) in all_bot_positions.items()
                            if obid != bot_id)
                clear = clear and all(abs(bnx-bkx) >= 5 or abs(bny-bky) >= 5
                                      for bkx,bky in all_block_positions.values())
                if clear:
                    fin_dx, fin_dy = bdx, bdy
                    break
            s['stuck_counter']  = 0
            s['waypoint_queue'] = []
            s['push_subgoals']  = []   # force full replan after backoff

        # ── Long-term stuck: look for a different accessible block ──────────────
        if (s.get('no_move_count', 0) >= REASSIGN_LIMIT
                and _is_real_task(task)):
            # Find current block's actual position
            cur_blk_pos = next((b['pos'] for b in blocks
                                if b['id'] == task['block_id']),
                               task.get('block_pos', (0,0)))
            # Return stuck task to pool with updated position
            ms['task_pool'].append({
                'block_id':  task['block_id'],
                'block_pos': cur_blk_pos,
                'goal_pos':  task['goal_pos'],
            })
            # Check if any pooled task has a clear corridor+staging right now
            all_blk_pos = {b['id']: b['pos'] for b in blocks}
            accessible = []
            for t in ms['task_pool']:
                if t['block_id'] == task['block_id']:
                    continue  # skip the one we just returned
                tb_pos = next((b['pos'] for b in blocks
                               if b['id'] == t['block_id']), t['block_pos'])
                tg_pos = t['goal_pos']
                others = {k: v for k, v in all_blk_pos.items()
                          if k != t['block_id']}
                for order in ('XY', 'YX'):
                    corr_ok = _corridor_clear(tb_pos[0], tb_pos[1],
                                              tg_pos[0], tg_pos[1],
                                              G, order, others, t['block_id'])
                    _, _, sx, sy, _ = _push_plan(tb_pos[0], tb_pos[1],
                                                  tg_pos[0], tg_pos[1], G, order)
                    stg_ok = all(k == t['block_id']
                                 or abs(sx-p[0]) >= SNAP or abs(sy-p[1]) >= SNAP
                                 for k, p in others.items())
                    if corr_ok and stg_ok:
                        accessible.append(t)
                        break
            if accessible:
                # Claim the closest accessible block
                best = min(accessible,
                           key=lambda t: _manhattan(
                               (bot_x, bot_y),
                               next((b['pos'] for b in blocks
                                     if b['id'] == t['block_id']), t['block_pos'])))
                ms['task_pool'] = [t for t in ms['task_pool']
                                   if t['block_id'] != best['block_id']]
                ms['assignments'][bot_id] = best
                ms['bot_states'][bot_id] = _idle_state(s.get('facing', [1,0]))
                ms['bot_states'][bot_id]['phase'] = 'NAV'
            else:
                ms['assignments'][bot_id] = None
                ms['bot_states'][bot_id] = _idle_state(s.get('facing', [1,0]))
            fin_dx, fin_dy = 0, 0

        block_id_out = push_map.get(bot_id) if was_allowed else None
        moves[bot_id] = (fin_dx, fin_dy, block_id_out)

        task = ms['assignments'].get(bot_id)   # re-read after possible release
        if task is None:
            assignments_out[bot_id] = None
        elif task.get('done'):
            assignments_out[bot_id] = {'done': True}
        elif _is_real_task(task):
            assignments_out[bot_id] = {
                'block_pos': task.get('block_pos', (0,0)),
                'goal_pos':  task['goal_pos'],
            }
        else:
            assignments_out[bot_id] = None

    # Reassign any newly-released bots
    newly_idle = [
        b['id'] for b in bots
        if ms['bot_states'].get(b['id'], {}).get('phase') == 'IDLE'
        and not _is_real_task(ms['assignments'].get(b['id']))
    ]
    if newly_idle and ms['task_pool']:
        new_assigned = _do_reassign(
            newly_idle, bot_pos_map, ms['bot_states'], ms['task_pool'], G)
        for bot_id, task in new_assigned.items():
            ms['assignments'][bot_id] = task
            assignments_out[bot_id] = {
                'block_pos': task.get('block_pos', (0,0)),
                'goal_pos':  task['goal_pos'],
            }

    return {'moves': moves, 'assignments': assignments_out}