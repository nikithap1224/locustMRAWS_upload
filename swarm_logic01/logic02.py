"""
logic01.py — Enhanced swarm push logic with obstacle classification
Now supports: multi-axis routing, obstacle-aware planning, cost-based path selection

Features:
- Obstacle classification (static immovable, static movable, dynamic)
- Cost-based path evaluation (steps + turns + wait + requests)
- Intelligent corridor-based routing when direct paths blocked
- Bot move requests for idle bots blocking path
- Dynamic obstacle prediction and wait strategies
- NAV-phase collision avoidance (bots don't walk through each other)
"""

import copy

BOT_STATES  = {}    # bot_id -> {phase, facing, waypoint_queue, axis_order, ...}
SNAP        = 5     # pixel tolerance
TURN_WEIGHT = 3     # cost of one 90° turn in equivalent steps
WAIT_TICK_COST = 1  # cost per tick spent waiting
# BOT_MOVE_REQUEST_COST is now dynamic based on moves needed

# ── Pending move requests: bot_id -> {'to': (x,y), 'countdown': int} ──
PENDING_MOVE_REQUESTS = {}


def reset_logic():
    BOT_STATES.clear()
    PENDING_MOVE_REQUESTS.clear()
    HELPER_PUSH_PLANS.clear()


# ── helpers ───────────────────────────────────────────────────────────────────

def _close(a, b):       return abs(a - b) < SNAP
def _pos_eq(p1, p2):    return _close(p1[0], p2[0]) and _close(p1[1], p2[1])
def _snap(v, g):        return round(v / g) * g

def _step(current, target, grid):
    d = target - current
    if abs(d) < SNAP: return 0
    return grid if d > 0 else -grid


# ── OBSTACLE CLASSIFICATION ──────────────────────────────────────────────────

def classify_obstacles(state, grid_size):
    """
    Classify all objects in the arena by obstacle type.
    
    CRITICAL RULE:
    - Only the FIRST spawned bot is active
    - Only the FIRST spawned block is the active block
    - Only the FIRST spawned goal is the target goal
    - ALL OTHER bots, blocks, goals are OBSTACLES (they don't move, don't act)
    
    Returns:
        Dictionary with classified obstacles:
        {
            'static_immovable': [{'pos': (x,y), 'block_id': '...'}, ...],
            'static_movable': [{'pos': (x,y), 'bot_id': '...'}, ...],
            'dynamic_predictable': [{'pos': (x,y), 'bot_id': '...', 'block_pos': (x,y), ...}, ...],
            'all_obstacle_positions': set([(x,y), ...])
        }
    """
    
    classification = {
        'static_immovable': [],
        'static_movable': [],
        'dynamic_predictable': [],
        'all_obstacle_positions': set()
    }
    
    if not state['bots'] or not state['blocks']:
        return classification
    
    # Classify all NON-FIRST bots as obstacles
    for i, bot in enumerate(state['bots']):
        if i == 0:  # Skip first bot (active)
            continue
        
        bot_pos = (_snap(bot['pos'][0], grid_size), 
                   _snap(bot['pos'][1], grid_size))
        bot_id = bot['id']
        
        classification['static_movable'].append({
            'type': 'STATIC_MOVABLE',
            'pos': bot_pos,
            'bot_id': bot_id,
            'facing': bot.get('facing', (1, 0))  # default right if not provided
        })
        classification['all_obstacle_positions'].add(bot_pos)
    
    # Classify all NON-FIRST blocks as static immovable obstacles
    for i, block in enumerate(state['blocks']):
        if i == 0:  # Skip first block (active block being pushed)
            continue
        
        block_pos = (_snap(block['pos'][0], grid_size), 
                    _snap(block['pos'][1], grid_size))
        block_id = block['id']
        
        classification['static_immovable'].append({
            'type': 'STATIC_IMMOVABLE',
            'pos': block_pos,
            'block_id': block_id
        })
        classification['all_obstacle_positions'].add(block_pos)
    
    return classification


# ── PATH GENERATION ───────────────────────────────────────────────────────────

def generate_l_path(start, goal, order):
    """Generate L-shaped path (XY or YX order)."""
    sx, sy = start
    gx, gy = goal
    
    if order == 'XY':
        if _close(sx, gx):
            return [start, goal]
        elif _close(sy, gy):
            return [start, goal]
        else:
            return [start, (gx, sy), goal]
    else:  # YX
        if _close(sy, gy):
            return [start, goal]
        elif _close(sx, gx):
            return [start, goal]
        else:
            return [start, (sx, gy), goal]


def get_cells_along_segment(p1, p2, grid):
    """Get all grid cells along a line segment."""
    cells = []
    x1, y1 = p1
    x2, y2 = p2
    
    # Horizontal segment
    if _close(y1, y2):
        min_x, max_x = min(x1, x2), max(x1, x2)
        x = min_x
        while x <= max_x + SNAP:
            cells.append((_snap(x, grid), _snap(y1, grid)))
            x += grid
    # Vertical segment
    elif _close(x1, x2):
        min_y, max_y = min(y1, y2), max(y1, y2)
        y = min_y
        while y <= max_y + SNAP:
            cells.append((_snap(x1, grid), _snap(y, grid)))
            y += grid
    
    return cells


def check_path_conflicts(path, obstacle_classification, grid):
    """
    Check which obstacles intersect with the path.
    
    CRITICAL: Checks EVERY cell along EVERY segment, not just waypoints!
    """
    conflicts = []
    
    for i in range(len(path) - 1):
        segment_cells = get_cells_along_segment(path[i], path[i+1], grid)
        
        for cell in segment_cells:
            if _pos_eq(cell, path[i]):
                continue
            
            for obs_type in ['static_immovable', 'static_movable', 'dynamic_predictable']:
                for obstacle in obstacle_classification[obs_type]:
                    if _pos_eq(cell, obstacle['pos']):
                        conflicts.append({
                            'obstacle': obstacle,
                            'waypoint_idx': i,
                            'cell': cell,
                            'segment': f"{path[i]} -> {path[i+1]}"
                        })
                    
                    if obs_type == 'dynamic_predictable' and 'block_pos' in obstacle:
                        if _pos_eq(cell, obstacle['block_pos']):
                            conflicts.append({
                                'obstacle': obstacle,
                                'waypoint_idx': i,
                                'cell': cell,
                                'segment': f"{path[i]} -> {path[i+1]}"
                            })
    
    return conflicts


def get_full_path_corridor(block_waypoints, grid):
    """
    Compute the SET of every cell the block will pass through on its planned path.
    
    ONLY includes the block's actual travel cells — NOT the staging positions.
    This keeps the corridor narrow so idle bots only need to move 1 cell
    to get out of the way (perpendicular to the path direction).
    
    Returns: set of (x, y) tuples
    """
    corridor = set()
    
    if not block_waypoints:
        return corridor
    
    for i in range(len(block_waypoints) - 1):
        cells = get_cells_along_segment(block_waypoints[i], block_waypoints[i+1], grid)
        for cell in cells:
            corridor.add(cell)
    
    return corridor


def find_clear_cell_outside_path(pos, obstacle_positions, path_corridor, grid, bot_facing=None):
    """
    Find a clear cell for an idle bot to move to that is:
    1. Not occupied by any obstacle
    2. COMPLETELY OUTSIDE the block's planned path corridor
    
    Searches outward in rings (Manhattan distance 1, 2, 3...) until
    a valid cell is found.
    
    Args:
        bot_facing: (dx, dy) tuple like (1,0) for right, (0,-1) for up, etc.
                    If provided, prefers cells the bot can reach WITHOUT turning.
    
    Returns: (x, y) or None if no cell found within search radius
    """
    x, y = pos
    
    # Search outward in rings up to 10 grid cells away
    for radius in range(1, 11):
        candidates = []
        for dx_cells in range(-radius, radius + 1):
            for dy_cells in range(-radius, radius + 1):
                if abs(dx_cells) + abs(dy_cells) != radius:
                    continue  # Only check cells at exactly this Manhattan distance
                cx = x + dx_cells * grid
                cy = y + dy_cells * grid
                
                # Bounds check
                if cx < 0 or cy < 0:
                    continue
                
                candidate = (cx, cy)
                
                # Must not be occupied
                if candidate in obstacle_positions:
                    continue
                
                # Must NOT be in the path corridor
                if candidate in path_corridor:
                    continue
                
                candidates.append(candidate)
        
        if candidates:
            def sort_key(c):
                dx_dir = 1 if c[0] > x else (-1 if c[0] < x else 0)
                dy_dir = 1 if c[1] > y else (-1 if c[1] < y else 0)
                
                # If we know the bot's facing direction, prefer moves
                # that match it (0 turns needed)
                if bot_facing:
                    fx, fy = bot_facing
                    # Exact same direction as facing → 0 turns
                    if (dx_dir, dy_dir) == (fx, fy):
                        return 0
                    # Axis-aligned but different direction (perpendicular) → 1 turn
                    on_same_x = _close(c[0], x)
                    on_same_y = _close(c[1], y)
                    if on_same_x or on_same_y:
                        return 1
                    # Diagonal / opposite → 2 turns
                    return 2
                else:
                    # No facing info — just prefer axis-aligned
                    on_same_x = _close(c[0], x)
                    on_same_y = _close(c[1], y)
                    if on_same_x or on_same_y:
                        return 0
                    return 1
            
            candidates.sort(key=sort_key)
            return candidates[0]
    
    return None  # Nothing found within 10-cell radius


def find_clear_adjacent_cell(pos, obstacle_positions, grid, preferred_away_from=None):
    """
    LEGACY fallback: Find a clear cell adjacent to pos.
    Used only when no path corridor information is available.
    
    If preferred_away_from is given, prefer cells that move AWAY from that point.
    """
    x, y = pos
    candidates = [
        (x + grid, y),
        (x - grid, y),
        (x, y + grid),
        (x, y - grid)
    ]
    
    if preferred_away_from is not None:
        ax, ay = preferred_away_from
        candidates.sort(key=lambda c: -(abs(c[0]-ax) + abs(c[1]-ay)))
    
    for candidate in candidates:
        if candidate[0] < 0 or candidate[1] < 0:
            continue
        if candidate not in obstacle_positions:
            return candidate
    
    # Diagonal fallback
    candidates = [
        (x + grid, y + grid),
        (x + grid, y - grid),
        (x - grid, y + grid),
        (x - grid, y - grid)
    ]
    for candidate in candidates:
        if candidate[0] < 0 or candidate[1] < 0:
            continue
        if candidate not in obstacle_positions:
            return candidate
    
    return None  # Completely surrounded


def calculate_bot_move_cost(from_pos, to_pos, grid):
    """Calculate cost for bot to move from one position to another."""
    if to_pos is None:
        return float('inf')
    cells = (abs(to_pos[0] - from_pos[0]) + abs(to_pos[1] - from_pos[1])) / grid
    return int(cells)


# ── COST CALCULATION ──────────────────────────────────────────────────────────

def calculate_path_cost(waypoints, bot_facing, grid, bot_pos=None):
    """
    Calculate total cost for a block path INCLUDING navigation overhead.
    
    This simulates the FULL bot journey:
    1. Nav from bot's current position to first staging position
    2. For each segment: push the block, then nav to next staging position
    
    Cost = total_steps + TURN_WEIGHT × total_turns
    
    Args:
        waypoints: block path waypoints
        bot_facing: (dx, dy) tuple of bot's current facing direction
        grid: grid size
        bot_pos: (x, y) of bot's current position. If None, only counts
                 push segment costs (legacy behavior).
    """
    if not waypoints or len(waypoints) < 2:
        return 0
    
    total_steps = 0
    total_turns = 0
    current_facing = bot_facing
    
    # If we have bot position, simulate the full journey
    if bot_pos is not None:
        current_bot_pos = bot_pos
    else:
        current_bot_pos = None
    
    for i in range(len(waypoints) - 1):
        block_from = waypoints[i]
        block_to = waypoints[i + 1]
        
        dx = block_to[0] - block_from[0]
        dy = block_to[1] - block_from[1]
        
        if abs(dx) < SNAP and abs(dy) < SNAP:
            continue
        
        # Determine push direction for this segment
        if abs(dx) > SNAP:
            push_dir = (1 if dx > 0 else -1, 0)
            # Staging position: opposite side of push
            staging = (block_from[0] - (grid if dx > 0 else -grid), block_from[1])
        else:
            push_dir = (0, 1 if dy > 0 else -1)
            staging = (block_from[0], block_from[1] - (grid if dy > 0 else -grid))
        
        # Count NAV cost: bot moves from current position to staging
        if current_bot_pos is not None:
            nav_dx = abs(staging[0] - current_bot_pos[0])
            nav_dy = abs(staging[1] - current_bot_pos[1])
            nav_steps = int((nav_dx + nav_dy) / grid)
            total_steps += nav_steps
            
            if nav_steps > 0:
                # Determine if bot needs to detour around the block.
                # After pushing along segment i, the bot is directly behind
                # block_from (the start of segment i+1). If the next staging
                # is on a different axis, the bot can't go straight through
                # the block — it must go AROUND, adding an extra turn.
                
                # Check if direct path from bot to staging passes through the block
                block_in_way = False
                if i > 0:
                    # The block is at block_from for segment i
                    bfx, bfy = block_from
                    bpx, bpy = current_bot_pos
                    spx, spy = staging
                    
                    # If bot and staging are on opposite sides of block on one axis,
                    # and aligned on the other, the block is in the way
                    if _close(bpy, bfy) and _close(spy, bfy):
                        # Same row — check if block is between bot and staging on X
                        if (bpx < bfx < spx) or (spx < bfx < bpx):
                            block_in_way = True
                    if _close(bpx, bfx) and _close(spx, bfx):
                        # Same col — check if block is between bot and staging on Y
                        if (bpy < bfy < spy) or (spy < bfy < bpy):
                            block_in_way = True
                
                if block_in_way:
                    # Detour around block: adds 2 extra steps and 2 extra turns
                    # (go perpendicular, go parallel, go back)
                    total_steps += 2
                    total_turns += 2
                    # Final facing after detour = push direction (we end at staging)
                    current_facing = push_dir
                else:
                    # Normal L-shaped or straight nav
                    if nav_dx > SNAP and nav_dy > SNAP:
                        # L-shaped: 2 directions, count turns for each
                        first_dir = (1 if staging[0] > current_bot_pos[0] else -1, 0)
                        second_dir = (0, 1 if staging[1] > current_bot_pos[1] else -1)
                        
                        # Try both orders, pick the one with fewer turns
                        # Order 1: X first then Y
                        turns_xy = (0 if first_dir == current_facing else 1) + (1)  # always turn for second leg
                        # Order 2: Y first then X
                        turns_yx = (0 if second_dir == current_facing else 1) + (1)  # always turn for second leg
                        
                        best_nav_turns = min(turns_xy, turns_yx)
                        total_turns += best_nav_turns
                        
                        # Final facing = direction of last leg
                        if turns_xy <= turns_yx:
                            current_facing = second_dir
                        else:
                            current_facing = first_dir
                    elif nav_dx > SNAP:
                        nav_dir = (1 if staging[0] > current_bot_pos[0] else -1, 0)
                        if nav_dir != current_facing:
                            total_turns += 1
                        current_facing = nav_dir
                    elif nav_dy > SNAP:
                        nav_dir = (0, 1 if staging[1] > current_bot_pos[1] else -1)
                        if nav_dir != current_facing:
                            total_turns += 1
                        current_facing = nav_dir
        
        # Count push turn: does bot need to turn to face push direction?
        if push_dir != current_facing:
            total_turns += 1
        
        # Count push steps
        segment_steps = int((abs(dx) + abs(dy)) / grid)
        total_steps += segment_steps
        
        # After pushing, bot is one cell behind the block's final position
        current_facing = push_dir
        if current_bot_pos is not None:
            # Bot ends at block_to minus one push step
            current_bot_pos = (block_to[0] - push_dir[0] * grid,
                              block_to[1] - push_dir[1] * grid)
    
    return total_steps + TURN_WEIGHT * total_turns


def calculate_cost_with_obstacles(path, conflicts, obstacle_classification, grid, bot_facing, bot_pos=None):
    """
    Calculate path cost considering how to handle each obstacle type.
    
    FIXED: Uses path corridor to ensure displaced bots don't land
    back on the block's travel path.
    
    Returns: (total_cost, requests_list, wait_periods_list)
    """
    base_cost = calculate_path_cost(path, bot_facing, grid, bot_pos=bot_pos)
    additional_cost = 0
    requests = []
    waits = []
    
    # Compute the full corridor of cells the block will traverse
    path_corridor = get_full_path_corridor(path, grid)
    
    for conflict in conflicts:
        obstacle = conflict['obstacle']
        obs_type = obstacle['type']
        
        if obs_type == 'STATIC_IMMOVABLE':
            return float('inf'), [], []
        
        elif obs_type == 'STATIC_MOVABLE':
            # Use path-aware displacement: find cell OUTSIDE the corridor
            target_cell = find_clear_cell_outside_path(
                obstacle['pos'], 
                obstacle_classification['all_obstacle_positions'],
                path_corridor,
                grid,
                bot_facing=obstacle.get('facing')
            )
            
            if target_cell is None:
                return float('inf'), [], []
            
            move_cost = calculate_bot_move_cost(obstacle['pos'], target_cell, grid)
            additional_cost += move_cost
            
            already_requested = any(r['bot_id'] == obstacle['bot_id'] for r in requests)
            if not already_requested:
                requests.append({
                    'type': 'move_bot',
                    'bot_id': obstacle['bot_id'],
                    'from': obstacle['pos'],
                    'to': target_cell,
                    'cost': move_cost
                })
        
        elif obs_type == 'DYNAMIC_PREDICTABLE':
            wait_ticks = 3
            additional_cost += wait_ticks * WAIT_TICK_COST
            waits.append({
                'at_waypoint': conflict['waypoint_idx'],
                'ticks': wait_ticks,
                'reason': f"waiting for dynamic obstacle {obstacle['bot_id']}"
            })
    
    return base_cost + additional_cost, requests, waits


# ── EXHAUSTIVE PATH GENERATION ───────────────────────────────────────────────

def generate_strategic_paths(start, goal, grid):
    """
    Generate path candidates covering ALL viable routes.
    """
    sx, sy = start
    gx, gy = goal
    
    all_paths = []
    
    # 1. Direct L-shaped paths
    all_paths.append(generate_l_path(start, goal, 'XY'))
    all_paths.append(generate_l_path(start, goal, 'YX'))
    
    # 2. Define EXPANDED search area
    margin = grid * 5
    min_x = max(0, min(sx, gx) - margin)
    min_y = max(0, min(sy, gy) - margin)
    max_x = min(grid * 20, max(sx, gx) + margin)
    max_y = min(grid * 20, max(sy, gy) + margin)
    
    # 3. Try EVERY horizontal corridor
    y = min_y
    while y <= max_y + SNAP:
        if not _close(y, sy) and not _close(y, gy):
            path = [(sx, sy), (sx, y), (gx, y), (gx, gy)]
            all_paths.append(simplify_path(path))
        y += grid
    
    # 4. Try EVERY vertical corridor
    x = min_x
    while x <= max_x + SNAP:
        if not _close(x, sx) and not _close(x, gx):
            path = [(sx, sy), (x, sy), (x, gy), (gx, gy)]
            all_paths.append(simplify_path(path))
        x += grid
    
    # 5. Combined corridors (sample every grid cell for thorough coverage)
    y = min_y
    step = grid  # Changed from grid*2 — ensures tight paths aren't missed
    while y <= max_y + SNAP:
        x = min_x
        while x <= max_x + SNAP:
            # Generate two path shapes through (x, y)
            # Shape 1: start → go vertical to y → go horizontal to x → go vertical to goal_y → goal
            path1 = [(sx, sy), (sx, y), (x, y), (x, gy), (gx, gy)]
            all_paths.append(simplify_path(path1))
            
            # Shape 2: start → go horizontal to x → go vertical to y → go horizontal to goal_x → goal  
            path2 = [(sx, sy), (x, sy), (x, y), (gx, y), (gx, gy)]
            all_paths.append(simplify_path(path2))
            
            x += step
        y += step
    
    # Remove duplicates
    unique_paths = []
    seen = set()
    for path in all_paths:
        path_tuple = tuple(tuple(p) for p in path)
        if path_tuple not in seen:
            seen.add(path_tuple)
            unique_paths.append(path)
    
    print(f"   Generated {len(unique_paths)} path candidates (including detours)")
    return unique_paths


def simplify_path(waypoints):
    """Remove redundant waypoints (collinear points)."""
    if len(waypoints) <= 2:
        return waypoints
    
    simplified = [waypoints[0]]
    for i in range(1, len(waypoints) - 1):
        prev = simplified[-1]
        curr = waypoints[i]
        next_wp = waypoints[i + 1]
        
        if _close(prev[0], curr[0]) and _close(curr[0], next_wp[0]):
            continue
        elif _close(prev[1], curr[1]) and _close(curr[1], next_wp[1]):
            continue
        else:
            simplified.append(curr)
    
    simplified.append(waypoints[-1])
    return simplified


# ── PATH EVALUATION WITH OBSTACLE ANALYSIS ────────────────────────────────────

def evaluate_path_with_obstacles(path, obstacle_classification, grid, bot_facing, bot_pos=None):
    """
    Evaluate a single path and calculate its total cost.
    
    Checks:
    1. Staging positions not blocked by immovable obstacles
    2. Staging positions with movable bots → generate move requests
    3. NAV reachability: bot can actually REACH each staging position
    4. Path conflicts (obstacles on the block's travel path)
    """
    
    # STEP 1: Check if staging positions are accessible AND reachable
    staging_blocked_immovable = []
    staging_requests = []
    
    # Compute corridor FIRST so we can use it for displacement
    path_corridor = get_full_path_corridor(path, grid)
    
    # Build obstacle set for nav reachability checks
    all_obstacles = obstacle_classification['all_obstacle_positions'].copy()
    
    # Simulate bot position through the path to check nav reachability
    sim_bot_pos = bot_pos  # None if not provided
    sim_bot_facing = bot_facing
    
    for i in range(len(path) - 1):
        current_block_pos = path[i]
        next_block_pos = path[i + 1]
        
        dx = next_block_pos[0] - current_block_pos[0]
        dy = next_block_pos[1] - current_block_pos[1]
        
        if abs(dx) < SNAP and abs(dy) < SNAP:
            continue
        
        # Staging position is OPPOSITE to push direction
        if abs(dx) > SNAP:
            push_dir = (1 if dx > 0 else -1, 0)
            staging_x = current_block_pos[0] - (grid if dx > 0 else -grid)
            staging_y = current_block_pos[1]
        else:
            push_dir = (0, 1 if dy > 0 else -1)
            staging_x = current_block_pos[0]
            staging_y = current_block_pos[1] - (grid if dy > 0 else -grid)
        
        staging_pos = (staging_x, staging_y)
        
        # Check if staging position is blocked by IMMOVABLE obstacle
        for obs in obstacle_classification.get('static_immovable', []):
            if _pos_eq(staging_pos, obs['pos']):
                staging_blocked_immovable.append({
                    'waypoint_idx': i,
                    'staging_pos': staging_pos,
                    'blocked_by': obs['pos'],
                    'segment': f"{current_block_pos} -> {next_block_pos}"
                })
                break
        
        # Check if staging position has a MOVABLE bot
        for obs in obstacle_classification.get('static_movable', []):
            if _pos_eq(staging_pos, obs['pos']):
                target_cell = find_clear_cell_outside_path(
                    obs['pos'],
                    obstacle_classification['all_obstacle_positions'],
                    path_corridor,
                    grid,
                    bot_facing=obs.get('facing')
                )
                
                if target_cell is None:
                    staging_blocked_immovable.append({
                        'waypoint_idx': i,
                        'staging_pos': staging_pos,
                        'blocked_by': obs['pos'],
                        'segment': f"{current_block_pos} -> {next_block_pos}"
                    })
                else:
                    already_requested = any(
                        r['bot_id'] == obs['bot_id'] for r in staging_requests
                    )
                    if not already_requested:
                        move_cost = calculate_bot_move_cost(obs['pos'], target_cell, grid)
                        staging_requests.append({
                            'type': 'move_bot',
                            'bot_id': obs['bot_id'],
                            'from': obs['pos'],
                            'to': target_cell,
                            'cost': move_cost,
                            'reason': 'blocking_staging_position'
                        })
                        print(f"    📢 Staging at {staging_pos} has idle bot {obs['bot_id']}"
                              f" → will ask it to move to {target_cell}")
        
        # ── NAV REACHABILITY CHECK ──────────────────────────────────────
        # Can the bot actually NAVIGATE from its current position to staging?
        # After the previous push, the bot is behind the block. Check if the
        # route to the next staging position is blocked by obstacles.
        if sim_bot_pos is not None and i > 0:
            # Check if route from sim_bot_pos to staging is blocked
            # CRITICAL: Include the block's position at this segment start,
            # because the bot can't walk through the active block!
            nav_obstacles = all_obstacles | {current_block_pos}
            
            push_axis = "X" if abs(dx) > SNAP else "Y"
            
            nav_reachable = False
            
            # Direct route
            if _direct_clear(sim_bot_pos[0], sim_bot_pos[1], 
                           staging_x, staging_y,
                           current_block_pos[0], current_block_pos[1]):
                if not _route_blocked_by_obstacles(sim_bot_pos, [staging_pos], nav_obstacles, grid):
                    nav_reachable = True
            
            # Detour routes (both sides)
            if not nav_reachable:
                for side in (+1, -1):
                    wps = _detour_wps(sim_bot_pos[0], sim_bot_pos[1],
                                     staging_x, staging_y,
                                     current_block_pos[0], current_block_pos[1],
                                     push_axis, grid, side)
                    if wps and not _route_blocked_by_obstacles(sim_bot_pos, wps, nav_obstacles, grid):
                        nav_reachable = True
                        break
            
            if not nav_reachable:
                # Bot can't reach staging — this path is INVALID
                staging_blocked_immovable.append({
                    'waypoint_idx': i,
                    'staging_pos': staging_pos,
                    'blocked_by': 'nav_unreachable',
                    'segment': f"{current_block_pos} -> {next_block_pos}"
                })
        
        # Update simulated bot position for next segment
        if sim_bot_pos is not None:
            # After pushing this segment, bot ends one cell behind block's end position
            seg_end = next_block_pos
            sim_bot_pos = (seg_end[0] - push_dir[0] * grid,
                          seg_end[1] - push_dir[1] * grid)
            sim_bot_facing = push_dir
    
    # If ANY staging position is blocked by IMMOVABLE obstacle → path INVALID
    if staging_blocked_immovable:
        return {
            'path': path,
            'cost': float('inf'),
            'conflicts': [],
            'requests': [],
            'waits': [],
            'valid': False,
            'staging_blocked': staging_blocked_immovable
        }
    
    # STEP 2: Check what conflicts exist along the path cells
    conflicts = check_path_conflicts(path, obstacle_classification, grid)
    
    # STEP 3: Calculate cost considering obstacles
    cost, requests, waits = calculate_cost_with_obstacles(
        path, conflicts, obstacle_classification, grid, bot_facing, bot_pos=bot_pos
    )
    
    # Merge staging requests into the main requests list
    for sr in staging_requests:
        already_in = any(r['bot_id'] == sr['bot_id'] for r in requests)
        if not already_in:
            requests.append(sr)
            cost += sr['cost']
    
    return {
        'path': path,
        'cost': cost,
        'conflicts': conflicts,
        'requests': requests,
        'waits': waits,
        'valid': cost < float('inf')
    }


# ── MAIN PATH PLANNING ────────────────────────────────────────────────────────

def find_optimal_block_path(block_pos, goal_pos, obstacle_classification, grid, bot_facing=(1,0), bot_pos=None):
    """
    Find the lowest-cost path for block.
    Now includes bot_pos for accurate NAV turn counting.
    """
    
    print(f"\n{'='*60}")
    print(f"PLANNING PATH: {block_pos} → {goal_pos}")
    print(f"{'='*60}")
    
    strategic_paths = generate_strategic_paths(block_pos, goal_pos, grid)
    print(f"Generated {len(strategic_paths)} strategic paths")
    
    evaluated_paths = []
    
    for i, path in enumerate(strategic_paths):
        evaluation = evaluate_path_with_obstacles(
            path, obstacle_classification, grid, bot_facing, bot_pos=bot_pos
        )
        evaluated_paths.append(evaluation)
        
        if evaluation['valid']:
            print(f"\nPath {i+1}: {path}")
            print(f"  Cost: {evaluation['cost']}")
            if evaluation['conflicts']:
                print(f"  ⚠️  Conflicts: {len(evaluation['conflicts'])}")
                segments = {}
                for conflict in evaluation['conflicts']:
                    seg = conflict.get('segment', 'unknown')
                    if seg not in segments:
                        segments[seg] = []
                    segments[seg].append(conflict)
                for seg, conflicts in segments.items():
                    print(f"    Segment {seg}:")
                    for conf in conflicts[:3]:
                        print(f"      - {conf['obstacle']['type']} at {conf['obstacle']['pos']}")
                    if len(conflicts) > 3:
                        print(f"      ... and {len(conflicts)-3} more")
            if evaluation['requests']:
                print(f"  📋 Requests: {len(evaluation['requests'])}")
                for req in evaluation['requests']:
                    reason = req.get('reason', 'blocking_path')
                    print(f"    - Move {req['bot_id']} to {req['to']} ({reason}, cost: {req['cost']})")
        else:
            print(f"\nPath {i+1}: INVALID")
            if evaluation.get('staging_blocked'):
                print(f"  ❌ STAGING POSITIONS BLOCKED:")
                for block in evaluation['staging_blocked']:
                    print(f"    Segment {block['segment']}:")
                    print(f"      Need staging at {block['staging_pos']} - BLOCKED at {block['blocked_by']}")
            else:
                print(f"  ❌ Blocked by immovable obstacles in path")
    
    valid_paths = [ep for ep in evaluated_paths if ep['valid']]
    
    if not valid_paths:
        print("\n❌ NO VALID PATH EXISTS!")
        return None
    
    valid_paths.sort(key=lambda x: x['cost'])
    best_path = valid_paths[0]
    
    print(f"\n{'='*60}")
    print(f"✅ SELECTED BEST PATH (Cost: {best_path['cost']})")
    print(f"Path: {best_path['path']}")
    if best_path['requests']:
        print(f"Move requests: {len(best_path['requests'])}")
        for req in best_path['requests']:
            print(f"  → {req['bot_id']}: {req['from']} → {req['to']}")
    print(f"{'='*60}\n")
    
    return {
        'waypoints': best_path['path'],
        'cost': best_path['cost'],
        'requests': best_path['requests'],
        'wait_periods': best_path['waits'],
        'path_details': {
            'total_candidates': len(strategic_paths),
            'valid_candidates': len(valid_paths),
            'conflicts': best_path['conflicts']
        }
    }


# ── PUSH PLAN (adapted for waypoint-based system) ─────────────────────────────

def _push_plan_to_waypoint(b_x, b_y, waypoint_x, waypoint_y, grid):
    """Return next push move toward a waypoint."""
    dx, dy = waypoint_x - b_x, waypoint_y - b_y
    
    if abs(dx) >= SNAP:
        p = grid if dx > 0 else -grid
        return p, 0, b_x - p, b_y, "X"
    elif abs(dy) >= SNAP:
        p = grid if dy > 0 else -grid
        return 0, p, b_x, b_y - p, "Y"
    else:
        return 0, 0, b_x, b_y, None


# ── navigation helpers ────────────────────────────────────────────────────────

def _direct_clear(bx, by, sx, sy, b_x, b_y):
    """Does X-first path bot→staging pass through the block?"""
    if not _close(bx, sx) and _close(by, b_y):
        lo, hi = min(bx, sx), max(bx, sx)
        if lo - SNAP <= b_x <= hi + SNAP: return False
    if not _close(by, sy) and _close(sx, b_x):
        lo, hi = min(by, sy), max(by, sy)
        if lo - SNAP <= b_y <= hi + SNAP: return False
    return True


def _detour_wps(bx, by, sx, sy, b_x, b_y, push_axis, grid, side):
    """3-waypoint detour route around the block on the given side."""
    wps = []
    def _add(p):
        prev = wps[-1] if wps else (bx, by)
        if not _pos_eq(p, prev): wps.append(p)
    if push_axis == "Y":
        dc = b_x + side * grid
        _add((dc, by)); _add((dc, sy)); _add((sx, sy))
    else:
        dr = b_y + side * grid
        _add((bx, dr)); _add((sx, dr)); _add((sx, sy))
    return wps


def _best_route(bx, by, sx, sy, b_x, b_y, push_axis, grid, facing, obstacles=None):
    """
    Return cheapest waypoint list from bot to staging.
    
    When two routes have equal cost, prefers the one where the bot
    arrives at staging ALREADY FACING the push direction — so it can
    push immediately without a rotation tick. This also means the bot
    travels in longer straight corridors (more practical/efficient).
    """
    if obstacles is None:
        obstacles = set()
    
    # Determine push direction from push_axis and block/staging positions
    if push_axis == "X":
        push_dir = (1 if b_x < sx + SNAP else -1, 0)  # staging is behind block
        # Actually: staging is opposite to push. If staging_x < block_x, push is RIGHT
        push_dir = (1 if sx < b_x else -1, 0)
    else:
        push_dir = (0, 1 if sy < b_y else -1)
    
    cands = []
    
    # Direct route (X-first)
    if _direct_clear(bx, by, sx, sy, b_x, b_y):
        wps = [(sx, sy)]
        blocked = _route_blocked_by_obstacles((bx, by), wps, obstacles, grid)
        s, t, final_facing = _score_waypoints(wps, (bx, by), facing)
        if final_facing != push_dir:
            t += 1
        aligned = 0 if final_facing == push_dir else 1
        cands.append((_cost(s, t), blocked, aligned, wps))
        
        # Also try Y-first variant
        if not _close(bx, sx) and not _close(by, sy):
            wps_yf = [(bx, sy), (sx, sy)]
            blocked_yf = _route_blocked_by_obstacles((bx, by), wps_yf, obstacles, grid)
            if not ((_close(bx, b_x) and 
                     min(by, sy) - SNAP <= b_y <= max(by, sy) + SNAP)):
                s_yf, t_yf, ff_yf = _score_waypoints(wps_yf, (bx, by), facing)
                if ff_yf != push_dir:
                    t_yf += 1
                aligned_yf = 0 if ff_yf == push_dir else 1
                cands.append((_cost(s_yf, t_yf), blocked_yf, aligned_yf, wps_yf))
    
    # Detour routes
    for side in (+1, -1):
        wps = _detour_wps(bx, by, sx, sy, b_x, b_y, push_axis, grid, side)
        if wps:
            blocked = _route_blocked_by_obstacles((bx, by), wps, obstacles, grid)
            s, t, final_facing = _score_waypoints(wps, (bx, by), facing)
            if final_facing != push_dir:
                t += 1
            aligned = 0 if final_facing == push_dir else 1
            cands.append((_cost(s, t), blocked, aligned, wps))
    
    if not cands:
        return [(sx, sy)]
    
    # Sort by: 1. blocked status, 2. cost, 3. alignment with push direction
    cands.sort(key=lambda x: (x[1], x[0], x[2]))
    return cands[0][3]


def _route_blocked_by_obstacles(start, waypoints, obstacles, grid):
    """
    Check if route from start through waypoints hits any obstacles.
    
    Simulates the actual X-first then Y movement the bot takes,
    not a diagonal line. For each waypoint, the bot goes horizontally
    first, then vertically — check ALL cells along both legs.
    
    IMPORTANT: Every cell along the route (including waypoint cells
    themselves) is checked against obstacles. Only the bot's current
    position (prev) is skipped since the bot is already standing there.
    """
    prev = start
    for wp in waypoints:
        px, py = prev
        wx, wy = wp
        
        # Leg 1: horizontal from (px, py) to (wx, py)
        if not _close(px, wx):
            cells_h = get_cells_along_segment((px, py), (wx, py), grid)
            for cell in cells_h:
                if _pos_eq(cell, prev):
                    continue
                if cell in obstacles:
                    return True
        
        # Leg 2: vertical from (wx, py) to (wx, wy)
        if not _close(py, wy):
            cells_v = get_cells_along_segment((wx, py), (wx, wy), grid)
            for cell in cells_v:
                if _pos_eq(cell, prev):
                    continue
                if cell in obstacles:
                    return True
        
        # Also check the waypoint cell itself — if it's purely horizontal
        # or purely vertical from prev, the waypoint is already in the
        # cells list above. But if prev == wp (no movement), ensure we
        # still check it.
        if wp in obstacles and not _pos_eq(wp, prev):
            return True
        
        prev = wp
    return False


def _score_waypoints(waypoints, start, facing):
    """Simulate X-first movement through waypoints; return (steps, turns, final_facing)."""
    pos, steps, turns, cur = start, 0, 0, facing
    for wp in waypoints:
        if not _close(pos[0], wp[0]):
            nd = (1 if wp[0] > pos[0] else -1, 0)
            if nd != cur: turns += 1; cur = nd
            steps += abs(round(wp[0] - pos[0]))
            pos = (wp[0], pos[1])
        if not _close(pos[1], wp[1]):
            nd = (0, 1 if wp[1] > pos[1] else -1)
            if nd != cur: turns += 1; cur = nd
            steps += abs(round(wp[1] - pos[1]))
            pos = (wp[0], wp[1])
    return steps, turns, cur


def _cost(steps, turns): return steps + TURN_WEIGHT * turns


def _estimate_ticks(waypoints, bot_facing, grid, bot_pos=None):
    """
    Estimate actual tick count for a block path (steps + 1 per turn).
    Unlike calculate_path_cost which weights turns at TURN_WEIGHT=3 for
    path selection, this counts real ticks for timing comparisons.
    """
    if not waypoints or len(waypoints) < 2:
        return 0
    total = 0
    cur_facing = bot_facing
    cur_bot = bot_pos
    
    for i in range(len(waypoints) - 1):
        bfrom = waypoints[i]
        bto   = waypoints[i + 1]
        dx = bto[0] - bfrom[0]
        dy = bto[1] - bfrom[1]
        if abs(dx) < SNAP and abs(dy) < SNAP:
            continue
        
        if abs(dx) > SNAP:
            pdir = (1 if dx > 0 else -1, 0)
            staging = (bfrom[0] - (grid if dx > 0 else -grid), bfrom[1])
        else:
            pdir = (0, 1 if dy > 0 else -1)
            staging = (bfrom[0], bfrom[1] - (grid if dy > 0 else -grid))
        
        # Nav to staging
        if cur_bot is not None:
            nav_d = int((abs(staging[0]-cur_bot[0]) + abs(staging[1]-cur_bot[1])) / grid)
            total += nav_d
            # Turns during nav (rough: 1 if L-shaped, 0 if straight)
            if nav_d > 0:
                sx = abs(staging[0]-cur_bot[0]) > SNAP
                sy = abs(staging[1]-cur_bot[1]) > SNAP
                if sx and sy:
                    total += 1  # L-shaped nav = 1 turn
                nav_end_dir = pdir  # approximate
                if nav_end_dir != cur_facing:
                    total += 1
                cur_facing = nav_end_dir
        
        # Turn to push
        if pdir != cur_facing:
            total += 1
        
        # Push
        seg_cells = int((abs(dx) + abs(dy)) / grid)
        total += seg_cells
        cur_facing = pdir
        if cur_bot is not None:
            cur_bot = (bto[0] - pdir[0]*grid, bto[1] - pdir[1]*grid)
    
    return total


def _emergency_step(bx, by, b_x, b_y, push_axis, grid):
    if push_axis == "X":
        dy = grid if by <= b_y else -grid
        if _close(by + dy, b_y): dy = -dy
        return 0, dy
    else:
        dx = grid if bx <= b_x else -grid
        if _close(bx + dx, b_x): dx = -dx
        return dx, 0


# ── REAL-TIME NAV COLLISION CHECK ─────────────────────────────────────────────
# FIX 2: Before the active bot takes a step, check if the target cell
# has an idle bot. If so, issue a move request and WAIT one tick.

def _check_nav_collision(next_x, next_y, obstacle_classification, grid, active_bot_pos, path_corridor=None):
    """
    Check if the active bot's next position collides with an idle bot.
    
    FIXED: Uses path_corridor to ensure displaced bot moves completely
    outside the block's planned travel path.
    
    Returns:
        None if no collision, or a dict:
        {
            'bot_id': '...',
            'bot_pos': (x, y),
            'move_to': (x, y),   # where idle bot should go
        }
    """
    if path_corridor is None:
        path_corridor = set()
    
    for obs in obstacle_classification.get('static_movable', []):
        if _pos_eq((next_x, next_y), obs['pos']):
            # Found an idle bot in the way!
            all_obstacles = obstacle_classification['all_obstacle_positions'].copy()
            all_obstacles.add(active_bot_pos)
            
            # Use path-aware displacement
            target = find_clear_cell_outside_path(
                obs['pos'], all_obstacles, path_corridor, grid,
                bot_facing=obs.get('facing')
            )
            
            if target is not None:
                return {
                    'bot_id': obs['bot_id'],
                    'bot_pos': obs['pos'],
                    'move_to': target
                }
            else:
                return {
                    'bot_id': obs['bot_id'],
                    'bot_pos': obs['pos'],
                    'move_to': None
                }
    
    return None


# ── Helper push plans: bot_id -> push instructions ──
HELPER_PUSH_PLANS = {}


def _plan_helper_push(state, block_waypoints, active_bot_pos, active_bot_facing,
                      block_pos, grid, obstacle_classification,
                      active_solo_cost=None):
    """
    Find the globally optimal (path, segment, helper_bot) combination.
    
    Evaluates ALL segments of ALL candidate paths (XY, YX, and the actual
    detour plan). For each segment S, computes timing model:
      Phase 1 (parallel): active does prefix segs 0..S-1, helper navigates
      Phase 2 (gated):    helper pushes seg S (waits for prefix if needed)
      Phase 3 (sequential): active does suffix segs S+1..N
    
    For execution: helper always operates on seg 0 of override_path.
    When best seg is S>0, helper is deferred — navigates in background
    while active does prefix, then activates when block reaches seg S.
    
    Returns: helper plan dict or None
    """
    if len(block_waypoints) < 2:
        return None
    
    goal_pos = block_waypoints[-1]
    
    # Build candidate paths: L-paths + actual plan
    path_xy = generate_l_path(block_pos, goal_pos, 'XY')
    path_yx = generate_l_path(block_pos, goal_pos, 'YX')
    
    candidate_paths = []
    seen = set()
    for label, p in [('XY', path_xy), ('YX', path_yx), ('PLAN', list(block_waypoints))]:
        key = tuple(tuple(w) for w in p)
        if key not in seen and len(p) >= 2:
            seen.add(key)
            candidate_paths.append((label, p))
    
    if active_solo_cost is None:
        active_solo_cost = _estimate_ticks(block_waypoints, active_bot_facing,
                                            grid, bot_pos=active_bot_pos)
    
    best = None
    best_total = active_solo_cost  # must beat solo
    
    for path_label, path in candidate_paths:
        num_segs = len(path) - 1
        
        for seg_idx in range(num_segs):
            seg_start = path[seg_idx]
            seg_end   = path[seg_idx + 1]
            seg_dx = seg_end[0] - seg_start[0]
            seg_dy = seg_end[1] - seg_start[1]
            if abs(seg_dx) < SNAP and abs(seg_dy) < SNAP:
                continue
            
            if abs(seg_dx) > SNAP:
                push_dir  = (1 if seg_dx > 0 else -1, 0)
                num_cells = int(abs(seg_dx) / grid)
            else:
                push_dir  = (0, 1 if seg_dy > 0 else -1)
                num_cells = int(abs(seg_dy) / grid)
            if num_cells < 1:
                continue
            
            helper_staging = (seg_start[0] - push_dir[0] * grid,
                              seg_start[1] - push_dir[1] * grid)
            
            # PREFIX tick count (active bot does segs 0..S-1)
            if seg_idx > 0:
                prefix_cost = _estimate_ticks(
                    path[:seg_idx + 1], active_bot_facing, grid,
                    bot_pos=active_bot_pos)
            else:
                prefix_cost = 0
            
            # SUFFIX cost (active bot does segs S+1..N)
            has_suffix = seg_idx + 1 < num_segs
            suf_staging = None
            if has_suffix:
                suf_s = path[seg_idx + 1]
                suf_e = path[seg_idx + 2]
                sdx   = suf_e[0] - suf_s[0]
                sdy   = suf_e[1] - suf_s[1]
                if abs(sdx) > SNAP:
                    suf_staging = (suf_s[0] - (grid if sdx > 0 else -grid), suf_s[1])
                elif abs(sdy) > SNAP:
                    suf_staging = (suf_s[0], suf_s[1] - (grid if sdy > 0 else -grid))
                else:
                    continue
                suffix_cost = _estimate_ticks(
                    path[seg_idx + 1:], push_dir, grid, bot_pos=suf_staging)
            else:
                suffix_cost = 0
            
            # Check each idle bot
            for idle_info in obstacle_classification.get('static_movable', []):
                idle_pos    = idle_info['pos']
                idle_id     = idle_info['bot_id']
                idle_facing = idle_info.get('facing', (1, 0))
                
                stg_obs = obstacle_classification['all_obstacle_positions'] - {idle_pos}
                if helper_staging in stg_obs:
                    continue
                
                h_dist = int((abs(idle_pos[0] - helper_staging[0]) +
                              abs(idle_pos[1] - helper_staging[1])) / grid)
                if h_dist > 6:
                    continue
                
                # Route check: X-first then Y-first
                if h_dist > 0:
                    route_ok = False
                    if not _route_blocked_by_obstacles(idle_pos, [helper_staging], stg_obs, grid):
                        route_ok = True
                    if not route_ok:
                        ix, iy = idle_pos
                        hsx, hsy = helper_staging
                        if not _close(ix, hsx) and not _close(iy, hsy):
                            yblk = False
                            for c in get_cells_along_segment((ix, iy), (ix, hsy), grid):
                                if not _pos_eq(c, idle_pos) and c in stg_obs:
                                    yblk = True; break
                            if not yblk:
                                for c in get_cells_along_segment((ix, hsy), (hsx, hsy), grid):
                                    if not _pos_eq(c, (ix, hsy)) and c in stg_obs:
                                        yblk = True; break
                            if not yblk:
                                route_ok = True
                    if not route_ok:
                        continue
                
                # Helper timing
                h_turns = 0
                if h_dist > 0:
                    if abs(idle_pos[0] - helper_staging[0]) > SNAP:
                        fd = (1 if helper_staging[0] > idle_pos[0] else -1, 0)
                    else:
                        fd = (0, 1 if helper_staging[1] > idle_pos[1] else -1)
                    if fd != idle_facing:
                        h_turns = 1
                elif push_dir != idle_facing:
                    h_turns = 1
                
                helper_ready = h_dist + h_turns
                push_start   = max(prefix_cost, helper_ready)
                helper_done  = push_start + num_cells
                # Penalty: after helper finishes, it may block active bot's suffix nav.
                # The longer the helper push, the more likely it's in the way.
                # Add a small penalty for helper segments > 2 cells.
                collision_penalty = max(0, num_cells - 2) * 2
                total_ticks  = helper_done + suffix_cost + collision_penalty
                
                if total_ticks >= best_total:
                    continue
                
                # Validate path (exclude helper from obstacles)
                excl_obs = {
                    'static_immovable': obstacle_classification['static_immovable'],
                    'static_movable': [o for o in obstacle_classification['static_movable']
                                      if o['bot_id'] != idle_id],
                    'dynamic_predictable': obstacle_classification['dynamic_predictable'],
                    'all_obstacle_positions': obstacle_classification['all_obstacle_positions'] - {idle_pos}
                }
                ev = evaluate_path_with_obstacles(path, excl_obs, grid,
                                                   active_bot_facing, bot_pos=active_bot_pos)
                if not ev['valid']:
                    continue
                
                savings = active_solo_cost - total_ticks
                print(f"  \u1f91d Helper {idle_id} on {path_label} seg{seg_idx}: "
                      f"nav={h_dist}+turn={h_turns} push={num_cells} "
                      f"prefix={prefix_cost} total={total_ticks} saves={savings}")
                
                if seg_idx == 0:
                    best = {
                        'helper_bot_id': idle_id,
                        'helper_staging': helper_staging,
                        'push_direction': push_dir,
                        'num_pushes': num_cells,
                        'block_id': state['blocks'][0]['id'],
                        'active_bot_target': suf_staging,
                        'new_waypoint_start': 1,
                        'override_path': path,
                        'helper_seg_idx': 0,
                        'savings': savings,
                    }
                else:
                    best = {
                        'helper_bot_id': idle_id,
                        'helper_staging': helper_staging,
                        'push_direction': push_dir,
                        'num_pushes': num_cells,
                        'block_id': state['blocks'][0]['id'],
                        'active_bot_target': suf_staging,
                        'new_waypoint_start': seg_idx + 1,
                        'override_path': path[seg_idx:],
                        'full_path': path,
                        'helper_seg_idx': seg_idx,
                        'savings': savings,
                    }
                best_total = total_ticks
    
    if best:
        seg = best['helper_seg_idx']
        print(f"\n  \u2705 HELPER: {best['helper_bot_id']} on seg{seg} "
              f"({best['num_pushes']} pushes, saves {best['savings']})")
    
    return best


def _helper_nav_step(hx, hy, target_x, target_y, facing, occupied, grid_size):
    """
    Pick the best single-step move for helper bot toward target.
    Prefers the axis matching current facing to minimize rotation ticks.
    Returns (dx, dy) or None if completely blocked.
    """
    hdx = target_x - hx
    hdy = target_y - hy
    blocked = occupied - {(hx, hy)}
    
    # Build candidate moves: (dx, dy, remaining_distance_on_this_axis)
    candidates = []
    if abs(hdx) >= SNAP:
        candidates.append((grid_size if hdx > 0 else -grid_size, 0, abs(hdx)))
    if abs(hdy) >= SNAP:
        candidates.append((0, grid_size if hdy > 0 else -grid_size, abs(hdy)))
    
    if not candidates:
        return None
    
    # Sort: prefer axis matching current facing (avoids rotation tick),
    # then prefer the longer remaining axis (finish one axis before switching)
    def sort_key(c):
        dx, dy, dist = c
        move_dir = (1 if dx > 0 else (-1 if dx < 0 else 0),
                    1 if dy > 0 else (-1 if dy < 0 else 0))
        matches_facing = 1 if move_dir == facing else 0
        return (-matches_facing, -dist)
    
    candidates.sort(key=sort_key)
    
    for dx, dy, _ in candidates:
        if (hx + dx, hy + dy) not in blocked:
            return (dx, dy)
    
    return None


# ── MAIN ENTRY POINT ──────────────────────────────────────────────────────────

def compute_swarm_moves(state, grid_size):
    """
    Main function called by UI.
    Now with obstacle classification, intelligent path planning,
    and NAV-phase collision avoidance.
    
    CRITICAL: Only the FIRST bot moves. All others stay idle
    (unless asked to move out of the way).
    
    ROTATION HANDLING: The UI may rotate a bot without moving it.
    If we detect the bot hasn't moved since our last command,
    we re-issue the same command (the UI needs it again after rotation).
    """
    moves = {}

    # First, set ALL bots to idle
    for bot in state['bots']:
        moves[bot['id']] = (0, 0, None)
    
    if not state['bots'] or not state['goals'] or not state['blocks']:
        return moves
    
    # ── Build OCCUPIED set: every cell with ANY entity ─────────────────
    # This is the ground truth for collision checking
    occupied = set()
    for bot in state['bots']:
        pos = (_snap(bot['pos'][0], grid_size), _snap(bot['pos'][1], grid_size))
        occupied.add(pos)
    for block in state['blocks']:
        pos = (_snap(block['pos'][0], grid_size), _snap(block['pos'][1], grid_size))
        occupied.add(pos)
    
    # ── Process pending move requests for idle bots ──────────────────
    completed_requests = []
    for req_bot_id, req_data in list(PENDING_MOVE_REQUESTS.items()):
        req_data['countdown'] -= 1
        
        if req_data['countdown'] <= 0:
            target = req_data['to']
            for other_bot in state['bots']:
                if other_bot['id'] == req_bot_id:
                    curr_x = _snap(other_bot['pos'][0], grid_size)
                    curr_y = _snap(other_bot['pos'][1], grid_size)
                    dx = target[0] - curr_x
                    dy = target[1] - curr_y
                    
                    # Check if idle bot already reached target
                    if abs(dx) < SNAP and abs(dy) < SNAP:
                        completed_requests.append(req_bot_id)
                        break
                    
                    # Move one step at a time
                    step_dx = step_dy = 0
                    if abs(dx) >= SNAP:
                        step_dx = grid_size if dx > 0 else -grid_size
                    elif abs(dy) >= SNAP:
                        step_dy = grid_size if dy > 0 else -grid_size
                    
                    # HARD CHECK: don't move into occupied cell
                    next_pos = (curr_x + step_dx, curr_y + step_dy)
                    # Remove self from occupied (we're moving FROM here)
                    self_pos = (curr_x, curr_y)
                    other_occupied = occupied - {self_pos}
                    
                    if next_pos in other_occupied:
                        # Can't move there — skip this tick, try next
                        print(f"  ⚠️ Idle bot {req_bot_id} blocked at {next_pos}, waiting...")
                        break
                    
                    moves[req_bot_id] = (step_dx, step_dy, None)
                    
                    # DO NOT mark complete based on projected position!
                    # The UI might rotate instead of moving (heading mismatch).
                    # Completion is checked at the TOP of this loop on the NEXT
                    # tick, using the bot's ACTUAL position.
                    
                    print(f"  🚶 Idle bot {req_bot_id} step ({step_dx},{step_dy}) toward {target}")
                    break
    
    for rid in completed_requests:
        del PENDING_MOVE_REQUESTS[rid]
    
    # Get first (active) bot, block, goal
    bot = state['bots'][0]
    bot_id = bot['id']
    block = state['blocks'][0]
    goal  = state['goals'][0]

    bot_x = _snap(bot['pos'][0],   grid_size)
    bot_y = _snap(bot['pos'][1],   grid_size)
    b_x   = _snap(block['pos'][0], grid_size)
    b_y   = _snap(block['pos'][1], grid_size)
    g_x   = _snap(goal['pos'][0],  grid_size)
    g_y   = _snap(goal['pos'][1],  grid_size)

    # ── DEFERRED HELPER: navigate to staging in background ──────────
    # This must run BEFORE rotation detection, because rotation detection
    # does an early return that would skip the helper.
    if bot_id in BOT_STATES:
        _s = BOT_STATES[bot_id]
        _deferred = _s.get('deferred_helper')
        if _deferred and not _s.get('helper_plan'):
            _d_id = _deferred['helper_bot_id']
            _d_staging = _deferred['helper_staging']
            _d_push_dir = _deferred['push_direction']
            for _hb in state['bots']:
                if _hb['id'] == _d_id:
                    _dhx = _snap(_hb['pos'][0], grid_size)
                    _dhy = _snap(_hb['pos'][1], grid_size)
                    _d_facing = _hb.get('facing', (1, 0))
                    if abs(_d_staging[0] - _dhx) >= SNAP or abs(_d_staging[1] - _dhy) >= SNAP:
                        _nav = _helper_nav_step(_dhx, _dhy, _d_staging[0], _d_staging[1],
                                                _d_facing, occupied, grid_size)
                        if _nav:
                            moves[_d_id] = (_nav[0], _nav[1], None)
                    else:
                        # At staging — pre-orient to push direction
                        if _d_facing != _d_push_dir:
                            moves[_d_id] = (_d_push_dir[0] * grid_size,
                                            _d_push_dir[1] * grid_size, None)
                    break

    # ── ROTATION DETECTION: If bot hasn't moved since last tick, ──────
    # the UI is doing a rotation. Re-issue the same move command.
    if bot_id in BOT_STATES:
        s = BOT_STATES[bot_id]
        last_move = s.get('last_issued_move')
        last_pos = s.get('last_seen_pos')
        
        if (last_move and last_pos and 
            last_move != (0, 0, None) and
            _pos_eq((bot_x, bot_y), last_pos)):
            # Bot hasn't moved! UI must be rotating. Re-issue same command.
            print(f"  🔄 Bot at same position {last_pos} — re-issuing last move {last_move}")
            moves[bot_id] = last_move
            return moves

    # Check if goal reached
    if _pos_eq((b_x, b_y), (g_x, g_y)):
        BOT_STATES[bot_id] = {
            'phase': 'DONE',
            'facing': (1,0),
            'waypoint_queue': [],
            'block_waypoints': [],
            'current_waypoint_idx': 0
        }
        moves[bot_id] = (0, 0, None)
        return moves

    # ── Initialize with obstacle-aware path planning ──────────────────
    if bot_id not in BOT_STATES:
        obstacle_classification = classify_obstacles(state, grid_size)
        
        print(f"\n{'='*60}")
        print(f"🤖 INITIALIZING PATH FOR BOT 1")
        print(f"   Block position: ({b_x}, {b_y})")
        print(f"   Goal position: ({g_x}, {g_y})")
        print(f"   Obstacles detected:")
        print(f"     - Static immovable: {len(obstacle_classification['static_immovable'])}")
        print(f"     - Static movable: {len(obstacle_classification['static_movable'])}")
        print(f"     - Dynamic: {len(obstacle_classification['dynamic_predictable'])}")
        print(f"{'='*60}")
        
        path_plan = find_optimal_block_path(
            (b_x, b_y),
            (g_x, g_y),
            obstacle_classification,
            grid_size,
            bot_facing=(1, 0),
            bot_pos=(bot_x, bot_y)
        )
        
        if path_plan is None:
            print("\n" + "="*60)
            print("❌ GOAL IS UNREACHABLE")
            print("="*60)
            
            BOT_STATES[bot_id] = {
                'phase': 'IMPOSSIBLE',
                'facing': [1, 0],
                'waypoint_queue': [],
                'block_waypoints': [],
                'current_waypoint_idx': 0,
                'error_message': 'Goal is completely surrounded - no path exists'
            }
            moves[bot_id] = (0, 0, None)
            return moves
        
        # Execute PLANNING-PHASE bot move requests
        if path_plan.get('requests'):
            print(f"\n📋 Scheduling {len(path_plan['requests'])} bot move requests:")
            for request in path_plan['requests']:
                if request['type'] == 'move_bot':
                    req_bot_id = request['bot_id']
                    target_pos = request['to']
                    print(f"  ➜ Will move {req_bot_id} to {target_pos}")
                    PENDING_MOVE_REQUESTS[req_bot_id] = {
                        'to': target_pos,
                        'countdown': 0
                    }
        
        # ── HELPER BOT PLANNING ───────────────────────────────────────
        solo_ticks = _estimate_ticks(path_plan['waypoints'], (1, 0),
                                      grid_size, bot_pos=(bot_x, bot_y))
        helper_plan = _plan_helper_push(
            state, path_plan['waypoints'],
            (bot_x, bot_y), (1, 0),
            (b_x, b_y), grid_size, obstacle_classification,
            active_solo_cost=solo_ticks
        )
        
        # If helper plan overrides the path, use the helper's preferred path
        # When helper is on seg > 0, we defer it: active bot does prefix solo,
        # and the helper will be re-evaluated when the block reaches that segment.
        final_waypoints = path_plan['waypoints']
        active_helper_plan = None
        
        if helper_plan:
            seg = helper_plan.get('helper_seg_idx', 0)
            # Cancel any pending move request for the helper bot —
            # the helper has its own navigation plan to staging
            h_id = helper_plan['helper_bot_id']
            if h_id in PENDING_MOVE_REQUESTS:
                print(f"  🤝 Cancelling pending move request for helper {h_id}")
                del PENDING_MOVE_REQUESTS[h_id]
            if seg == 0:
                # Helper on seg 0 — activate immediately
                active_helper_plan = helper_plan
                if helper_plan.get('override_path'):
                    final_waypoints = helper_plan['override_path']
                    print(f"  🤝 Overriding path with helper-optimal: {final_waypoints}")
            else:
                # Helper on later segment — use full path, defer helper activation
                final_waypoints = helper_plan['full_path']
                print(f"  🤝 Deferred helper on seg{seg} — active bot does prefix first")
                print(f"     Full path: {final_waypoints}")
        
        BOT_STATES[bot_id] = {
            'phase': 'NAV',
            'facing': [1, 0],
            'waypoint_queue': [],
            'block_waypoints': final_waypoints,
            'current_waypoint_idx': 0,
            'requests': path_plan['requests'],
            'wait_periods': path_plan['wait_periods'],
            'obstacle_classification': obstacle_classification,
            'waiting_for_clear': False,
            'helper_plan': active_helper_plan,
            'deferred_helper': helper_plan if helper_plan and helper_plan.get('helper_seg_idx', 0) > 0 else None,
        }

    s = BOT_STATES[bot_id]
    
    if s['phase'] == 'STUCK' or s['phase'] == 'DONE' or s['phase'] == 'IMPOSSIBLE':
        moves[bot_id] = (0, 0, None)
        return moves
    
    # ── If we're waiting for an idle bot to clear, check if it's done ──
    if s.get('waiting_for_clear'):
        if PENDING_MOVE_REQUESTS:
            # Still waiting — don't move this tick
            print(f"  ⏳ Active bot waiting for idle bot(s) to clear...")
            moves[bot_id] = (0, 0, None)
            return moves
        else:
            # Clear! Resume navigation
            print(f"  ✅ Path cleared! Resuming navigation.")
            s['waiting_for_clear'] = False
            # Refresh obstacle classification since bots have moved
            s['obstacle_classification'] = classify_obstacles(state, grid_size)
    
    phase = s['phase']
    facing = tuple(s['facing'])
    wq = s['waypoint_queue']
    block_waypoints = s['block_waypoints']
    current_waypoint_idx = s['current_waypoint_idx']
    
    # ── HELPER BOT EXECUTION ──────────────────────────────────────────
    # The helper pushes segment S while the active bot handles prefix/suffix.
    # If helper_seg_idx > 0, the active bot works on prefix segments normally
    # while the helper navigates to its staging in the background.
    helper_plan = s.get('helper_plan')
    helper_pushing_this_tick = False
    helper_seg_idx = helper_plan.get('helper_seg_idx', 0) if helper_plan else 0
    
    # Is the helper currently handling a segment?
    # True when block is at or past the helper's segment start AND helper has pushes left
    block_at_helper_seg = False
    if helper_plan and helper_plan.get('num_pushes', 0) > 0:
        # Helper is active — gate the active bot regardless of block position
        block_at_helper_seg = True
    
    if helper_plan and helper_plan.get('num_pushes', 0) > 0:
        helper_id = helper_plan['helper_bot_id']
        push_dir = helper_plan['push_direction']
        block_id_to_push = helper_plan['block_id']
        
        # Find the helper bot in state
        helper_bot = None
        for hb in state['bots']:
            if hb['id'] == helper_id:
                helper_bot = hb
                break
        
        if helper_bot:
            hx = _snap(helper_bot['pos'][0], grid_size)
            hy = _snap(helper_bot['pos'][1], grid_size)
            helper_facing = helper_bot.get('facing', (1, 0))
            
            # Track helper's last position to detect rotation ticks
            last_helper_pos = helper_plan.get('last_helper_pos')
            last_helper_move = helper_plan.get('last_helper_move')
            
            # Rotation detection for helper
            if (last_helper_move and last_helper_pos and
                last_helper_move != (0, 0, None) and
                _pos_eq((hx, hy), last_helper_pos)):
                print(f"  🤝 Helper rotating — re-issuing last move")
                moves[helper_id] = last_helper_move
                helper_pushing_this_tick = (last_helper_move[2] is not None)
            elif (last_helper_move and last_helper_pos and
                  last_helper_move[2] is not None and
                  not _pos_eq((hx, hy), last_helper_pos)):
                # Helper push landed
                helper_plan['num_pushes'] -= 1
                print(f"  🤝 Helper push landed! ({helper_plan['num_pushes']} remaining)")
                
                if helper_plan['num_pushes'] <= 0:
                    print(f"  🤝 Helper DONE. Active bot takes over.")
                    s['current_waypoint_idx'] = helper_plan.get('new_waypoint_start', helper_seg_idx + 1)
                    s['helper_plan'] = None
                    s['waypoint_queue'] = []
                    s['obstacle_classification'] = classify_obstacles(state, grid_size)
                else:
                    expected_block_x = hx + push_dir[0] * grid_size
                    expected_block_y = hy + push_dir[1] * grid_size
                    if _pos_eq((b_x, b_y), (expected_block_x, expected_block_y)):
                        moves[helper_id] = (push_dir[0]*grid_size, push_dir[1]*grid_size, block_id_to_push)
                        helper_pushing_this_tick = True
            elif block_at_helper_seg:
                # Block is at helper's segment — helper should push
                expected_block_x = hx + push_dir[0] * grid_size
                expected_block_y = hy + push_dir[1] * grid_size
                
                if _pos_eq((b_x, b_y), (expected_block_x, expected_block_y)):
                    moves[helper_id] = (push_dir[0]*grid_size, push_dir[1]*grid_size, block_id_to_push)
                    helper_pushing_this_tick = True
                    if helper_facing == push_dir:
                        print(f"  🤝 Helper {helper_id} pushing (facing correct)")
                    else:
                        print(f"  🤝 Helper {helper_id} turning to face {push_dir}")
                else:
                    # Helper at staging but block not adjacent yet
                    helper_staging = helper_plan.get('helper_staging')
                    if helper_staging and _pos_eq((hx, hy), helper_staging):
                        # At staging, waiting for block — pre-orient to push direction
                        if helper_facing != push_dir:
                            orient_dx = push_dir[0] * grid_size
                            orient_dy = push_dir[1] * grid_size
                            moves[helper_id] = (orient_dx, orient_dy, None)
                            print(f"  🤝 Helper pre-orienting to face {push_dir}")
                        else:
                            print(f"  🤝 Helper at staging, ready to push")
                    elif helper_staging:
                        # Still navigating
                        hdx = helper_staging[0] - hx
                        hdy = helper_staging[1] - hy
                        if abs(hdx) < SNAP and abs(hdy) < SNAP:
                            print(f"  🤝 Helper at staging, block not adjacent — cancelling")
                            s['helper_plan'] = None
                        else:
                            nav = _helper_nav_step(hx, hy, helper_staging[0], helper_staging[1],
                                                   helper_facing, occupied, grid_size)
                            if nav:
                                moves[helper_id] = (nav[0], nav[1], None)
                            else:
                                s['helper_plan'] = None
                    else:
                        s['helper_plan'] = None
            else:
                # Block hasn't reached helper's segment yet —
                # helper navigates to staging in the background
                helper_staging = helper_plan.get('helper_staging')
                if helper_staging:
                    hdx = helper_staging[0] - hx
                    hdy = helper_staging[1] - hy
                    
                    if abs(hdx) >= SNAP or abs(hdy) >= SNAP:
                        nav = _helper_nav_step(hx, hy, helper_staging[0], helper_staging[1],
                                               helper_facing, occupied, grid_size)
                        if nav:
                            moves[helper_id] = (nav[0], nav[1], None)
                            print(f"  🤝 Helper {helper_id} nav toward staging {helper_staging}")
                        else:
                            print(f"  🤝 Helper blocked, cancelling")
                            s['helper_plan'] = None
                    # else: already at staging — pre-orient to push direction
                    else:
                        if helper_facing != push_dir:
                            orient_dx = push_dir[0] * grid_size
                            orient_dy = push_dir[1] * grid_size
                            moves[helper_id] = (orient_dx, orient_dy, None)
            
            # Store position and move for rotation detection next tick
            helper_plan_ref = s.get('helper_plan')
            if helper_plan_ref:
                helper_plan_ref['last_helper_pos'] = (hx, hy)
                helper_plan_ref['last_helper_move'] = moves.get(helper_id, (0, 0, None))
        
        # Gate active bot: only wait if block is at helper's segment
        # AND helper still has pushes to do
        helper_still_active = helper_plan and helper_plan.get('num_pushes', 0) > 0
        active_target = helper_plan.get('active_bot_target') if helper_plan else None
        
        if block_at_helper_seg and helper_still_active:
            # Block is at helper's segment — active bot must NOT push
            if active_target:
                adx = active_target[0] - bot_x
                ady = active_target[1] - bot_y
                if abs(adx) >= SNAP or abs(ady) >= SNAP:
                    move_dx = move_dy = 0
                    if abs(adx) >= SNAP:
                        move_dx = grid_size if adx > 0 else -grid_size
                    elif abs(ady) >= SNAP:
                        move_dy = grid_size if ady > 0 else -grid_size
                    
                    check_occupied = occupied - {(bot_x, bot_y)}
                    helper_move = moves.get(helper_id, (0,0,None))
                    if helper_move[2] is not None:
                        check_occupied.add((b_x + push_dir[0]*grid_size, b_y + push_dir[1]*grid_size))
                        if helper_bot:
                            check_occupied.add((_snap(helper_bot['pos'][0],grid_size)+push_dir[0]*grid_size,
                                               _snap(helper_bot['pos'][1],grid_size)+push_dir[1]*grid_size))
                    
                    next_pos = (bot_x + move_dx, bot_y + move_dy)
                    if next_pos not in check_occupied:
                        facing = (move_dx//grid_size if move_dx else 0, move_dy//grid_size if move_dy else 0)
                        BOT_STATES[bot_id]['facing'] = list(facing)
                        final_move = (move_dx, move_dy, None)
                        BOT_STATES[bot_id]['last_issued_move'] = final_move
                        BOT_STATES[bot_id]['last_seen_pos'] = (bot_x, bot_y)
                        moves[bot_id] = final_move
                        print(f"  🏃 Active bot navigating to suffix staging")
                        return moves
                
                print(f"  ⏳ Active bot waiting for helper to finish seg{helper_seg_idx}...")
                BOT_STATES[bot_id]['last_issued_move'] = (0, 0, None)
                BOT_STATES[bot_id]['last_seen_pos'] = (bot_x, bot_y)
                return moves
            else:
                print(f"  ⏳ Active bot waiting for helper to finish seg{helper_seg_idx}...")
                BOT_STATES[bot_id]['last_issued_move'] = (0, 0, None)
                BOT_STATES[bot_id]['last_seen_pos'] = (bot_x, bot_y)
                return moves
        
        # If helper_seg_idx == 0 and block not at seg yet, active must also wait
        # (because block starts at seg 0, so block_at_helper_seg should be true)
        # For seg_idx > 0, active bot falls through to normal NAV/PUSH below
    
    # Get obstacle classification (refresh for moved bots)
    obstacle_classification = s.get('obstacle_classification')
    if obstacle_classification is None:
        obstacle_classification = classify_obstacles(state, grid_size)
        s['obstacle_classification'] = obstacle_classification
    
    # Refresh local variables — helper completion may have changed them
    phase = s['phase']
    facing = tuple(s['facing'])
    wq = s['waypoint_queue']
    block_waypoints = s['block_waypoints']
    current_waypoint_idx = s['current_waypoint_idx']
    
    obstacles = obstacle_classification['all_obstacle_positions'].copy()
    # CRITICAL: Also add the active block's position — the bot can't walk through it!
    obstacles.add((b_x, b_y))
    
    # Check if we've reached final waypoint
    if current_waypoint_idx >= len(block_waypoints):
        moves[bot_id] = (0, 0, None)
        return moves
    
    target_waypoint = block_waypoints[current_waypoint_idx]
    
    print(f"\n🎯 Current waypoint: {current_waypoint_idx}/{len(block_waypoints)-1}")
    print(f"   Block at: ({b_x}, {b_y})")
    print(f"   Target waypoint: {target_waypoint}")
    print(f"   Final goal: ({g_x}, {g_y})")
    
    # Check if block reached this waypoint
    if _pos_eq((b_x, b_y), target_waypoint):
        print(f"   ✅ Waypoint {current_waypoint_idx} REACHED!")
        current_waypoint_idx += 1
        s['current_waypoint_idx'] = current_waypoint_idx
        
        if current_waypoint_idx >= len(block_waypoints):
            print(f"   🎉 ALL WAYPOINTS COMPLETED - GOAL REACHED!")
            s['phase'] = 'DONE'
            moves[bot_id] = (0, 0, None)
            return moves
        
        target_waypoint = block_waypoints[current_waypoint_idx]
        print(f"   Next target: {target_waypoint}")
        
        # ── DEFERRED HELPER ACTIVATION ──
        # When block reaches the deferred helper's segment start, activate it
        deferred = s.get('deferred_helper')
        if deferred and not s.get('helper_plan'):
            dseg = deferred.get('helper_seg_idx', 0)
            if current_waypoint_idx == dseg + 1:
                helper_id = deferred['helper_bot_id']
                helper_staging = deferred['helper_staging']
                for hb in state['bots']:
                    if hb['id'] == helper_id:
                        hx = _snap(hb['pos'][0], grid_size)
                        hy = _snap(hb['pos'][1], grid_size)
                        h_dist = int((abs(hx - helper_staging[0]) + abs(hy - helper_staging[1])) / grid_size)
                        if h_dist <= 1:
                            print(f"   🤝 Activating deferred helper {helper_id} on seg{dseg}")
                            s['helper_plan'] = deferred
                            s['deferred_helper'] = None
                            # Stop active bot immediately — helper takes over
                            BOT_STATES[bot_id]['last_issued_move'] = (0, 0, None)
                            BOT_STATES[bot_id]['last_seen_pos'] = (bot_x, bot_y)
                            moves[bot_id] = (0, 0, None)
                            return moves
                        else:
                            print(f"   🤝 Deferred helper too far ({h_dist} cells), skipping")
                            s['deferred_helper'] = None
                        break
    
    # Plan push toward current waypoint
    push_dx, push_dy, stg_x, stg_y, push_axis = _push_plan_to_waypoint(
        b_x, b_y, target_waypoint[0], target_waypoint[1], grid_size
    )
    
    move_dx = move_dy = 0
    pushing = False

    if phase == 'PUSH':
        phase = 'NAV'; wq = []

    # ── NAV phase ──────────────────────────────────────────────────
    if phase == 'NAV':
        while wq and _pos_eq((bot_x, bot_y), wq[0]): wq.pop(0)

        if _pos_eq((bot_x, bot_y), (stg_x, stg_y)):
            phase = 'PUSH'
            s['nav_stuck_count'] = 0
        else:
            if not wq:
                wq = _best_route(
                    bot_x, bot_y, stg_x, stg_y,
                    b_x, b_y, push_axis, grid_size, facing,
                    obstacles=obstacles
                )
                while wq and _pos_eq((bot_x, bot_y), wq[0]): wq.pop(0)

            if wq:
                tgt_x, tgt_y = wq[0]
                dx = _step(bot_x, tgt_x, grid_size)
                dy = _step(bot_y, tgt_y, grid_size)
                if dx != 0:   move_dx = dx
                elif dy != 0: move_dy = dy

                if _close(bot_x + move_dx, b_x) and _close(bot_y + move_dy, b_y):
                    move_dx, move_dy = _emergency_step(
                        bot_x, bot_y, b_x, b_y, push_axis, grid_size
                    )
                    wq = []
            
            # ── STUCK DETECTION: if NAV isn't making progress, replan ──
            nav_stuck = s.get('nav_stuck_count', 0) + 1
            s['nav_stuck_count'] = nav_stuck
            
            if nav_stuck > 6:
                # Bot is oscillating — NAV can't reach staging.
                # The block path itself needs to change.
                replan_attempts = s.get('replan_attempts', 0) + 1
                s['replan_attempts'] = replan_attempts
                
                if replan_attempts > 3:
                    # Too many failed replans — truly stuck
                    print(f"\n❌ Replan failed {replan_attempts} times — marking STUCK")
                    s['phase'] = 'STUCK'
                    move_dx, move_dy = 0, 0
                else:
                    print(f"\n🔄 NAV stuck for {nav_stuck} ticks — full replan (attempt {replan_attempts})!")
                    s['nav_stuck_count'] = 0
                    
                    new_obs = classify_obstacles(state, grid_size)
                    new_plan = find_optimal_block_path(
                        (b_x, b_y), (g_x, g_y),
                        new_obs, grid_size,
                        bot_facing=facing,
                        bot_pos=(bot_x, bot_y)
                    )
                    
                    if new_plan is None:
                        s['phase'] = 'STUCK'
                        move_dx, move_dy = 0, 0
                    else:
                        s['block_waypoints'] = new_plan['waypoints']
                        s['current_waypoint_idx'] = 0
                        s['waypoint_queue'] = []
                        s['obstacle_classification'] = new_obs
                        block_waypoints = new_plan['waypoints']
                        wq = []
                        move_dx, move_dy = 0, 0

    # ── PUSH phase ─────────────────────────────────────────────────
    if phase == 'PUSH':
        move_dx, move_dy = push_dx, push_dy
        pushing = True
        phase = 'NAV'
        wq = []

    # Update facing
    if move_dx != 0 or move_dy != 0:
        facing = (move_dx // grid_size if move_dx else 0,
                  move_dy // grid_size if move_dy else 0)

    # ── FIX 2: NAV-PHASE COLLISION CHECK ─────────────────────────────
    # Before committing the move, check if next cell has an idle bot
    next_bot_x = bot_x + move_dx
    next_bot_y = bot_y + move_dy
    
    collision_detected = False
    
    if move_dx != 0 or move_dy != 0:
        # Refresh obstacle positions from LIVE state (bots may have moved)
        live_obstacle_classification = classify_obstacles(state, grid_size)
        
        # Compute the block's full path corridor for path-aware displacement
        block_path_corridor = get_full_path_corridor(block_waypoints, grid_size)
        
        # Check for idle bot at the destination
        nav_collision = _check_nav_collision(
            next_bot_x, next_bot_y,
            live_obstacle_classification, grid_size,
            active_bot_pos=(bot_x, bot_y),
            path_corridor=block_path_corridor
        )
        
        if nav_collision is not None:
            collision_detected = True
            colliding_bot = nav_collision['bot_id']
            target_cell = nav_collision['move_to']
            
            if target_cell is not None:
                print(f"\n🚧 NAV COLLISION: Idle bot {colliding_bot} at ({next_bot_x}, {next_bot_y})")
                print(f"   📢 Asking it to move to {target_cell}")
                
                # Schedule the idle bot to move
                PENDING_MOVE_REQUESTS[colliding_bot] = {
                    'to': target_cell,
                    'countdown': 0
                }
                
                # Active bot WAITS this tick
                s['waiting_for_clear'] = True
                move_dx, move_dy = 0, 0
                pushing = False
                
                # Update obstacle classification for next tick
                s['obstacle_classification'] = live_obstacle_classification
            else:
                print(f"\n🚧 NAV COLLISION: Bot {colliding_bot} is surrounded — can't move it!")
                print(f"   🔄 Replanning path...")
                
                # Replan around this bot
                new_obstacle_class = classify_obstacles(state, grid_size)
                new_path_plan = find_optimal_block_path(
                    (b_x, b_y), (g_x, g_y),
                    new_obstacle_class, grid_size,
                    bot_facing=tuple(s['facing']),
                    bot_pos=(bot_x, bot_y)
                )
                
                if new_path_plan is None:
                    s['phase'] = 'STUCK'
                    move_dx, move_dy = 0, 0
                else:
                    s['block_waypoints'] = new_path_plan['waypoints']
                    s['current_waypoint_idx'] = 0
                    s['waypoint_queue'] = []
                    s['obstacle_classification'] = new_obstacle_class
                    move_dx, move_dy = 0, 0
        
        # Check for IMMOVABLE obstacle collision (existing logic)
        if not collision_detected:
            for obs in live_obstacle_classification.get('static_immovable', []):
                if _pos_eq((next_bot_x, next_bot_y), obs['pos']):
                    collision_detected = True
                    print(f"\n⚠️  COLLISION: Bot would hit static block at {obs['pos']}")
                    
                    # DON'T replan the entire block path — the path is fine,
                    # it's just the NAV route to staging that hit an obstacle.
                    # Clear the nav waypoint queue and refresh obstacles so
                    # _best_route recalculates a route AROUND the obstacle.
                    move_dx, move_dy = 0, 0
                    pushing = False
                    s['waypoint_queue'] = []
                    s['obstacle_classification'] = live_obstacle_classification
                    break
    
    # Check if BLOCK would hit obstacle during push
    if pushing and not collision_detected:
        next_block_x = b_x + move_dx
        next_block_y = b_y + move_dy
        
        live_obstacle_classification = classify_obstacles(state, grid_size)
        for obs in live_obstacle_classification.get('static_immovable', []):
            if _pos_eq((next_block_x, next_block_y), obs['pos']):
                print(f"\n⚠️  COLLISION: Block would hit obstacle at {obs['pos']}")
                s['phase'] = 'STUCK'
                move_dx, move_dy = 0, 0
                pushing = False
                break

    # ── HARD COLLISION CHECK against ALL occupied cells ─────────────
    # This is the final safety net — no bot should EVER overlap another entity.
    # Unlike the earlier checks, this one triggers PROPER responses:
    # - Idle bot in the way → issue move request, wait
    # - Block in the way → replan path
    if move_dx != 0 or move_dy != 0:
        next_bot_pos = (bot_x + move_dx, bot_y + move_dy)
        # Remove active bot's current position from occupied (it's moving away)
        check_occupied = occupied - {(bot_x, bot_y)}
        
        if next_bot_pos in check_occupied:
            # Check if it's the block we're pushing
            if pushing and _pos_eq(next_bot_pos, (b_x, b_y)):
                # That's OK — we're pushing the block
                # But check where the BLOCK would end up
                next_block_pos = (b_x + move_dx, b_y + move_dy)
                block_check = check_occupied - {(b_x, b_y)}
                if next_block_pos in block_check:
                    print(f"\n⛔ HARD BLOCK: Block would overlap entity at {next_block_pos}")
                    move_dx, move_dy = 0, 0
                    pushing = False
                    s['waypoint_queue'] = []
                    s['obstacle_classification'] = classify_obstacles(state, grid_size)
            else:
                # Something is at next_bot_pos. Figure out what and respond properly.
                # Check if it's an idle bot (movable)
                handled = False
                live_obs = classify_obstacles(state, grid_size)
                for obs in live_obs.get('static_movable', []):
                    if _pos_eq(next_bot_pos, obs['pos']):
                        # It's an idle bot — ask it to move out of the way
                        block_path_corridor = get_full_path_corridor(block_waypoints, grid_size)
                        target = find_clear_cell_outside_path(
                            obs['pos'],
                            live_obs['all_obstacle_positions'] | {(bot_x, bot_y)},
                            block_path_corridor,
                            grid_size,
                            bot_facing=obs.get('facing')
                        )
                        if target is not None:
                            print(f"\n⛔ HARD BLOCK → idle bot {obs['bot_id']} at {next_bot_pos}")
                            print(f"   📢 Asking it to move to {target}")
                            PENDING_MOVE_REQUESTS[obs['bot_id']] = {
                                'to': target,
                                'countdown': 0
                            }
                            s['waiting_for_clear'] = True
                        move_dx, move_dy = 0, 0
                        pushing = False
                        s['waypoint_queue'] = []
                        s['obstacle_classification'] = live_obs
                        handled = True
                        break
                
                if not handled:
                    # It's a block or the active block (not pushing) — replan
                    print(f"\n⛔ HARD BLOCK: Bot would overlap entity at {next_bot_pos}")
                    move_dx, move_dy = 0, 0
                    pushing = False
                    s['waypoint_queue'] = []
                    s['obstacle_classification'] = classify_obstacles(state, grid_size)

    BOT_STATES[bot_id]['phase'] = phase
    BOT_STATES[bot_id]['facing'] = list(facing)
    BOT_STATES[bot_id]['waypoint_queue'] = wq
    
    # ── Store last move and position for rotation detection ──────────
    final_move = (move_dx, move_dy, block['id'] if pushing else None)
    BOT_STATES[bot_id]['last_issued_move'] = final_move
    BOT_STATES[bot_id]['last_seen_pos'] = (bot_x, bot_y)

    moves[bot_id] = final_move

    return moves
