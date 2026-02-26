import tkinter as tk
from tkinter import messagebox
import logic01

class GridPlatform:
    def __init__(self, root):
        self.root = root
        self.root.title("Swarm Architect Pro")
        try: self.root.state('zoomed')
        except: pass
        self.root.configure(bg="#1e1e1e")

        self.rows, self.cols, self.grid_size = 20, 20, 40
        self.bg_color, self.grid_color = "#1e1e1e", "#444444"
        self.fit_screen_var = tk.BooleanVar(value=False)

        self.is_simulating = False
        self.initial_state = []
        self.undo_stack = []
        self.redo_stack = []
        self.selected_tool, self.tool_color = None, None
        self.last_vertex = None
        self.occupancy_map = {}

        # ── STATS TRACKING ──
        self.step_count = 0
        self.turn_count = 0
        self.last_path_dir = None
        self.path_history = []

        self.setup_ui()
        self.draw_grid()
        self.bind_shortcuts()

    # ── STATS ─────────────────────────────────────────────────────────────────

    def update_stats_display(self):
        self.step_label.config(text=f"STEPS:  {self.step_count}")
        self.turn_label.config(text=f"TURNS:  {self.turn_count}")

    def reset_counters(self):
        self.step_count = 0
        self.turn_count = 0
        self.last_path_dir = None
        self.path_history = []
        self.update_stats_display()

    def add_path_stats(self, dx, dy):
        self.step_count += 1
        current_dir = (int(dx), int(dy))
        if self.last_path_dir and self.last_path_dir != current_dir:
            self.turn_count += 1
        self.last_path_dir = current_dir
        self.path_history.append(current_dir)
        self.update_stats_display()

    def subtract_path_stats(self):
        if self.step_count > 0:
            self.step_count -= 1
            if len(self.path_history) >= 2 and self.path_history[-1] != self.path_history[-2]:
                self.turn_count = max(0, self.turn_count - 1)
            if self.path_history:
                self.path_history.pop()
                self.last_path_dir = self.path_history[-1] if self.path_history else None
            self.update_stats_display()

    # ── UNDO / REDO ───────────────────────────────────────────────────────────

    def undo(self, event=None):
        if not self.undo_stack or self.is_simulating: return
        action, data, pos = self.undo_stack.pop()

        if action == "add":
            # Check if it's a path item before deleting
            tags = self.canvas.gettags(data) if isinstance(data, str) else ()
            if "path" in tags:
                self.subtract_path_stats()
            items_to_save = [self.get_item_data(i) for i in self.canvas.find_withtag(data)]
            self.redo_stack.append(("add", items_to_save, pos))
            self.canvas.delete(data)
            if pos in self.occupancy_map:
                del self.occupancy_map[pos]

        elif action == "delete":
            new_ids = [self.recreate_item(d) for d in data]
            tags = self.canvas.gettags(new_ids[0])
            if "path" in tags:
                c = self.canvas.coords(new_ids[0])
                dx = (c[2] - c[0]) / self.grid_size
                dy = (c[3] - c[1]) / self.grid_size
                self.add_path_stats(dx, dy)
            group_tag = next((t for t in tags if t.startswith("group_")), None)
            if group_tag and isinstance(pos, tuple):
                self.occupancy_map[pos] = group_tag
            self.redo_stack.append(("delete", group_tag or new_ids[0], pos))

    def redo(self, event=None):
        if not self.redo_stack or self.is_simulating: return
        action, data, pos = self.redo_stack.pop()

        if action == "add":
            new_ids = [self.recreate_item(d) for d in data]
            tags = self.canvas.gettags(new_ids[0])
            if "path" in tags:
                c = self.canvas.coords(new_ids[0])
                dx = (c[2] - c[0]) / self.grid_size
                dy = (c[3] - c[1]) / self.grid_size
                self.add_path_stats(dx, dy)
            group_tag = next((t for t in tags if t.startswith("group_")), None)
            if group_tag and isinstance(pos, tuple):
                self.occupancy_map[pos] = group_tag
            self.undo_stack.append(("add", group_tag or new_ids[0], pos))

        elif action == "delete":
            items_to_save = [self.get_item_data(i) for i in self.canvas.find_withtag(data)]
            if items_to_save and "path" in items_to_save[0]['tags']:
                self.subtract_path_stats()
            self.undo_stack.append(("delete", items_to_save, pos))
            self.canvas.delete(data)
            if isinstance(pos, tuple) and pos in self.occupancy_map:
                del self.occupancy_map[pos]

    def bind_shortcuts(self):
        self.root.bind("<Control-z>", self.undo); self.root.bind("<Control-Z>", self.undo)
        self.root.bind("<Control-y>", self.redo); self.root.bind("<Control-Y>", self.redo)

    # ── SIMULATION ────────────────────────────────────────────────────────────

    def rotate_bot_visual(self, bot_tag, dx, dy):
        items = self.canvas.find_withtag(bot_tag)
        h_id = next((i for i in items if "heading" in self.canvas.gettags(i)), None)
        b_id = next((i for i in items if "body" in self.canvas.gettags(i)), None)
        if h_id and b_id:
            c = self.canvas.coords(b_id)
            cx, cy = (c[0]+c[2])/2, (c[1]+c[3])/2
            s = self.grid_size // 3
            h_c = self.canvas.coords(h_id)
            curr_vx, curr_vy = h_c[2]-h_c[0], h_c[3]-h_c[1]
            targ_vx = (dx/self.grid_size)*s
            targ_vy = (dy/self.grid_size)*s
            # 180° turn: animate through perpendicular, then finish
            if abs(curr_vx + targ_vx) < 1 and abs(curr_vy + targ_vy) < 1:
                inter_vx, inter_vy = -curr_vy, curr_vx
                self.canvas.coords(h_id, cx, cy, cx+inter_vx, cy+inter_vy)
                self.turn_count += 1
                self.update_stats_display()
                self.root.after(150, lambda: self._finish_180(h_id, cx, cy, targ_vx, targ_vy))
            else:
                self.canvas.coords(h_id, cx, cy, cx+targ_vx, cy+targ_vy)
                self.turn_count += 1
                self.update_stats_display()

    def _finish_180(self, h_id, cx, cy, tx, ty):
        self.canvas.coords(h_id, cx, cy, cx+tx, cy+ty)
        self.turn_count += 1
        self.update_stats_display()

    def run_logic_loop(self):
        if not self.is_simulating: return
        state = {'bots': [], 'blocks': [], 'goals': []}
        for item in self.canvas.find_all():
            tags = self.canvas.gettags(item)
            if "grid" in tags or "sudo" in tags or "path" in tags: continue
            coords = self.canvas.coords(item)
            if not coords: continue
            center = ((coords[0]+coords[-2])/2, (coords[1]+coords[-1])/2)
            if "bot" in tags and "body" in tags:
                group_tag = next((t for t in tags if t.startswith("group_")), None)
                if group_tag: state['bots'].append({'id': group_tag, 'pos': center})
            elif "block" in tags: state['blocks'].append({'id': item, 'pos': center})
            elif "goal" in tags:  state['goals'].append({'id': item, 'pos': center})

        moves = logic01.compute_swarm_moves(state, self.grid_size)
        action_taken = False
        for bot_id, (dx, dy, block_id) in moves.items():
            if dx == 0 and dy == 0: continue
            items = self.canvas.find_withtag(bot_id)
            h_id = next((i for i in items if "heading" in self.canvas.gettags(i)), None)
            if not h_id: continue
            h_c = self.canvas.coords(h_id)
            s = self.grid_size // 3
            target_v = ((dx/self.grid_size)*s, (dy/self.grid_size)*s)
            curr_dx, curr_dy = h_c[2]-h_c[0], h_c[3]-h_c[1]
            if abs(curr_dx - target_v[0]) > 1 or abs(curr_dy - target_v[1]) > 1:
                self.rotate_bot_visual(bot_id, dx, dy)
            else:
                self.canvas.move(bot_id, dx, dy)
                if block_id: self.canvas.move(block_id, dx, dy)
                self.step_count += 1
                self.update_stats_display()
            action_taken = True
        if action_taken: self.root.after(300, self.run_logic_loop)
        else: self.toggle_play()

    def confirm_clear(self):
        if messagebox.askyesno("Confirm Clear", "Are you sure?"):
            self.canvas.delete("all")
            self.draw_grid()
            self.occupancy_map.clear()
            self.undo_stack.clear()
            self.redo_stack.clear()
            self.reset_counters()

    def toggle_play(self):
        if self.is_simulating:
            self.is_simulating = False
            self.play_btn.config(text="▶ PLAY LOGIC", bg="#27ae60")
        else:
            self.initial_state = [self.get_item_data(i) for i in self.canvas.find_all()
                                  if "grid" not in self.canvas.gettags(i)]
            self.canvas.delete("path")
            self.reset_counters()
            logic01.reset_logic()
            self.is_simulating = True
            self.play_btn.config(text="⏹ STOP", bg="#c0392b")
            self.run_logic_loop()

    # ── UI SETUP ──────────────────────────────────────────────────────────────

    def setup_ui(self):
        self.sidebar = tk.Frame(self.root, width=200, bg="#2d2d2d", padx=10, pady=10)
        self.sidebar.pack(side='left', fill='y')
        self.sidebar.pack_propagate(False)

        self.play_btn = tk.Button(self.sidebar, text="▶ PLAY LOGIC", command=self.toggle_play,
                                  bg="#27ae60", fg="white", font=('Arial', 9, 'bold'))
        self.play_btn.pack(fill='x', pady=5)
        tk.Button(self.sidebar, text="🔄 RESET ARENA",  command=self.reset_simulation, bg="#2980b9", fg="white").pack(fill='x', pady=2)
        tk.Button(self.sidebar, text="Resize Grid",      command=self.open_resize_dialog, bg="#444",    fg="white").pack(fill='x', pady=2)
        tk.Button(self.sidebar, text="🗑️ DELETE TOOL",   command=self.set_delete_tool,    bg="#e67e22", fg="white").pack(fill='x', pady=5)
        tk.Button(self.sidebar, text="CLEAR ALL",        command=self.confirm_clear,       bg="#c0392b", fg="white").pack(fill='x', pady=2)

        tk.Label(self.sidebar, text="SOLID ASSETS", fg="white", bg="#2d2d2d",
                 font=('Arial', 9, 'bold')).pack(pady=(10, 5))
        for name, col in [("BOT", "#3498db"), ("BLOCK", "#e74c3c"), ("GOAL", "#2ecc71")]:
            tk.Button(self.sidebar, text=name, bg=col, fg="white",
                      command=lambda n=name, c=col: self.set_tool(n, c)).pack(fill='x', pady=2)

        tk.Label(self.sidebar, text="SUDO ASSETS", fg="#aaa", bg="#2d2d2d",
                 font=('Arial', 9, 'bold')).pack(pady=(10, 5))
        for name, col in [("SUDO BOT", "#3498db"), ("SUDO BLOCK", "#e74c3c")]:
            tk.Button(self.sidebar, text=name, bg=col, fg="white",
                      command=lambda n=name, c=col: self.set_tool(n, c)).pack(fill='x', pady=2)

        # ── LIVE STATS panel ──────────────────────────────────────────────────
        stats_frame = tk.LabelFrame(
            self.sidebar, text=" LIVE STATS ",
            bg="#2d2d2d", fg="#f1c40f",
            font=('Arial', 9, 'bold'),
            padx=8, pady=8
        )
        stats_frame.pack(side='bottom', fill='x', pady=(10, 0))

        self.step_label = tk.Label(
            stats_frame, text="STEPS:  0",
            bg="#1a1a2e", fg="#00e5ff",
            font=('Courier', 13, 'bold'),
            anchor='w', padx=6, pady=4,
            relief='flat'
        )
        self.step_label.pack(fill='x', pady=(0, 4))

        self.turn_label = tk.Label(
            stats_frame, text="TURNS:  0",
            bg="#1a1a2e", fg="#ff6b35",
            font=('Courier', 13, 'bold'),
            anchor='w', padx=6, pady=4,
            relief='flat'
        )
        self.turn_label.pack(fill='x')

        tk.Button(
            stats_frame, text="↺  RESET STATS",
            command=self.reset_counters,
            bg="#2d2d2d", fg="#888", font=('Arial', 8),
            relief='flat', cursor='hand2'
        ).pack(fill='x', pady=(6, 0))

        # ── Canvas ────────────────────────────────────────────────────────────
        self.canvas = tk.Canvas(self.root, bg=self.bg_color, highlightthickness=0)
        self.canvas.pack(side='right', expand=True, fill='both')
        self.canvas.bind("<Button-1>",        self.on_left_click)
        self.canvas.bind("<Button-3>",        self.on_right_click)
        self.canvas.bind("<B3-Motion>",       self.on_right_drag)
        self.canvas.bind("<ButtonRelease-3>", lambda e: setattr(self, 'last_vertex', None))

    # ── CLICK / DRAG HANDLERS ────────────────────────────────────────────────

    def on_left_click(self, event):
        if self.is_simulating: return
        cx, cy = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)

        if self.selected_tool == "DELETE":
            found = self.canvas.find_overlapping(cx-2, cy-2, cx+2, cy+2)
            for item in found:
                tags = self.canvas.gettags(item)
                if "grid" in tags: continue
                if "path" in tags:
                    self.subtract_path_stats()
                    self.undo_stack.append(("delete", [self.get_item_data(item)], item))
                    self.redo_stack.clear()                        # FIX B
                    self.canvas.delete(item)
                    return
                group_tag = next((t for t in tags if t.startswith("group_")), None)
                if group_tag:
                    items_to_del = self.canvas.find_withtag(group_tag)
                    c = self.canvas.coords(items_to_del[0])
                    # FIX C: compute proper grid position, not group_tag string
                    obj_pos = (
                        round(((c[0]+c[-2])/2) / self.grid_size) * self.grid_size,
                        round(((c[1]+c[-1])/2) / self.grid_size) * self.grid_size,
                    )
                    self.undo_stack.append(("delete",
                                            [self.get_item_data(i) for i in items_to_del],
                                            obj_pos))
                    self.redo_stack.clear()                        # FIX B
                    self.canvas.delete(group_tag)
                    keys_to_del = [k for k, v in self.occupancy_map.items() if v == group_tag]
                    for k in keys_to_del: del self.occupancy_map[k]
                    return
            return

        # Placement
        snap_x = round(cx / self.grid_size) * self.grid_size
        snap_y = round(cy / self.grid_size) * self.grid_size
        pos = (snap_x, snap_y)

        # Boundary check
        s = self.grid_size // 3
        if (snap_x - s < 0 or snap_x + s > self.cols * self.grid_size or
                snap_y - s < 0 or snap_y + s > self.rows * self.grid_size):
            return

        if self.selected_tool:
            if pos in self.occupancy_map: return        # FIX A: restore stacking guard
            obj_id = self.place_item(snap_x, snap_y, self.selected_tool, self.tool_color)
            if obj_id:
                self.occupancy_map[pos] = obj_id
                self.undo_stack.append(("add", obj_id, pos))
                self.redo_stack.clear()

    def place_item(self, x, y, tool, color):
        s = self.grid_size // 3
        g_tag = f"group_{x}_{y}_{tk.Tcl().call('clock', 'clicks')}"
        if tool == "BOT":
            self.canvas.create_oval(x-s, y-s, x+s, y+s, fill=color, outline="white",
                                    tags=("bot", "asset", g_tag, "body"))
            self.canvas.create_line(x, y, x+s, y, fill="yellow", width=2, arrow=tk.LAST,
                                    tags=("bot", "asset", g_tag, "heading"))
            return g_tag
        elif tool == "BLOCK":
            self.canvas.create_rectangle(x-s, y-s, x+s, y+s, fill=color, outline="white",
                                         tags=("block", "asset", g_tag))
            return g_tag
        elif tool == "GOAL":
            self.canvas.create_line(x-s, y-s, x+s, y+s, fill=color, width=3,
                                    tags=("goal", "asset", g_tag))
            self.canvas.create_line(x+s, y-s, x-s, y+s, fill=color, width=3,
                                    tags=("goal", "asset", g_tag))
            return g_tag
        elif "SUDO" in tool:
            tag = "sudo_bot" if "BOT" in tool else "sudo_block"
            shape = self.canvas.create_oval if "BOT" in tool else self.canvas.create_rectangle
            shape(x-s, y-s, x+s, y+s, fill="", outline=color, width=2, dash=(4, 4),
                  tags=(tag, "sudo", "asset", g_tag))
            return g_tag
        return None

    def on_right_click(self, event):
        cx, cy = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        # Right-click on a bot: cycle its orientation
        found = self.canvas.find_overlapping(cx-2, cy-2, cx+2, cy+2)
        for item in found:
            tags = self.canvas.gettags(item)
            if "bot" in tags:
                group_tag = next((t for t in tags if t.startswith("group_")), None)
                if group_tag:
                    self.cycle_bot_orientation(group_tag)
                    self.last_vertex = None
                    return
        sv = (round(cx / self.grid_size) * self.grid_size,
              round(cy / self.grid_size) * self.grid_size)
        if (0 <= sv[0] <= self.cols * self.grid_size and
                0 <= sv[1] <= self.rows * self.grid_size):
            self.last_vertex = sv
        else:
            self.last_vertex = None

    def on_right_drag(self, event):
        if self.last_vertex:
            cx, cy = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
            curr_v = (round(cx / self.grid_size) * self.grid_size,
                      round(cy / self.grid_size) * self.grid_size)
            if not (0 <= curr_v[0] <= self.cols * self.grid_size and
                    0 <= curr_v[1] <= self.rows * self.grid_size):
                return
            if curr_v != self.last_vertex:
                adx = abs(curr_v[0] - self.last_vertex[0])
                ady = abs(curr_v[1] - self.last_vertex[1])
                if (adx == self.grid_size and ady == 0) or (adx == 0 and ady == self.grid_size):
                    dx = (curr_v[0] - self.last_vertex[0]) / self.grid_size
                    dy = (curr_v[1] - self.last_vertex[1]) / self.grid_size
                    self.add_path_stats(dx, dy)
                    g_tag = f"group_arrow_{self.last_vertex[0]}_{self.last_vertex[1]}_{tk.Tcl().call('clock', 'clicks')}"
                    self.canvas.create_line(
                        self.last_vertex[0], self.last_vertex[1],
                        curr_v[0], curr_v[1],
                        fill="yellow", width=3, arrow=tk.LAST,
                        tags=("path", "asset", g_tag)
                    )
                    self.undo_stack.append(("add", g_tag, None))
                    self.redo_stack.clear()
                    self.last_vertex = curr_v

    def cycle_bot_orientation(self, bot_tag):
        items = self.canvas.find_withtag(bot_tag)
        h_id = next((i for i in items if "heading" in self.canvas.gettags(i)), None)
        b_id = next((i for i in items if "body" in self.canvas.gettags(i)), None)
        if h_id and b_id:
            c = self.canvas.coords(b_id)
            cx, cy = (c[0]+c[2])/2, (c[1]+c[3])/2
            h_c = self.canvas.coords(h_id)
            vx, vy = h_c[2]-h_c[0], h_c[3]-h_c[1]
            self.canvas.coords(h_id, cx, cy, cx-vy, cy+vx)
            logic01.reset_logic()

    # ── GRID / RESIZE ────────────────────────────────────────────────────────

    def open_resize_dialog(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Resize Arena")
        dialog.configure(bg="#2d2d2d")
        tk.Label(dialog, text="Columns:", fg="white", bg="#2d2d2d").grid(row=0, column=0, padx=10, pady=5)
        e_cols = tk.Entry(dialog); e_cols.insert(0, str(self.cols)); e_cols.grid(row=0, column=1, padx=10, pady=5)
        tk.Label(dialog, text="Rows:", fg="white", bg="#2d2d2d").grid(row=1, column=0, padx=10, pady=5)
        e_rows = tk.Entry(dialog); e_rows.insert(0, str(self.rows)); e_rows.grid(row=1, column=1, padx=10, pady=5)
        tk.Checkbutton(dialog, text="Fit to Screen", variable=self.fit_screen_var,
                       bg="#2d2d2d", fg="white", selectcolor="#2d2d2d",
                       activebackground="#2d2d2d").grid(row=2, columnspan=2, pady=5)
        tk.Button(dialog, text="APPLY", bg="#27ae60", fg="white",
                  command=lambda: self.apply_resize(e_rows.get(), e_cols.get(), dialog)
                  ).grid(row=3, columnspan=2, pady=10)

    def apply_resize(self, r, c, win):
        try:
            new_rows, new_cols = int(r), int(c)
            if new_rows < 1 or new_cols < 1: return
            if self.fit_screen_var.get():
                self.root.update_idletasks()
                canvas_w = self.canvas.winfo_width()
                canvas_h = self.canvas.winfo_height()
                if canvas_w <= 1: canvas_w = self.root.winfo_screenwidth() - 220
                if canvas_h <= 1: canvas_h = self.root.winfo_screenheight() - 60
                size_w = (canvas_w - 40) // new_cols
                size_h = (canvas_h - 40) // new_rows
                self.grid_size = max(10, min(size_w, size_h))
            self.rows, self.cols = new_rows, new_cols
            self.canvas.delete("all")
            self.draw_grid()
            self.occupancy_map.clear()
            self.undo_stack.clear()
            self.redo_stack.clear()
            self.reset_counters()
            win.destroy()
        except ValueError:
            messagebox.showerror("Error", "Please enter valid integers.")

    def draw_grid(self):
        self.canvas.delete("grid")
        w, h = self.cols * self.grid_size, self.rows * self.grid_size
        for i in range(self.cols + 1):
            self.canvas.create_line(i*self.grid_size, 0, i*self.grid_size, h,
                                    fill=self.grid_color, tags="grid")
        for j in range(self.rows + 1):
            self.canvas.create_line(0, j*self.grid_size, w, j*self.grid_size,
                                    fill=self.grid_color, tags="grid")
        self.canvas.tag_lower("grid")

    # ── SIMULATION RESET ─────────────────────────────────────────────────────

    def reset_simulation(self):
        if self.is_simulating:
            self.is_simulating = False
            self.play_btn.config(text="▶ PLAY LOGIC", bg="#27ae60")
        self.canvas.delete("all")
        self.draw_grid()
        self.occupancy_map.clear()
        self.reset_counters()
        for d in self.initial_state:
            new_id = self.recreate_item(d)
            tags = d.get('tags', [])
            if "asset" in tags and "path" not in tags:
                c = d['coords']
                center = (
                    round(((c[0]+c[-2])/2) / self.grid_size) * self.grid_size,
                    round(((c[1]+c[-1])/2) / self.grid_size) * self.grid_size,
                )
                self.occupancy_map[center] = new_id
        logic01.reset_logic()

    # ── SERIALISATION ────────────────────────────────────────────────────────

    def get_item_data(self, i):
        return {
            'type':   self.canvas.type(i),
            'coords': self.canvas.coords(i),
            'fill':   self.canvas.itemcget(i, "fill"),
            'width':  self.canvas.itemcget(i, "width"),
            'tags':   self.canvas.gettags(i),
        }

    def recreate_item(self, d):
        t, coords, fill, width, tags = d['type'], d['coords'], d['fill'], d['width'], d['tags']
        is_sudo  = "sudo" in tags
        is_arrow = "path" in tags or "heading" in tags
        dash     = (4, 4) if is_sudo else ()
        outline  = fill if is_sudo else "white"
        if t == 'line':
            return self.canvas.create_line(*coords, fill=fill, width=width,
                                           tags=tags, arrow=tk.LAST if is_arrow else "")
        elif t == 'oval':
            return self.canvas.create_oval(*coords, fill=fill, outline=outline,
                                           width=width, tags=tags, dash=dash)
        elif t == 'rectangle':
            return self.canvas.create_rectangle(*coords, fill=fill, outline=outline,
                                                width=width, tags=tags, dash=dash)

    # ── TOOL SELECTION ───────────────────────────────────────────────────────

    def set_tool(self, t, c): self.selected_tool, self.tool_color = t, c
    def set_delete_tool(self): self.selected_tool = "DELETE"


if __name__ == "__main__":
    root = tk.Tk()
    app = GridPlatform(root)
    root.mainloop()