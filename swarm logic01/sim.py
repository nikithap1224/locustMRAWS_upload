import tkinter as tk
from tkinter import messagebox

class GridPlatform:
    def __init__(self, root):
        self.root = root
        self.root.title("Grid Architect Pro - Undo/Redo Support")
        
        try: self.root.state('zoomed') 
        except: pass
            
        self.root.configure(bg="#1e1e1e")
        
        self.rows, self.cols = 20, 20
        self.grid_size = 40 
        self.bg_color, self.grid_color = "#1e1e1e", "#444444"
        
        self.undo_stack = []
        self.redo_stack = [] # New stack for redo actions
        
        self.selected_tool = None 
        self.tool_color = None
        self.selected_object = None  
        self.last_vertex = None
        self.drag_start_pos = None

        self.setup_ui()
        self.root.after(100, self.fit_to_screen_auto)
        self.bind_shortcuts()

    # ... (setup_ui, fit_to_screen_auto, open_resize_dialog, draw_grid remain the same)

    def setup_ui(self):
        self.sidebar = tk.Frame(self.root, width=180, bg="#2d2d2d", padx=10, pady=10)
        self.sidebar.pack(side='left', fill='y')
        self.sidebar.pack_propagate(False)

        tk.Label(self.sidebar, text="SYSTEM", fg="#888", bg="#2d2d2d", font=('Arial', 8, 'bold')).pack(pady=(0, 5))
        tk.Button(self.sidebar, text="Resize Grid", command=self.open_resize_dialog, bg="#444", fg="white").pack(fill='x', pady=2)
        
        self.tool_btns = {}
        self.del_btn = tk.Button(self.sidebar, text="🗑️ DELETE TOOL", command=self.set_delete_tool, bg="#e67e22", fg="white", font=('Arial', 9, 'bold'))
        self.del_btn.pack(fill='x', pady=5)
        self.tool_btns["DELETE"] = self.del_btn

        tk.Button(self.sidebar, text="CLEAR ALL", command=self.confirm_clear, bg="#c0392b", fg="white").pack(fill='x', pady=2)
        
        tk.Frame(self.sidebar, height=2, bg="#444").pack(fill='x', pady=10)
        
        tk.Label(self.sidebar, text="SOLID ASSETS", fg="white", bg="#2d2d2d", font=('Arial', 9, 'bold')).pack(pady=5)
        for name, col in [("BOT", "#3498db"), ("BLOCK", "#e74c3c"), ("GOAL", "#2ecc71")]:
            btn = tk.Button(self.sidebar, text=name, bg=col, fg="white", command=lambda n=name, c=col: self.set_tool(n, c))
            btn.pack(fill='x', pady=2)
            self.tool_btns[name] = btn

        tk.Frame(self.sidebar, height=1, bg="#444").pack(fill='x', pady=10)

        tk.Label(self.sidebar, text="SUDO ASSETS", fg="#aaa", bg="#2d2d2d", font=('Arial', 9, 'bold')).pack(pady=5)
        for name, col in [("SUDO BOT", "#3498db"), ("SUDO BLOCK", "#e74c3c")]:
            btn = tk.Button(self.sidebar, text=name, bg=col, fg="white", command=lambda n=name, c=col: self.set_tool(n, c))
            btn.pack(fill='x', pady=2)
            self.tool_btns[name] = btn

        self.status_label = tk.Label(self.sidebar, text="Tool: None", fg="#888", bg="#2d2d2d", font=('Arial', 9, 'italic'))
        self.status_label.pack(side='bottom', pady=20)

        self.container = tk.Frame(self.root, bg="#1e1e1e")
        self.container.pack(side='right', expand=True, fill='both')
        self.canvas = tk.Canvas(self.container, bg=self.bg_color, highlightthickness=0)
        self.canvas.pack(expand=True, fill='both')
        
        self.canvas.bind("<Button-1>", self.on_left_click)
        self.canvas.bind("<B1-Motion>", self.on_left_drag)
        self.canvas.bind("<Button-3>", self.on_right_click)
        self.canvas.bind("<B3-Motion>", self.on_right_drag)
        self.canvas.bind("<ButtonRelease-3>", self.on_right_release)

    def bind_shortcuts(self):
        # Undo Shortcuts
        self.root.bind("<Control-z>", lambda e: self.undo())
        self.root.bind("<Control-Z>", lambda e: self.undo())
        # Redo Shortcuts
        self.root.bind("<Control-y>", lambda e: self.redo())
        self.root.bind("<Control-Y>", lambda e: self.redo())
        self.root.bind("<Control-Shift-Z>", lambda e: self.redo())
        self.root.bind("<Control-Shift-z>", lambda e: self.redo())

    def add_to_undo(self, action_type, data):
        """Standard method to record actions and clear redo stack on new action."""
        self.undo_stack.append((action_type, data))
        self.redo_stack.clear() 

    def undo(self):
        if not self.undo_stack: return
        action = self.undo_stack.pop()
        
        if action[0] == "place":
            # Save data for redo before deleting
            data = self.get_item_data(action[1])
            self.canvas.delete(action[1])
            self.redo_stack.append(("delete_single", data))
        
        elif action[0] in ["delete_single", "delete_group"]:
            # Recreate items and save 'place' action for redo
            items = action[1] if action[0] == "delete_group" else [action[1]]
            new_ids = []
            for d in items:
                new_id = self.canvas._create(d['type'], d['coords'], {**d['opts'], 'tags': d['tags']})
                new_ids.append(new_id)
            
            # If it was a group, we store the group tag for the redo placement
            if action[0] == "delete_group":
                group_tag = [t for t in items[0]['tags'] if t.startswith("group_")][0]
                self.redo_stack.append(("place_group", group_tag))
            else:
                self.redo_stack.append(("place", new_ids[0]))

    def redo(self):
        if not self.redo_stack: return
        action = self.redo_stack.pop()
        
        if action[0] in ["place", "place_group"]:
            # Redoing a 'place' means we delete what was undeleted
            target = action[1]
            data = [self.get_item_data(i) for i in self.canvas.find_withtag(target)]
            self.canvas.delete(target)
            self.undo_stack.append(("delete_group" if len(data)>1 else "delete_single", data if len(data)>1 else data[0]))
            
        elif action[0] in ["delete_single", "delete_group"]:
            # Redoing a 'delete' means we recreate the item
            items = action[1] if action[0] == "delete_group" else [action[1]]
            new_ids = []
            for d in items:
                new_id = self.canvas._create(d['type'], d['coords'], {**d['opts'], 'tags': d['tags']})
                new_ids.append(new_id)
            
            if action[0] == "delete_group":
                group_tag = [t for t in items[0]['tags'] if t.startswith("group_")][0]
                self.undo_stack.append(("place", group_tag))
            else:
                self.undo_stack.append(("place", new_ids[0]))

    # ... (Rest of the interaction methods like on_left_click remain largely same, 
    # but use self.add_to_undo instead of manual appends)

    def on_left_click(self, event):
        cx, cy = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        snap_x, snap_y = round(cx/self.grid_size)*self.grid_size, round(cy/self.grid_size)*self.grid_size

        if self.selected_tool == "DELETE":
            items = self.canvas.find_overlapping(cx-5, cy-5, cx+5, cy+5)
            targets = [i for i in items if "grid" not in self.canvas.gettags(i)]
            if targets: self.record_and_delete(targets[-1])
            return

        if self.selected_tool:
            item_id = self.place_item(snap_x, snap_y, self.selected_tool, self.tool_color)
            self.add_to_undo("place", item_id)
            return

        items = self.canvas.find_overlapping(cx-10, cy-10, cx+10, cy+10)
        assets = [i for i in items if "asset" in self.canvas.gettags(i)]
        if assets:
            self.deselect_all()
            tags = self.canvas.gettags(assets[-1])
            group_tag = [t for t in tags if t.startswith("group_")]
            self.selected_object = group_tag[0] if group_tag else assets[-1]
            self.drag_start_pos = (snap_x, snap_y)
            self.highlight_selection(True)
        else: self.deselect_all()

    def record_and_delete(self, item_id):
        tags = self.canvas.gettags(item_id)
        group_tag = [t for t in tags if t.startswith("group_")]
        if group_tag:
            data = [self.get_item_data(tid) for tid in self.canvas.find_withtag(group_tag[0])]
            self.add_to_undo("delete_group", data)
            self.canvas.delete(group_tag[0])
        else:
            self.add_to_undo("delete_single", self.get_item_data(item_id))
            self.canvas.delete(item_id)

    def on_right_drag(self, event):
        if self.last_vertex:
            cx, cy = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
            curr_v = (round(cx/self.grid_size)*self.grid_size, round(cy/self.grid_size)*self.grid_size)
            if curr_v != self.last_vertex:
                dx, dy = abs(curr_v[0] - self.last_vertex[0]), abs(curr_v[1] - self.last_vertex[1])
                if (dx == self.grid_size and dy == 0) or (dx == 0 and dy == self.grid_size):
                    line_id = self.canvas.create_line(self.last_vertex[0], self.last_vertex[1], curr_v[0], curr_v[1], fill="yellow", width=5, arrow=tk.LAST, capstyle="round", tags=("path", "asset"))
                    self.add_to_undo("place", line_id)
                    self.last_vertex = curr_v

    # ... (Include all other helper methods from previous version to ensure script runs)
    def fit_to_screen_auto(self):
        self.root.update_idletasks()
        c_w, c_h = self.canvas.winfo_width(), self.canvas.winfo_height()
        self.grid_size = min(c_w // self.cols, c_h // self.rows)
        self.draw_grid()

    def open_resize_dialog(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Resize Grid"); dialog.geometry("250x220"); dialog.configure(bg="#2d2d2d")
        dialog.transient(self.root); dialog.grab_set()
        tk.Label(dialog, text="Columns:", fg="white", bg="#2d2d2d").pack(pady=(15,0))
        c_ent = tk.Entry(dialog, justify='center'); c_ent.insert(0, str(self.cols)); c_ent.pack()
        tk.Label(dialog, text="Rows:", fg="white", bg="#2d2d2d").pack(pady=(10,0))
        r_ent = tk.Entry(dialog, justify='center'); r_ent.insert(0, str(self.rows)); r_ent.pack()
        f_var = tk.BooleanVar(value=True)
        tk.Checkbutton(dialog, text="Fit to Screen", variable=f_var, bg="#2d2d2d", fg="white", selectcolor="#444").pack(pady=15)
        def apply():
            try:
                self.cols, self.rows = int(c_ent.get()), int(r_ent.get())
                if f_var.get(): self.fit_to_screen_auto()
                else: self.grid_size = 40; self.draw_grid()
                dialog.destroy()
            except: messagebox.showerror("Error", "Invalid entry")
        tk.Button(dialog, text="Apply Changes", command=apply, bg="#2ecc71", fg="white").pack(pady=5)

    def draw_grid(self):
        self.canvas.delete("all")
        w, h = self.cols * self.grid_size, self.rows * self.grid_size
        self.canvas.config(scrollregion=(0, 0, w, h))
        for i in range(self.cols + 1): self.canvas.create_line(i*self.grid_size, 0, i*self.grid_size, h, fill=self.grid_color, tags="grid")
        for j in range(self.rows + 1): self.canvas.create_line(0, j*self.grid_size, w, j*self.grid_size, fill=self.grid_color, tags="grid")
        self.canvas.tag_lower("grid")

    def move_object_relative(self, target, dx, dy):
        for i in self.canvas.find_withtag(target):
            c = self.canvas.coords(i)
            self.canvas.coords(i, *[v + (dx if k % 2 == 0 else dy) for k, v in enumerate(c)])

    def on_left_drag(self, event):
        if self.selected_object and not self.selected_tool:
            cx, cy = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
            snap_x, snap_y = round(cx/self.grid_size)*self.grid_size, round(cy/self.grid_size)*self.grid_size
            if self.drag_start_pos:
                dx, dy = snap_x - self.drag_start_pos[0], snap_y - self.drag_start_pos[1]
                if dx != 0 or dy != 0:
                    self.move_object_relative(self.selected_object, dx, dy)
                    self.drag_start_pos = (snap_x, snap_y)

    def highlight_selection(self, state):
        if not self.selected_object: return
        for i in self.canvas.find_withtag(self.selected_object):
            t = self.canvas.type(i)
            if t == 'line':
                c = "white" if state else ("yellow" if "path" in self.canvas.gettags(i) else self.canvas.itemcget(i, "fill"))
                self.canvas.itemconfig(i, width=5 if not state else 7, fill=c)
            else: self.canvas.itemconfig(i, width=4 if state else 2, outline="white")

    def deselect_all(self):
        if self.selected_object: self.highlight_selection(False); self.selected_object = None

    def get_item_data(self, item_id):
        obj_type = self.canvas.type(item_id)
        keys = ['fill', 'width', 'dash']; 
        if obj_type in ['oval', 'rectangle']: keys.append('outline')
        if obj_type == 'line': keys.append('arrow')
        opts = {k: self.canvas.itemcget(item_id, k) for k in keys if self.canvas.itemcget(item_id, k)}
        return {'type': obj_type, 'coords': self.canvas.coords(item_id), 'opts': opts, 'tags': self.canvas.gettags(item_id)}

    def on_right_click(self, event):
        cx, cy = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        self.last_vertex = (round(cx/self.grid_size)*self.grid_size, round(cy/self.grid_size)*self.grid_size)

    def on_right_release(self, event): self.last_vertex = None

    def place_item(self, x, y, tool, color):
        s = max(self.grid_size // 3, 4)
        if tool == "BOT": return self.canvas.create_oval(x-s, y-s, x+s, y+s, fill=color, outline="white", width=2, tags=("bot", "asset"))
        elif tool == "BLOCK": return self.canvas.create_rectangle(x-s, y-s, x+s, y+s, fill=color, outline="white", width=2, tags=("block", "asset"))
        elif tool == "GOAL":
            g = f"group_{x}_{y}"
            self.canvas.create_line(x-s, y-s, x+s, y+s, fill=color, width=4, tags=("goal", "asset", g))
            self.canvas.create_line(x+s, y-s, x-s, y+s, fill=color, width=4, tags=("goal", "asset", g))
            return g
        elif "SUDO" in tool:
            sh = self.canvas.create_oval if "BOT" in tool else self.canvas.create_rectangle
            return sh(x-s, y-s, x+s, y+s, fill="", outline=color, width=2, dash=(4,4), tags=("sudo", "asset"))

    def set_tool(self, t, c):
        if self.selected_tool == t: self.reset_tools()
        else: self.reset_tools(); self.selected_tool, self.tool_color = t, c; self.status_label.config(text=f"Active: {t}", fg=c); self.tool_btns[t].config(relief="sunken", borderwidth=3); self.deselect_all()
        
    def set_delete_tool(self):
        if self.selected_tool == "DELETE": self.reset_tools()
        else: self.reset_tools(); self.selected_tool = "DELETE"; self.canvas.config(cursor="X_cursor"); self.status_label.config(text="Active: DELETE", fg="#e67e22"); self.tool_btns["DELETE"].config(relief="sunken", borderwidth=3); self.deselect_all()

    def reset_tools(self):
        self.selected_tool = None; self.tool_color = None; self.canvas.config(cursor="arrow"); self.status_label.config(text="Tool: None", fg="#888")
        for btn in self.tool_btns.values(): btn.config(relief="raised", borderwidth=2)

    def confirm_clear(self): 
        if messagebox.askyesno("Clear", "Clear everything?"): self.draw_grid(); self.undo_stack.clear(); self.redo_stack.clear()

if __name__ == "__main__":
    root = tk.Tk()
    app = GridPlatform(root)
    root.mainloop()