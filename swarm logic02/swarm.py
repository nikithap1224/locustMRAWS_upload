import pygame
import numpy as np
from scipy.optimize import linear_sum_assignment as lsa
import random
import heapq
import time

W, H = 750, 450
CW = 30
PW = 300
GW, GH = 15, 15
FPS = 30

C_BG = (15, 15, 15)
C_LN = (40, 40, 40)
C_BT = (220, 220, 220)
C_RD = (200, 40, 40)
C_DR = (80, 40, 40)
C_TG = (255, 50, 50)
C_ED = (30, 30, 30)
C_WH = (255, 255, 255)
C_GR = (180, 180, 180)

class Bot:
    def __init__(self, id, x, y):
        self.id = id
        self.p = (x, y)
        self.path = []
        self.st = 0 
        self.tar = None
        self.ts = 0

class Obj:
    def __init__(self, id, x, y):
        self.id = id
        self.p = (x, y)
        self.got = False
        self.asn = False
        self.dst = None

class Edit:
    def __init__(self, x, y, sz):
        self.x, self.y, self.sz = x, y, sz
        self.g = [[False]*3 for _ in range(3)]
        self.r = []
        for i in range(3):
            for j in range(3):
                rect = pygame.Rect(x + i*sz, y + j*sz, sz, sz)
                self.r.append((rect, i, j))

    def show(self, sur):
        f = pygame.font.SysFont('Arial', 14, bold=True)
        lbl = f.render("PATTERN (3x3)", True, C_WH)
        sur.blit(lbl, (self.x, self.y - 20))
        for rect, i, j in self.r:
            c = C_RD if self.g[i][j] else C_ED
            pygame.draw.rect(sur, c, rect)
            pygame.draw.rect(sur, (100, 100, 100), rect, 1)

    def hit(self, pos):
        for rect, i, j in self.r:
            if rect.collidepoint(pos):
                self.g[i][j] = not self.g[i][j]
                return True
        return False

    def pts(self):
        res = []
        for i in range(3):
            for j in range(3):
                if self.g[i][j]:
                    res.append((i-1, j-1))
        return res

class Btn:
    def __init__(self, x, y, w, h, txt, fn):
        self.r = pygame.Rect(x, y, w, h)
        self.t = txt
        self.fn = fn

    def show(self, sur, mp):
        h = self.r.collidepoint(mp)
        c1 = C_GR if h else C_BG
        c2 = C_BG if h else C_WH
        pygame.draw.rect(sur, c1, self.r)
        pygame.draw.rect(sur, C_WH, self.r, 2)
        f = pygame.font.SysFont('Arial', 12, bold=True)
        txt = f.render(self.t, True, c2)
        rc = txt.get_rect(center=self.r.center)
        sur.blit(txt, rc)

    def hit(self, p):
        return self.r.collidepoint(p)

def dist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def nav(s, e, obs, w, h):
    q = []
    heapq.heappush(q, (0, s))
    par = {}
    g = {s: 0}
    f = {s: dist(s, e)}
    k = 0
    
    while q:
        k += 1
        if k > 3000: return []
        cur = heapq.heappop(q)[1]
        if cur == e:
            res = []
            while cur in par:
                res.append(cur)
                cur = par[cur]
            return res[::-1]
        
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            n = (cur[0] + dx, cur[1] + dy)
            if not (0 <= n[0] < w and 0 <= n[1] < h): continue
            if n in obs and n != e: continue
            
            tg = g[cur] + 1
            if n not in g or tg < g[n]:
                par[n] = cur
                g[n] = tg
                f[n] = tg + dist(n, e)
                heapq.heappush(q, (f[n], n))
    return []

class App:
    def __init__(self):
        pygame.init()
        self.scr = pygame.display.set_mode((W, H))
        pygame.display.set_caption("swarm bot")
        self.clk = pygame.time.Clock()
        self.fnt = pygame.font.SysFont('Arial', 16)
        
        ex = W - PW + (PW - 120) // 2
        self.ed = Edit(ex, 50, 40)
        
        bx = W - PW + 50
        bw = PW - 100
        self.btns = [
            Btn(bx, 200, bw, 40, "APPLY PATTERN", self.run_algo),
            Btn(bx, 350, bw, 40, "RESET", self.rst)
        ]
        
        self.rst()

    def rst(self):
        self.bots = [Bot(i, 0, i*2) for i in range(5)]
        self.objs = []
        u = set()
        while len(self.objs) < 9:
            rx = random.randint(0, GW - 1)
            ry = random.randint(0, GH - 1)
            if (rx, ry) not in u:
                self.objs.append(Obj(len(self.objs), rx, ry))
                u.add((rx, ry))
        self.tgs = []
        self.txt = "READY"
        print("\n--- SIMULATION RESET ---")

    def run_algo(self):
        rp = self.ed.pts()
        if not rp:
            self.txt = "EMPTY GRID"
            return
        
        cx, cy = GW // 2, GH // 2
        self.tgs = [(cx + p[0], cy + p[1]) for p in rp]
        self.txt = f"TASK: {len(self.tgs)} BLOCKS"
        
        print(f"\n[MANAGER] New Pattern Requested. Targets: {len(self.tgs)}")
        self.assign()

    def assign(self):
        for o in self.objs:
            o.dst = None
            if not o.got: o.asn = False

        if not self.tgs: return

        mx = np.zeros((len(self.objs), len(self.tgs)))
        for r, o in enumerate(self.objs):
            for c, t in enumerate(self.tgs):
                mx[r, c] = dist(o.p, t)
        
        print(f"[MATH] Cost Matrix (Blocks x Targets):\n{mx}")
        
        ri, ci = lsa(mx)
        
        print("[MATH] Optimal Assignment (Hungarian Algo):")
        total_cost = 0
        for i in range(len(ri)):
            o = self.objs[ri[i]]
            o.dst = self.tgs[ci[i]]
            cst = mx[ri[i], ci[i]]
            total_cost += cst
            print(f" -> Block {o.id} assigned to Target {o.dst} (Cost: {cst})")
        print(f"[MATH] Total Global Cost: {total_cost}")

        for b in self.bots:
            if b.st != 0: b.path = []

    def tick(self):
        idle = [b for b in self.bots if b.st == 0]
        req = [o for o in self.objs if o.dst and o.p != o.dst and not o.asn and not o.got]
        
        if idle and req:
            mx = np.zeros((len(idle), len(req)))
            for r, b in enumerate(idle):
                for c, o in enumerate(req):
                    mx[r, c] = dist(b.p, o.p)
            
            ri, ci = lsa(mx)
            for i in range(len(ri)):
                bot = idle[ri[i]]
                obj = req[ci[i]]
                bot.st = 1 
                bot.tar = obj
                bot.ts = time.time()
                obj.asn = True
                print(f"[BOT] Bot {bot.id} waking up to fetch Block {obj.id}")
                
                obs = {o.p for o in self.objs if not o.got and o != obj}
                bot.path = nav(bot.p, obj.p, obs, GW, GH)

        for b in self.bots:
            if not b.path and b.st != 0:
                gl = None
                if b.st == 1:
                    if b.tar.dst is None:
                        b.st = 0
                        b.tar.asn = False
                        b.tar = None
                        continue
                    gl = b.tar.p
                    obs = {o.p for o in self.objs if not o.got and o != b.tar}
                elif b.st == 2:
                    gl = b.tar.dst
                    if gl is None:
                        b.st = 0
                        b.tar.got = False
                        b.tar = None
                        continue
                    obs = {o.p for o in self.objs if o != b.tar}
                
                if gl: b.path = nav(b.p, gl, obs, GW, GH)

            if b.path:
                nxt = b.path.pop(0)
                b.p = nxt
                if b.tar and b.st == 2:
                    b.tar.p = nxt
            
            if b.st == 1 and b.p == b.tar.p:
                b.st = 2
                b.tar.got = True
                b.path = []
            elif b.st == 2 and b.p == b.tar.dst:
                b.st = 0
                b.tar.got = False
                b.tar.asn = False
                dur = time.time() - b.ts
                print(f"[STATS] Bot {b.id} finished task in {dur:.2f}s")
                b.tar = None

    def draw(self):
        self.scr.fill(C_BG)
        
        for x in range(0, GW * CW + 1, CW):
            pygame.draw.line(self.scr, C_LN, (x, 0), (x, GH * CW))
        for y in range(0, GH * CW + 1, CW):
            pygame.draw.line(self.scr, C_LN, (0, y), (GW * CW, y))

        for t in self.tgs:
            r = pygame.Rect(t[0]*CW, t[1]*CW, CW, CW)
            pygame.draw.rect(self.scr, C_TG, r, 2)

        for o in self.objs:
            r = pygame.Rect(o.p[0]*CW+4, o.p[1]*CW+4, CW-8, CW-8)
            c = C_RD if o.dst else C_DR
            pygame.draw.rect(self.scr, c, r)

        for b in self.bots:
            cx, cy = b.p[0]*CW + CW//2, b.p[1]*CW + CW//2
            pygame.draw.circle(self.scr, C_BT, (cx, cy), CW//2 - 2)
            if b.st == 2:
                 pygame.draw.circle(self.scr, C_RD, (cx, cy), 6)

        pygame.draw.rect(self.scr, C_BG, (W - PW, 0, PW, H))
        pygame.draw.line(self.scr, C_LN, (W - PW, 0), (W - PW, H))

        self.ed.show(self.scr)
        mp = pygame.mouse.get_pos()
        for b in self.btns: b.show(self.scr, mp)
        
        t = self.fnt.render(f"STATUS: {self.txt}", True, C_WH)
        self.scr.blit(t, (W - PW + 20, H - 50))

        pygame.display.flip()

    def loop(self):
        while True:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    pygame.quit()
                    return
                if e.type == pygame.MOUSEBUTTONDOWN:
                    p = pygame.mouse.get_pos()
                    if self.ed.hit(p): pass
                    for b in self.btns:
                        if b.hit(p): b.fn()
            
            self.tick()
            self.draw()
            self.clk.tick(FPS*0.1)

if __name__ == "__main__":
    a = App()
    a.loop()