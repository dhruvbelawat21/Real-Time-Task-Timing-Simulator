"""
Real-Time Task Timing Visualizer (Simulator)
Single-file Python implementation with Background Noise Generation.

Dependencies:
- Python 3.8+
- numpy
- matplotlib
- pandas (optional)
- tkinter (standard with Python)

UPDATES:
- Loads user app from app.py
- Generates logical background processes based on Device Type.
- FIX: Solved Matplotlib 'NoneType' crash by disabling blit.
- FIX: Solved GUI Dropdown text disappearing on selection.
"""

import json
import csv
import math
import random
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches
import statistics
import os
import datetime
import importlib.util
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import base64

# Try to import app.py if it exists
try:
    import app as user_app
    HAS_APP_FILE = True
except ImportError:
    HAS_APP_FILE = False

# ------------------------------
# Task & Job Models
# ------------------------------

@dataclass
class TaskSpec:
    """Specification for a task type (periodic or aperiodic)."""
    id: str
    wcet: int                   # worst-case execution time
    period: Optional[int] = None
    deadline: Optional[int] = None
    release: int = 0
    instances: Optional[int] = None
    color: Optional[str] = None
    type: str = "aperiodic"
    priority: Optional[int] = None
    jitter: int = 0
    dependencies: List[str] = field(default_factory=list)
    category: str = "Application"

@dataclass
class Job:
    """A specific job instance derived from a TaskSpec."""
    uid: int
    task_id: str
    release_time: int
    absolute_deadline: int
    exec_time: int
    remaining: int
    start_time: Optional[int] = None
    finish_time: Optional[int] = None
    response_time: Optional[int] = None
    spec: TaskSpec = None
    instance_no: int = 0
    dependencies: List[int] = field(default_factory=list)

# ------------------------------
# Scheduler Implementations
# ------------------------------

class SchedulerBase:
    preemptive: bool = True
    def __init__(self, name="BASE"): self.name = name
    def get_next_job(self, ready_jobs, current_time, running_job): raise NotImplementedError

class EDF(SchedulerBase):
    def __init__(self): super().__init__("EDF")
    def get_next_job(self, ready_jobs, current_time, running_job):
        if not ready_jobs: return None
        return min(ready_jobs, key=lambda j: j.absolute_deadline)

class RM(SchedulerBase):
    def __init__(self): super().__init__("RM")
    def get_next_job(self, ready_jobs, current_time, running_job):
        if not ready_jobs: return None
        return min(ready_jobs, key=lambda j: (math.inf if j.spec.period is None else j.spec.period, j.absolute_deadline))

class DM(SchedulerBase):
    def __init__(self): super().__init__("DM")
    def get_next_job(self, ready_jobs, current_time, running_job):
        if not ready_jobs: return None
        return min(ready_jobs, key=lambda j: ((j.absolute_deadline - j.release_time) if (j.absolute_deadline - j.release_time) > 0 else math.inf, j.absolute_deadline))

class LLF(SchedulerBase):
    def __init__(self): super().__init__("LLF")
    def get_next_job(self, ready_jobs, current_time, running_job):
        if not ready_jobs: return None
        def laxity(j): return (j.absolute_deadline - current_time - j.remaining)
        return min(ready_jobs, key=lambda j: (laxity(j), j.absolute_deadline))

class FCFS(SchedulerBase):
    def __init__(self): 
        super().__init__("FCFS")
        self.preemptive = False
    def get_next_job(self, ready_jobs, current_time, running_job):
        if not ready_jobs: return None
        return min(ready_jobs, key=lambda j: (j.release_time, j.uid))

class SRTF(SchedulerBase):
    def __init__(self): super().__init__("SRTF")
    def get_next_job(self, ready_jobs, current_time, running_job):
        if not ready_jobs: return None
        return min(ready_jobs, key=lambda j: (j.remaining, j.absolute_deadline))

class RR(SchedulerBase):
    def __init__(self, quantum=2):
        super().__init__("RR")
        self.quantum = quantum
        self._queue = []
        self._time_slice_left = 0
        self._current = None

    def get_next_job(self, ready_jobs, current_time, running_job):
        ready_set = {j.uid for j in ready_jobs}
        self._queue = [j for j in self._queue if j.uid in ready_set]
        existing_uids = {j.uid for j in self._queue}
        for j in sorted(ready_jobs, key=lambda x: (x.release_time, x.uid)):
            if j.uid not in existing_uids:
                self._queue.append(j); existing_uids.add(j.uid)

        if self._current and self._current.remaining > 0 and self._current.uid in existing_uids and self._time_slice_left > 0:
            return self._current

        if not self._queue:
            self._current = None; self._time_slice_left = 0; return None
        
        if self._current and self._current in self._queue:
            try: idx = self._queue.index(self._current)
            except ValueError: idx = None
            if idx == 0:
                self._queue.pop(0)
                if self._current.remaining > 0: self._queue.append(self._current)
        
        if not self._queue:
             self._current = None; self._time_slice_left = 0; return None
        self._current = self._queue[0]
        self._time_slice_left = self.quantum
        return self._current

    def consume_quantum(self, amount=1):
        self._time_slice_left = max(0, self._time_slice_left - amount)

# ------------------------------
# Background Generator
# ------------------------------

def get_background_tasks(device_type: str, sim_time: int) -> List[TaskSpec]:
    bg_tasks = []
    if device_type == "None": return []

    def make_bg(id, wcet, period, type="periodic", color="#555555", jitter=0):
        return TaskSpec(id=f"{id} (BG)", wcet=wcet, period=period, release=random.randint(0, 5),
                        type=type, category="Background", color=color, jitter=jitter)

    if device_type == "Android (Mobile)":
        bg_tasks.append(make_bg("SurfaceFlinger", 1, 8, color="#7f8c8d")) 
        bg_tasks.append(make_bg("System_Server", 2, 20, color="#95a5a6"))
        bg_tasks.append(make_bg("kswapd0", 1, 40, color="#bdc3c7"))
        bg_tasks.append(make_bg("Binder_IPC", 1, None, type="interrupt", release=15, color="#34495e"))
    elif device_type == "iOS (Mobile)":
        bg_tasks.append(make_bg("SpringBoard", 1, 10, color="#7f8c8d"))
        bg_tasks.append(make_bg("backboardd", 1, 6, color="#95a5a6"))
        bg_tasks.append(make_bg("kernel_task", 1, 25, color="#34495e"))
        bg_tasks.append(make_bg("mds_stores", 3, 60, color="#bdc3c7"))
    elif device_type == "Windows (Laptop)":
        bg_tasks.append(make_bg("dwm.exe", 1, 12, color="#2c3e50"))
        bg_tasks.append(make_bg("svchost.exe", 2, 30, color="#7f8c8d"))
        bg_tasks.append(make_bg("SysInterrupts", 1, 15, jitter=2, color="#bdc3c7"))
        bg_tasks.append(TaskSpec(id="MsMpEng (BG)", wcet=4, release=sim_time//2, type="aperiodic", category="Background", color="#555555"))
    elif device_type == "macOS (Laptop)":
        bg_tasks.append(make_bg("WindowServer", 1, 10, color="#2c3e50"))
        bg_tasks.append(make_bg("launchd", 1, 35, color="#7f8c8d"))
        bg_tasks.append(make_bg("mdworker", 2, 45, color="#bdc3c7"))
        bg_tasks.append(make_bg("kernel_task", 1, 15, color="#34495e"))

    return bg_tasks

# ------------------------------
# Simulator Core
# ------------------------------

class Simulator:
    def __init__(self, task_specs, scheduler, sim_time=200, preemptive=True, cores=1, energy_aware=False):
        self.specs = task_specs
        self.scheduler = scheduler
        self.sim_time = sim_time
        self.preemptive = preemptive
        self.cores = cores
        self.energy_aware = energy_aware
        self.power_active = 2.5
        self.power_idle = 0.5
        self.cpu_power_log = [[self.power_idle] * cores for _ in range(sim_time)] if energy_aware else []
        self.jobs = []
        self.timeline = [[None] * cores for _ in range(sim_time)]
        self.gantt_segments = {}
        self.job_records = {}
        self.uid_counter = 1

    def instantiate_job(self, spec, release_time, instance_no):
        rel = release_time
        if spec.jitter: rel = max(0, rel + random.randint(-spec.jitter, spec.jitter))
        
        if spec.deadline: abs_deadline = rel + spec.deadline
        elif spec.period: abs_deadline = rel + spec.period
        else: abs_deadline = rel + spec.wcet * 10 

        job = Job(uid=self.uid_counter, task_id=spec.id, release_time=rel, absolute_deadline=abs_deadline,
                  exec_time=spec.wcet, remaining=spec.wcet, spec=spec, instance_no=instance_no)
        self.uid_counter += 1
        self.jobs.append(job)
        self.job_records[job.uid] = job
        return job

    def run(self):
        # Reset
        self.jobs = []
        self.timeline = [[None] * self.cores for _ in range(self.sim_time)]
        self.gantt_segments = {}
        self.job_records = {}
        if self.energy_aware: self.cpu_power_log = [[self.power_idle] * self.cores for _ in range(self.sim_time)]

        # Prepare Jobs
        releases = []
        for spec in self.specs:
            if spec.type == "periodic" and spec.period:
                t = spec.release
                instances = spec.instances if spec.instances else 1000
                i = 1
                while t < self.sim_time and i <= instances:
                    releases.append((t, spec, i)); t += spec.period; i += 1
            else:
                releases.append((spec.release, spec, 1))
        releases.sort(key=lambda x: x[0])
        
        # Link deps
        instance_map = {}
        for (r, s, i) in releases:
            j = self.instantiate_job(s, r, i)
            instance_map[(s.id, i)] = j
        for j in self.jobs:
            for dep in j.spec.dependencies:
                dj = instance_map.get((dep, j.instance_no))
                if dj: j.dependencies.append(dj.uid)

        # Loop
        pending = []
        ready = []
        running = [None] * self.cores
        jobs_to_release = sorted(self.jobs, key=lambda j: j.release_time)
        rel_idx = 0

        for t in range(self.sim_time):
            # Release
            while rel_idx < len(jobs_to_release) and jobs_to_release[rel_idx].release_time <= t:
                pending.append(jobs_to_release[rel_idx])
                rel_idx += 1
            
            # Deps
            moved = []
            for j in pending:
                met = True
                for d_uid in j.dependencies:
                    d_job = self.job_records.get(d_uid)
                    if not d_job or d_job.finish_time is None: met = False; break
                if met: ready.append(j); moved.append(j)
            for j in moved: pending.remove(j)

            # Schedule
            for c in range(self.cores):
                curr = running[c]
                if curr and curr.remaining <= 0:
                    curr.finish_time = t
                    curr = None
                    running[c] = None
                
                # Candidates
                candidates = ready[:]
                if curr and curr not in candidates: candidates.append(curr)
                others = [running[x] for x in range(self.cores) if x != c and running[x]]
                candidates = [x for x in candidates if x not in others]

                next_job = self.scheduler.get_next_job(candidates, t, curr)

                if next_job != curr:
                    if curr and curr.remaining > 0 and curr not in ready: ready.append(curr)
                    running[c] = next_job
                    if next_job and next_job in ready: ready.remove(next_job)
                    if next_job and next_job.start_time is None: next_job.start_time = t

                # Execute
                if running[c]:
                    running[c].remaining -= 1
                    self.timeline[t][c] = running[c].uid
                    if self.energy_aware: self.cpu_power_log[t][c] = self.power_active
                    if isinstance(self.scheduler, RR): self.scheduler.consume_quantum()

            ready = [j for j in ready if j.remaining > 0]

        # Gantt segments
        uid_map = {j.uid: j.task_id for j in self.jobs}
        for c in range(self.cores):
            cur_uid = None
            start = 0
            for t in range(self.sim_time):
                uid = self.timeline[t][c]
                if uid != cur_uid:
                    if cur_uid: self.gantt_segments.setdefault(uid_map.get(cur_uid, "IDLE"), []).append((c, start, t))
                    cur_uid = uid; start = t
            if cur_uid: self.gantt_segments.setdefault(uid_map.get(cur_uid, "IDLE"), []).append((c, start, self.sim_time))
        
        return self._metrics()

    def _metrics(self):
        busy = sum(1 for t in range(self.sim_time) for c in range(self.cores) if self.timeline[t][c])
        total = self.sim_time * self.cores
        finished = [j for j in self.jobs if j.finish_time]
        missed = sum(1 for j in self.jobs if (j.finish_time and j.finish_time > j.absolute_deadline) or (not j.finish_time and j.absolute_deadline < self.sim_time))
        m = {"cpu_utilization": busy/total if total else 0, "missed_deadlines": missed, "total_jobs": len(self.jobs), "finished_jobs": len(finished)}
        if self.energy_aware:
            eng = sum(sum(self.cpu_power_log[t]) for t in range(self.sim_time))
            m["total_energy_j"] = eng; m["avg_power_w"] = eng/self.sim_time if self.sim_time else 0
        return m

# ------------------------------
# Visualization (Fixed)
# ------------------------------

def plot_gantt_and_animate(sim: Simulator, title: str = "Gantt"):
    app_task_ids = sorted(list(set(j.task_id for j in sim.jobs if j.spec.category == "Application")))
    bg_task_ids = sorted(list(set(j.task_id for j in sim.jobs if j.spec.category == "Background")))
    all_task_ids = app_task_ids + bg_task_ids
    task_indices = {tid: i for i, tid in enumerate(all_task_ids)}
    
    num_subplots = sim.cores + (1 if sim.energy_aware else 0)
    fig_height = 2 + sim.cores * (len(all_task_ids) * 0.4)
    if sim.energy_aware: fig_height += 2

    plt.style.use('dark_background')
    fig, axes = plt.subplots(num_subplots, 1, figsize=(12, fig_height), sharex=True, squeeze=False)
    fig.suptitle(title, fontsize=16, color="#00e5ff", fontweight='bold')

    # Color Logic
    cmap = plt.get_cmap('cool')
    app_colors = [cmap(i) for i in np.linspace(0.1, 0.9, 15)]
    
    def get_color(tid):
        spec = next((j.spec for j in sim.jobs if j.task_id == tid), None)
        if spec and spec.color: return spec.color
        if tid in app_task_ids: return app_colors[app_task_ids.index(tid) % len(app_colors)]
        return "#424242"

    # Bars
    for tid, segs in sim.gantt_segments.items():
        if tid == "IDLE": continue
        row = task_indices.get(tid)
        c = get_color(tid)
        for (core, s, e) in segs:
            axes[core][0].barh(row, e-s, left=s, height=0.6, color=c, edgecolor='black')

    # Axes Config
    for c in range(sim.cores):
        ax = axes[c][0]
        ax.set_yticks(range(len(all_task_ids)))
        ax.set_yticklabels(all_task_ids, color="white")
        ax.set_ylabel(f"CORE {c}", fontweight='bold', color="#00e5ff")
        ax.grid(True, axis='x', linestyle=':', alpha=0.2)
        if app_task_ids and bg_task_ids:
            ax.axhline(len(app_task_ids)-0.5, color='#00e5ff', linestyle='--', linewidth=0.5)

    # Deadlines
    cursors = []
    for job in sim.jobs:
        if 0 <= job.absolute_deadline <= sim.sim_time:
             for c in range(sim.cores):
                 axes[c][0].axvline(job.absolute_deadline, color='#ff3333', linestyle=':', alpha=0.6)
        if job.finish_time and job.finish_time > job.absolute_deadline:
             row = task_indices.get(job.task_id)
             finish_tick = min(job.finish_time-1, sim.sim_time-1)
             for c in range(sim.cores):
                 if sim.timeline[finish_tick][c] == job.uid:
                     axes[c][0].text(job.finish_time, row, "FAIL", color='red', fontsize=8)

    # Power
    if sim.energy_aware:
        p_ax = axes[-1][0]
        power_vals = [sum(sim.cpu_power_log[t]) for t in range(sim.sim_time)]
        p_ax.plot(power_vals, color='#00e5ff')
        p_ax.fill_between(range(sim.sim_time), power_vals, color='#00e5ff', alpha=0.2)
        p_ax.set_ylabel("Power (W)", color="#00e5ff")
        cursors.append(p_ax.axvline(0, color='white'))
    
    for c in range(sim.cores): cursors.append(axes[c][0].axvline(0, color='white'))

    # Animation
    # CRITICAL FIX: Attach text to an Axes, or just disable blit to avoid 'NoneType' crash.
    # We will disable blit for maximum compatibility.
    metrics_text = fig.text(0.02, 0.02, "", fontsize=10, color="#00e5ff", fontfamily="monospace")

    def update(frame):
        for c in cursors: c.set_xdata([frame])
        active = []
        for c in range(sim.cores):
            uid = sim.timeline[min(frame, sim.sim_time-1)][c]
            if uid: active.append(f"C{c}:{sim.job_records[uid].task_id}")
        
        status = f"T={frame} | RUN: {', '.join(active) if active else 'IDLE'}"
        if sim.energy_aware:
            eng = sum(sum(sim.cpu_power_log[t]) for t in range(frame))
            status += f" | ENG: {eng:.2f} J"
        metrics_text.set_text(status)
        return cursors + [metrics_text]

    # FIX: blit=False prevents the 'AttributeError: NoneType object has no attribute _get_view'
    ani = FuncAnimation(fig, update, frames=range(sim.sim_time+1), interval=100, blit=False)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()

# ------------------------------
# GUI (Fixed Style)
# ------------------------------

ICON_DATA = """
R0lGODlhIAAgAPcAAAAAAIAAAACAAICAAAAAgIAAgACAgMDAwMDcwKbK8P/w1Pjw1P/w8O7u7t7e
3t7i5u7i5ubm5iIiIiIzMzMzMzM3Nzc3NzdEREQiIiJEREQzMzMzRERERERmZmaAgIAiIiIzMzM3
Nzc3REREZmZmd3d3iIiImZmZzMzM/7u7/8zM///M/wAA/wAz/wA3/wBE/wBm/wB3/wiI/wCZ/wDM
/wD//zMzADM3ADNEADNmADODADPMADP/ADNmMzN3MziIMzOZMzPMMzP/M0REREdmZneIiJmZmdzc
3P///wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAACH5BAEAAAAALAAAAAAgACAAAAikAAEIHEiwoMGDCBMqXMiwocOHECNK
nEixosWLGDNq3Mixo8ePIEOKHEmypMmTKFOqXMmypcuXMGPKnEmzps2bOHPq3Mmzp8+fQIMKHUq0
qNGjSJMqXcq0qdOnUKNKnUq1qtWrWLNq3cq1q9evYMOKHUu2rNmzaNOqXcu2rdu3cOPKnUu3rt27
ePPq3cu3r9+/gAMLHky4sOHDiBMrXsy4sePHkCNLnky5suXLmDNr3sy5s+fPoEOLHk26tOnTqFOr
Xs26tevXsGPLnk27tu3buHPr3s27t+/fwIMLH068uPHjyJMrX868ufPn0KNLn069uvXr2LNr3869
u/fv4MOLH0++vPnz6NOrX8++vfv38OPLn0+/vv37+PPr38+/v///AAYo4IAEFmjggQgmqOCCDDbo
4IMQRijhhBRWaOGFGGao4YYcdujhhyCGKOKIJJZo4okopqjiiiy26OKLMMYo44w01mjjjTjmqOOO
PPbo449ABinkkEQWaeSRSCap5JJMNunkk1BGKeWUVFZp5ZVYZqnlllx26eWXYIYp5phklmnmmWim
qeaabLbp5ptwxinnnHTWaeedeOap55589unnn4AGKuighBZq6KGIJqrooow26uijkEYq6aSUVmrppZhm
qummnHbq6aeghirqqKSWauqpqKaq6qqsturqq7DGKuustNZq66245qrrrrz26uuvwAYr7LDEFmvsscgmq+yyzDbr7LPQRivttNRWa+212Gar7bbcduvtt+CGK+645JZr7rnopqvuuuy26+678MYr77z01mvvvfjmq+++/Pbr778AByzwwAQXbPDBCCes8MIMN+zwwxBHLPHEFFds8cUYZ6zxxhx37PHHIIcs8sgkl2zyySinrPLKLLfs8sswxyzzzDTXbPPNOOes88489+zzz0AHLfTQRBdt9NFIJ6300kw37fTTUEct9dRUV2311VhnrfXWXHft9ddghy322GSXbfbZaKet9tpst+3223DHLffcdNdt991456333nz37fffgAcueN9AA7AAAA7
"""

def gui_config_menu():
    COLOR_BG, COLOR_PANEL, COLOR_ACCENT = "#121212", "#1E1E1E", "#00E5FF"
    COLOR_TEXT, COLOR_BUTTON = "#E0E0E0", "#0069D9"
    
    root = tk.Tk()
    root.title("RT-SIM CONFIG")
    root.geometry("500x750")
    root.configure(bg=COLOR_BG)

    try:
        icon_bytes = base64.b64decode(ICON_DATA.strip())
        root.iconphoto(False, tk.PhotoImage(data=icon_bytes))
    except: pass

    style = ttk.Style()
    style.theme_use('clam')
    style.configure("Dark.TFrame", background=COLOR_BG)
    style.configure("Panel.TFrame", background=COLOR_PANEL, relief="flat")
    style.configure("Dark.TLabel", background=COLOR_PANEL, foreground=COLOR_TEXT)
    style.configure("Header.TLabel", background=COLOR_BG, foreground=COLOR_ACCENT, font=("Segoe UI", 16, "bold"))
    style.configure("Section.TLabel", background=COLOR_PANEL, foreground=COLOR_ACCENT, font=("Segoe UI", 10, "bold"))
    style.configure("Dark.TCheckbutton", background=COLOR_PANEL, foreground=COLOR_TEXT)
    style.map("Dark.TCheckbutton", background=[('active', COLOR_PANEL)], indicatorcolor=[('selected', COLOR_ACCENT)])

    # FIX: Explicit style map for Combobox to prevent text vanishing on focus loss
    style.configure("Dark.TCombobox", fieldbackground=COLOR_PANEL, background=COLOR_PANEL, foreground=COLOR_TEXT, arrowcolor=COLOR_ACCENT)
    style.map("Dark.TCombobox", 
              fieldbackground=[('readonly', COLOR_PANEL), ('!disabled', COLOR_PANEL)],
              foreground=[('readonly', COLOR_TEXT), ('!disabled', COLOR_TEXT)],
              selectbackground=[('readonly', COLOR_PANEL)],
              selectforeground=[('readonly', COLOR_ACCENT)])

    # Vars
    vars = {
        "scheduler": tk.StringVar(value="EDF"),
        "preemption": tk.BooleanVar(value=True),
        "cores": tk.IntVar(value=1),
        "sim_time": tk.IntVar(value=50),
        "device": tk.StringVar(value="Android (Mobile)"),
        "energy": tk.BooleanVar(value=True),
        "file_path": tk.StringVar(value=""),
        "file_type": tk.StringVar(value="app.py")
    }

    # Layout
    header = ttk.Frame(root, style="Dark.TFrame"); header.pack(fill="x", pady=20, padx=20)
    ttk.Label(header, text="REAL-TIME\nSIMULATOR", style="Header.TLabel").pack(side="left")

    main = ttk.Frame(root, style="Dark.TFrame"); main.pack(fill="both", expand=True, padx=20)
    
    # Kernel
    p1 = ttk.Frame(main, style="Panel.TFrame", padding=15); p1.pack(fill="x", pady=10)
    ttk.Label(p1, text="KERNEL SETTINGS", style="Section.TLabel").pack(anchor="w", pady=(0,5))
    g1 = ttk.Frame(p1, style="Panel.TFrame"); g1.pack(fill="x")
    ttk.Label(g1, text="Algorithm:", style="Dark.TLabel").grid(row=0, column=0, sticky="w", pady=5)
    ttk.Combobox(g1, textvariable=vars["scheduler"], values=["EDF", "RM", "DM", "LLF", "FCFS", "RR"], state="readonly", style="Dark.TCombobox").grid(row=0, column=1, sticky="e", pady=5)
    ttk.Label(g1, text="Cores:", style="Dark.TLabel").grid(row=1, column=0, sticky="w", pady=5)
    tk.Spinbox(g1, from_=1, to=8, textvariable=vars["cores"], width=5, bg=COLOR_BG, fg=COLOR_TEXT, relief="flat", buttonbackground=COLOR_PANEL).grid(row=1, column=1, sticky="e", pady=5)
    ttk.Checkbutton(p1, text="Preemptive", variable=vars["preemption"], style="Dark.TCheckbutton").pack(anchor="w")

    # Hardware
    p2 = ttk.Frame(main, style="Panel.TFrame", padding=15); p2.pack(fill="x", pady=10)
    ttk.Label(p2, text="HARDWARE PROFILE", style="Section.TLabel").pack(anchor="w", pady=(0,5))
    ttk.Label(p2, text="Device Simulation:", style="Dark.TLabel").pack(anchor="w")
    ttk.Combobox(p2, textvariable=vars["device"], values=["None", "Android (Mobile)", "iOS (Mobile)", "Windows (Laptop)", "macOS (Laptop)"], state="readonly", width=35, style="Dark.TCombobox").pack(fill="x", pady=5)
    g2 = ttk.Frame(p2, style="Panel.TFrame"); g2.pack(fill="x")
    ttk.Label(g2, text="Sim Duration:", style="Dark.TLabel").grid(row=0, column=0, sticky="w")
    tk.Entry(g2, textvariable=vars["sim_time"], width=10, bg=COLOR_BG, fg=COLOR_ACCENT, relief="flat", insertbackground="white").grid(row=0, column=1, sticky="e")
    ttk.Checkbutton(p2, text="Track Energy", variable=vars["energy"], style="Dark.TCheckbutton").pack(anchor="w", pady=5)

    # Source
    p3 = ttk.Frame(main, style="Panel.TFrame", padding=15); p3.pack(fill="x", pady=10)
    ttk.Label(p3, text="SOURCE", style="Section.TLabel").pack(anchor="w")
    status = tk.Frame(p3, bg=COLOR_PANEL); status.pack(fill="x", pady=5)
    
    if HAS_APP_FILE:
        tk.Label(status, text="●", fg="#00FF00", bg=COLOR_PANEL).pack(side="left")
        tk.Label(status, text="app.py detected", fg=COLOR_TEXT, bg=COLOR_PANEL).pack(side="left")
    else:
        tk.Label(status, text="●", fg="red", bg=COLOR_PANEL).pack(side="left")
        tk.Label(status, text="No app.py found", fg=COLOR_TEXT, bg=COLOR_PANEL).pack(side="left")

    def load_ext():
        fp = filedialog.askopenfilename(filetypes=[("JSON/CSV", "*.json *.csv")])
        if fp: vars["file_path"].set(fp); vars["file_type"].set(fp.split('.')[-1])

    tk.Button(p3, text="LOAD FILE", command=load_ext, bg=COLOR_PANEL, fg=COLOR_ACCENT, relief="groove", borderwidth=1).pack(fill="x")

    def submit(): root.destroy()
    tk.Button(root, text="START SIMULATION", command=submit, bg=COLOR_BUTTON, fg="white", font=("Segoe UI", 12, "bold"), relief="flat", pady=10).pack(fill="x", side="bottom")
    
    # Center
    root.update_idletasks()
    w, h = root.winfo_width(), root.winfo_height()
    x, y = (root.winfo_screenwidth()//2)-(w//2), (root.winfo_screenheight()//2)-(h//2)
    root.geometry(f'{w}x{h}+{x}+{y}')
    
    root.mainloop()
    return {k: v.get() for k, v in vars.items()}

def load_tasks_from_dict(data):
    return [TaskSpec(id=x.get("id"), wcet=int(x.get("wcet")), period=x.get("period"), 
            deadline=x.get("deadline"), release=int(x.get("release", 0)), 
            type=x.get("type", "aperiodic"), color=x.get("color"), category="Application",
            dependencies=x.get("dependencies", [])) for x in data]

def main():
    config = gui_config_menu()
    if not config["scheduler"]: return

    app_specs = []
    if config["file_path"]:
        if config["file_type"] == "json":
             with open(config["file_path"]) as f: app_specs = load_tasks_from_dict(json.load(f))
    elif HAS_APP_FILE:
        app_specs = load_tasks_from_dict(user_app.get_user_application_tasks())
    else:
        app_specs = [TaskSpec("DummyApp", 2, 10, category="Application", type="periodic")]

    bg_specs = get_background_tasks(config["device"], config["sim_time"])
    all_specs = app_specs + bg_specs
    
    print(f"Tasks: {len(all_specs)} ({len(app_specs)} App, {len(bg_specs)} BG)")
    
    sched_map = {"EDF": EDF, "RM": RM, "DM": DM, "LLF": LLF, "FCFS": FCFS, "RR": RR}
    sim = Simulator(all_specs, sched_map.get(config["scheduler"], EDF)(), 
                    config["sim_time"], config["preemption"], config["cores"], config["energy"])
    
    metrics = sim.run()
    for k, v in metrics.items(): print(f"{k}: {v}")
    
    plot_gantt_and_animate(sim, f"{config['device']} | {config['scheduler']}")

if __name__ == "__main__":
    main()