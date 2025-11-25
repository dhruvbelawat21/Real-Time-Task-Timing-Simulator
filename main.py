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
- Generates logical background processes based on Device Type (Android/iOS/Windows/Mac).
- Visual distinction between App and Background tasks.
- NEW: Modern Dark/Blue GUI for Input Configuration.
- FIXED: Matplotlib Colormap AttributeError.
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
    wcet: int                   # worst-case execution time (time units)
    period: Optional[int] = None   # if periodic, the period (and implicit deadline)
    deadline: Optional[int] = None # relative deadline (if None and periodic -> equals period)
    release: int = 0               # initial release time
    instances: Optional[int] = None # number of instances for periodic tasks (None => run until sim end)
    color: Optional[str] = None
    type: str = "aperiodic"        # "periodic" or "aperiodic" or "interrupt"
    priority: Optional[int] = None # fixed priority
    jitter: int = 0                # max jitter added to release
    dependencies: List[str] = field(default_factory=list)
    category: str = "Application"  # NEW: "Application" or "Background"

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
# Scheduler base + implementations
# ------------------------------

class SchedulerBase:
    preemptive: bool = True
    def __init__(self, name="BASE"): self.name = name
    def add_job(self, job: Job): pass
    def remove_job(self, job: Job): pass
    def get_next_job(self, ready_jobs: List[Job], current_time: int, running_job: Optional[Job]) -> Optional[Job]: raise NotImplementedError

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
        self._queue: List[Job] = []
        self._time_slice_left = 0
        self._current: Optional[Job] = None

    def get_next_job(self, ready_jobs, current_time, running_job):
        ready_set = {j.uid for j in ready_jobs}
        self._queue = [j for j in self._queue if j.uid in ready_set]
        existing_uids = {j.uid for j in self._queue}
        for j in sorted(ready_jobs, key=lambda x: (x.release_time, x.uid)):
            if j.uid not in existing_uids:
                self._queue.append(j)
                existing_uids.add(j.uid)

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
    """
    Generates synthetic background processes based on the selected device.
    """
    bg_tasks = []
    
    if device_type == "None":
        return []

    # --- Common Helpers ---
    def make_bg(id, wcet, period, type="periodic", color="#555555", jitter=0):
        return TaskSpec(
            id=f"{id} (BG)",
            wcet=wcet,
            period=period,
            release=random.randint(0, 5), # Random start offset
            type=type,
            category="Background",
            color=color,
            jitter=jitter
        )

    if device_type == "Android (Mobile)":
        bg_tasks.append(make_bg("SurfaceFlinger", wcet=1, period=8, color="#7f8c8d")) 
        bg_tasks.append(make_bg("System_Server", wcet=2, period=20, color="#95a5a6"))
        bg_tasks.append(make_bg("kswapd0", wcet=1, period=40, color="#bdc3c7"))
        bg_tasks.append(make_bg("Binder_IPC", wcet=1, period=None, type="interrupt", release=15, color="#34495e"))

    elif device_type == "iOS (Mobile)":
        bg_tasks.append(make_bg("SpringBoard", wcet=1, period=10, color="#7f8c8d"))
        bg_tasks.append(make_bg("backboardd", wcet=1, period=6, color="#95a5a6"))
        bg_tasks.append(make_bg("kernel_task", wcet=1, period=25, color="#34495e"))
        bg_tasks.append(make_bg("mds_stores", wcet=3, period=60, color="#bdc3c7"))

    elif device_type == "Windows (Laptop)":
        bg_tasks.append(make_bg("dwm.exe", wcet=1, period=12, color="#2c3e50"))
        bg_tasks.append(make_bg("svchost.exe", wcet=2, period=30, color="#7f8c8d"))
        bg_tasks.append(make_bg("SysInterrupts", wcet=1, period=15, jitter=2, color="#bdc3c7"))
        bg_tasks.append(TaskSpec(id="MsMpEng (BG)", wcet=4, release=sim_time//2, type="aperiodic", category="Background", color="#555555"))

    elif device_type == "macOS (Laptop)":
        bg_tasks.append(make_bg("WindowServer", wcet=1, period=10, color="#2c3e50"))
        bg_tasks.append(make_bg("launchd", wcet=1, period=35, color="#7f8c8d"))
        bg_tasks.append(make_bg("mdworker", wcet=2, period=45, color="#bdc3c7"))
        bg_tasks.append(make_bg("kernel_task", wcet=1, period=15, color="#34495e"))

    return bg_tasks

# ------------------------------
# Simulator core
# ------------------------------

class Simulator:
    def __init__(self, task_specs: List[TaskSpec], scheduler: SchedulerBase, sim_time: int = 200, 
                 preemptive: bool = True, cores: int = 1, energy_aware: bool = False, 
                 power_active: float = 2.5, power_idle: float = 0.5):
        self.specs = task_specs
        self.scheduler = scheduler
        self.sim_time = sim_time
        self.preemptive = preemptive
        self.cores = cores
        
        # Energy
        self.energy_aware = energy_aware
        self.power_active = power_active
        self.power_idle = power_idle
        self.cpu_power_log = [[self.power_idle] * cores for _ in range(sim_time)] if energy_aware else []

        # State
        self.jobs: List[Job] = []
        self.pending_jobs: List[Job] = []
        self.jobs_to_release: List[Job] = []
        self.ready: List[Job] = []
        self.running: List[Optional[Job]] = [None] * cores 
        self.timeline: List[List[Optional[int]]] = [[None] * cores for _ in range(sim_time)]
        self.gantt_segments: Dict[str, List[Tuple[int, int, int]]] = {} 
        self.job_records: Dict[int, Job] = {}
        self.uid_counter = 1

    def instantiate_job(self, spec: TaskSpec, release_time: int, instance_no: int) -> Job:
        rel = release_time
        if spec.jitter:
            rel += random.randint(-spec.jitter, spec.jitter)
            rel = max(rel, 0)
        rel = int(rel)

        if spec.deadline is not None:
            abs_deadline = rel + spec.deadline
        elif spec.period is not None:
            abs_deadline = rel + spec.period
        else:
            abs_deadline = rel + spec.wcet * 10 

        job = Job(
            uid=self.uid_counter, task_id=spec.id, release_time=rel, absolute_deadline=abs_deadline,
            exec_time=spec.wcet, remaining=spec.wcet, spec=spec, instance_no=instance_no,
        )
        self.uid_counter += 1
        self.jobs.append(job)
        self.job_records[job.uid] = job
        return job

    def prepare_job_releases(self):
        releases = []
        for spec in self.specs:
            if spec.type == "periodic" and spec.period is not None:
                instances = spec.instances if spec.instances is not None else math.inf
                t = spec.release
                i = 1
                while t < self.sim_time and i <= instances:
                    releases.append((t, spec, i))
                    t += spec.period
                    i += 1
            elif spec.type in ("aperiodic", "interrupt"):
                releases.append((spec.release, spec, 1))
        releases.sort(key=lambda x: x[0])
        return releases

    def _pre_instantiate_and_link(self):
        releases = self.prepare_job_releases()
        instance_map: Dict[Tuple[str, int], Job] = {} 

        for (rel_time, spec, inst_no) in releases:
            job = self.instantiate_job(spec, rel_time, inst_no)
            instance_map[(spec.id, inst_no)] = job

        for job in self.jobs:
            if not job.spec.dependencies: continue
            for dep_task_id in job.spec.dependencies:
                dep_job = instance_map.get((dep_task_id, job.instance_no))
                if dep_job: job.dependencies.append(dep_job.uid)
        
        self.jobs_to_release = sorted(self.jobs, key=lambda j: j.release_time)

    def _dependencies_met(self, job: Job) -> bool:
        if not job.dependencies: return True
        for dep_uid in job.dependencies:
            dep_job = self.job_records.get(dep_uid)
            if not dep_job or dep_job.finish_time is None: return False
        return True

    def run(self):
        self.jobs, self.pending_jobs, self.jobs_to_release, self.ready = [], [], [], []
        self.running = [None] * self.cores
        self.timeline = [[None] * self.cores for _ in range(self.sim_time)]
        self.gantt_segments, self.job_records, self.uid_counter = {}, {}, 1
        if self.energy_aware: self.cpu_power_log = [[self.power_idle] * self.cores for _ in range(self.sim_time)]
        
        self._pre_instantiate_and_link()
        release_idx = 0
        
        for t in range(self.sim_time):
            # 1. Release
            while release_idx < len(self.jobs_to_release) and self.jobs_to_release[release_idx].release_time <= t:
                self.pending_jobs.append(self.jobs_to_release[release_idx])
                release_idx += 1
            
            # 2. Check deps
            newly_ready = []
            for job in self.pending_jobs:
                if self._dependencies_met(job):
                    self.ready.append(job)
                    newly_ready.append(job)
            if newly_ready: self.pending_jobs = [j for j in self.pending_jobs if j not in newly_ready]

            # 3. Schedule per core
            for core_id in range(self.cores):
                current = self.running[core_id]
                if current and current.remaining <= 0:
                    current.finish_time = t
                    current.response_time = current.finish_time - current.release_time
                    current = None
                    self.running[core_id] = None

                available_jobs = self.ready[:]
                if current and current not in available_jobs: available_jobs.append(current)
                # Exclude jobs on other cores
                other_running = [self.running[c] for c in range(self.cores) if c != core_id and self.running[c] is not None]
                available_jobs = [j for j in available_jobs if j not in other_running]

                candidate = self.scheduler.get_next_job(available_jobs, t, current)

                if candidate is None:
                    if current:
                        if current.remaining > 0 and current not in self.ready: self.ready.append(current)
                    self.running[core_id] = None
                    self.timeline[t][core_id] = None
                    if self.energy_aware: self.cpu_power_log[t][core_id] = self.power_idle
                    continue

                if current is None:
                    self.running[core_id] = candidate
                    if candidate.start_time is None: candidate.start_time = t
                    if candidate in self.ready: self.ready.remove(candidate)
                elif candidate.uid != current.uid:
                    if self.preemptive and self.scheduler.preemptive:
                        if current.remaining > 0 and current not in self.ready: self.ready.append(current)
                        self.running[core_id] = candidate
                        if candidate.start_time is None: candidate.start_time = t
                        if candidate in self.ready: self.ready.remove(candidate)
                    else:
                        candidate = current
                        self.running[core_id] = current

                # Execute
                job_to_run = self.running[core_id]
                if job_to_run:
                    job_to_run.remaining -= 1
                    self.timeline[t][core_id] = job_to_run.uid
                    if self.energy_aware: self.cpu_power_log[t][core_id] = self.power_active
                    if isinstance(self.scheduler, RR): self.scheduler.consume_quantum(1)
                    if job_to_run.remaining <= 0:
                        job_to_run.finish_time = t + 1
                        job_to_run.response_time = job_to_run.finish_time - job_to_run.release_time
                        self.running[core_id] = None
                else:
                    if self.energy_aware: self.cpu_power_log[t][core_id] = self.power_idle

            self.ready = [j for j in self.ready if j.remaining > 0]

        # Post-processing for Gantt
        uid_to_task = {j.uid: j.task_id for j in self.jobs}
        for core_id in range(self.cores):
            current_uid = None
            seg_start = 0
            for t in range(self.sim_time):
                uid = self.timeline[t][core_id]
                if uid != current_uid:
                    if current_uid is not None:
                        task_id = uid_to_task.get(current_uid, "IDLE")
                        self.gantt_segments.setdefault(task_id, []).append((core_id, seg_start, t))
                    current_uid = uid
                    seg_start = t
            if current_uid is not None:
                task_id = uid_to_task.get(current_uid, "IDLE")
                self.gantt_segments.setdefault(task_id, []).append((core_id, seg_start, self.sim_time))

        return self._compute_metrics()

    def _compute_metrics(self):
        busy = sum(1 for t in range(self.sim_time) for c in range(self.cores) if self.timeline[t][c] is not None)
        total_core_time = (self.sim_time * self.cores)
        cpu_util = busy / total_core_time if total_core_time > 0 else 0.0
        finished = [j for j in self.jobs if j.finish_time is not None]
        missed = sum(1 for j in self.jobs if (j.finish_time and j.finish_time > j.absolute_deadline) or (j.finish_time is None and j.absolute_deadline < self.sim_time))
        
        metrics = {
            "cpu_utilization": cpu_util,
            "missed_deadlines": missed,
            "total_jobs": len(self.jobs),
            "finished_jobs": len(finished),
        }
        if self.energy_aware:
            total_energy = sum(sum(self.cpu_power_log[t]) for t in range(self.sim_time))
            metrics["total_energy_j"] = total_energy
            metrics["avg_power_w"] = total_energy / self.sim_time if self.sim_time > 0 else 0
        return metrics

# ------------------------------
# Visualization
# ------------------------------

def plot_gantt_and_animate(sim: Simulator, title: str = "Gantt"):
    # Separate tasks by category for easier reading
    app_task_ids = sorted(list(set(j.task_id for j in sim.jobs if j.spec.category == "Application")))
    bg_task_ids = sorted(list(set(j.task_id for j in sim.jobs if j.spec.category == "Background")))
    
    # All tasks combined, Apps on top
    all_task_ids = app_task_ids + bg_task_ids
    
    # Map tasks to row indices (inverted so 0 is top)
    task_indices = {tid: i for i, tid in enumerate(all_task_ids)}
    
    num_gantt_subplots = sim.cores
    num_total = sim.cores + (1 if sim.energy_aware else 0)
    fig_height = 2 + num_gantt_subplots * (len(all_task_ids) * 0.4)
    if sim.energy_aware: fig_height += 2

    # Matplotlib Dark Theme adjustment for plot
    plt.style.use('dark_background')
    
    fig, axes = plt.subplots(num_total, 1, figsize=(12, fig_height), sharex=True, squeeze=False)
    fig.suptitle(title, fontsize=16, color="#00e5ff", fontweight='bold')

    # Color logic - FIXED to avoid AttributeError
    # Sample the 'cool' colormap manually to get a list of colors
    cmap = plt.get_cmap('cool')
    # Generate 15 sample colors from the cool (blue-cyan-magenta) spectrum
    app_colors = [cmap(i) for i in np.linspace(0.1, 0.9, 15)]
    
    def get_color(task_id):
        spec = next((j.spec for j in sim.jobs if j.task_id == task_id), None)
        if spec and spec.color: return spec.color
        
        if task_id in app_task_ids:
            idx = app_task_ids.index(task_id)
            return app_colors[idx % len(app_colors)]
        elif task_id in bg_task_ids:
            return "#424242" # Dark Grey for background
        return "grey"

    # Draw initial bars
    for tid, segs in sim.gantt_segments.items():
        if tid == "IDLE": continue
        row = task_indices.get(tid)
        if row is None: continue
        c = get_color(tid)
        for (core, s, e) in segs:
            ax = axes[core][0]
            # Fancy edges
            rect = ax.barh(row, e - s, left=s, height=0.6, color=c, edgecolor='#000000', alpha=0.9, linewidth=1)

    # Setup Axes
    for c in range(sim.cores):
        ax = axes[c][0]
        ax.set_yticks(range(len(all_task_ids)))
        ax.set_yticklabels(all_task_ids, color="white")
        ax.set_ylabel(f"CORE {c}", fontweight='bold', color="#00e5ff")
        ax.grid(True, axis='x', linestyle=':', alpha=0.2, color="white")
        if app_task_ids and bg_task_ids:
            boundary = len(app_task_ids) - 0.5
            ax.axhline(boundary, color='#00e5ff', linestyle='--', linewidth=0.5)
            
    # Deadlines & Misses
    deadline_artists = []
    for job in sim.jobs:
        if 0 <= job.absolute_deadline <= sim.sim_time:
             for c in range(sim.cores):
                 dl = axes[c][0].axvline(job.absolute_deadline, color='#ff3333', linestyle=':', linewidth=0.8, alpha=0.6)
                 if c == 0: deadline_artists.append(dl)

        if job.finish_time and job.finish_time > job.absolute_deadline:
             tid = job.task_id
             row = task_indices.get(tid)
             finish_core = 0 
             finish_tick = min(job.finish_time - 1, sim.sim_time - 1)
             for c in range(sim.cores):
                 if sim.timeline[finish_tick][c] == job.uid:
                     finish_core = c
                     break
             if row is not None:
                 axes[finish_core][0].text(job.finish_time, row, "FAIL", color='#ff3333', fontweight='bold', ha='center', va='center', fontsize=8)

    # Energy Plot
    cursors = []
    if sim.energy_aware:
        p_ax = axes[-1][0]
        times = np.arange(sim.sim_time)
        power_vals = [sum(sim.cpu_power_log[t]) for t in range(sim.sim_time)]
        p_ax.plot(times, power_vals, color='#00e5ff', linewidth=1.5)
        p_ax.fill_between(times, power_vals, color='#00e5ff', alpha=0.2)
        p_ax.set_ylabel("Power (W)", color="#00e5ff")
        p_ax.set_ylim(0, (sim.power_active * sim.cores) * 1.2)
        p_ax.grid(True, alpha=0.2)
        cursors.append(p_ax.axvline(0, color='white', linewidth=1))

    for c in range(sim.cores):
        cursors.append(axes[c][0].axvline(0, color='white', linewidth=1))
    
    axes[-1][0].set_xlabel("Time (ticks)", color="white")

    # Animation
    metrics_text = fig.text(0.02, 0.02, "", fontsize=10, color="#00e5ff", fontfamily="monospace")
    
    def update(frame):
        for c in cursors: c.set_xdata([frame])
        
        active_now = []
        for c in range(sim.cores):
            uid = sim.timeline[min(frame, sim.sim_time-1)][c]
            if uid:
                job = sim.job_records.get(uid)
                if job: active_now.append(f"C{c}:{job.task_id}")
        
        status_str = f"T={frame} | RUN: {', '.join(active_now) if active_now else 'IDLE'}"
        
        if sim.energy_aware:
            current_energy = sum(sum(sim.cpu_power_log[t]) for t in range(frame))
            status_str += f" | ENG: {current_energy:.2f} J"
            
        metrics_text.set_text(status_str)
        return cursors + [metrics_text]

    ani = FuncAnimation(fig, update, frames=range(sim.sim_time + 1), interval=100, blit=True)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()

# ------------------------------
# FANCY GUI Config & Main
# ------------------------------

# Base64 icon to avoid external file dependencies (Small Chip Icon)
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
    # --- Theme Constants ---
    COLOR_BG = "#121212"       # Deep Black
    COLOR_PANEL = "#1E1E1E"    # Panel Grey
    COLOR_ACCENT = "#00E5FF"   # Cyber Blue
    COLOR_TEXT = "#E0E0E0"     # Light Grey/White
    COLOR_TEXT_DIM = "#888888" # Dim Text
    COLOR_BUTTON = "#0069D9"   # Button Blue
    COLOR_BUTTON_HOVER = "#005cbf"
    FONT_MAIN = ("Segoe UI", 10)
    FONT_BOLD = ("Segoe UI", 10, "bold")
    FONT_HEADER = ("Segoe UI", 16, "bold")

    root = tk.Tk()
    root.title("RT-SIM CONFIGURATOR")
    root.geometry("500x750")
    root.configure(bg=COLOR_BG)
    root.resizable(False, False)

    # Convert Base64 Icon for display
    try:
        icon_bytes = base64.b64decode(ICON_DATA.strip())
        logo_img = tk.PhotoImage(data=icon_bytes)
        root.iconphoto(False, logo_img) # Set window icon
    except:
        logo_img = None

    # --- Styles ---
    style = ttk.Style()
    style.theme_use('clam') # Clam supports custom colors better than 'vista'
    
    # Configure generic widget styles to match dark theme
    style.configure("Dark.TFrame", background=COLOR_BG)
    style.configure("Panel.TFrame", background=COLOR_PANEL, relief="flat")
    
    style.configure("Dark.TLabel", background=COLOR_PANEL, foreground=COLOR_TEXT, font=FONT_MAIN)
    style.configure("Header.TLabel", background=COLOR_BG, foreground=COLOR_ACCENT, font=FONT_HEADER)
    style.configure("Section.TLabel", background=COLOR_PANEL, foreground=COLOR_ACCENT, font=FONT_BOLD)

    style.configure("Dark.TCheckbutton", background=COLOR_PANEL, foreground=COLOR_TEXT, font=FONT_MAIN)
    style.map("Dark.TCheckbutton", background=[('active', COLOR_PANEL)], indicatorcolor=[('selected', COLOR_ACCENT)])

    style.configure("Action.TButton", background=COLOR_BUTTON, foreground="white", font=FONT_BOLD, borderwidth=0)
    style.map("Action.TButton", background=[('active', COLOR_BUTTON_HOVER)])

    # Combobox styling is tricky in Tkinter, best effort:
    style.configure("Dark.TCombobox", fieldbackground=COLOR_PANEL, background=COLOR_PANEL, foreground=COLOR_TEXT, arrowcolor=COLOR_ACCENT)
    
    # Variables
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

    # --- Header Section ---
    header_frame = ttk.Frame(root, style="Dark.TFrame")
    header_frame.pack(fill="x", pady=20, padx=20)

    if logo_img:
        logo_lbl = tk.Label(header_frame, image=logo_img, bg=COLOR_BG)
        logo_lbl.pack(side="left", padx=(0, 15))
    
    title_lbl = ttk.Label(header_frame, text="REAL-TIME\nSIMULATOR", style="Header.TLabel")
    title_lbl.pack(side="left")

    subtitle_lbl = tk.Label(header_frame, text="v2.0 // Enhanced GUI", bg=COLOR_BG, fg=COLOR_TEXT_DIM, font=("Consolas", 8))
    subtitle_lbl.pack(side="right", anchor="s")

    # --- Main Content Area ---
    main_frame = ttk.Frame(root, style="Dark.TFrame")
    main_frame.pack(fill="both", expand=True, padx=20)

    # 1. Scheduler Settings Panel
    sched_panel = ttk.Frame(main_frame, style="Panel.TFrame", padding=15)
    sched_panel.pack(fill="x", pady=(0, 15))

    ttk.Label(sched_panel, text="KERNEL KERNEL SETTINGS", style="Section.TLabel").pack(anchor="w", pady=(0, 10))
    
    # Grid for options
    sched_grid = ttk.Frame(sched_panel, style="Panel.TFrame")
    sched_grid.pack(fill="x")

    # Scheduler
    ttk.Label(sched_grid, text="Algorithm:", style="Dark.TLabel").grid(row=0, column=0, sticky="w", pady=5)
    sched_cb = ttk.Combobox(sched_grid, textvariable=vars["scheduler"], values=["EDF", "RM", "DM", "LLF", "FCFS", "RR"], state="readonly", style="Dark.TCombobox")
    sched_cb.grid(row=0, column=1, sticky="e", pady=5)

    # Cores
    ttk.Label(sched_grid, text="CPU Cores:", style="Dark.TLabel").grid(row=1, column=0, sticky="w", pady=5)
    core_spin = tk.Spinbox(sched_grid, from_=1, to=8, textvariable=vars["cores"], width=5, bg=COLOR_BG, fg=COLOR_TEXT, buttonbackground=COLOR_PANEL, relief="flat")
    core_spin.grid(row=1, column=1, sticky="e", pady=5)

    # Preemption
    ttk.Checkbutton(sched_panel, text="Enable Preemption", variable=vars["preemption"], style="Dark.TCheckbutton").pack(anchor="w", pady=5)

    # 2. Hardware Environment Panel
    hw_panel = ttk.Frame(main_frame, style="Panel.TFrame", padding=15)
    hw_panel.pack(fill="x", pady=(0, 15))

    ttk.Label(hw_panel, text="HARDWARE PROFILE", style="Section.TLabel").pack(anchor="w", pady=(0, 10))

    # Device
    ttk.Label(hw_panel, text="Device Simulation (Background Noise):", style="Dark.TLabel").pack(anchor="w")
    device_cb = ttk.Combobox(hw_panel, textvariable=vars["device"], 
                                values=["None", "Android (Mobile)", "iOS (Mobile)", "Windows (Laptop)", "macOS (Laptop)"], 
                                state="readonly", width=35, style="Dark.TCombobox")
    device_cb.pack(fill="x", pady=(5, 10))

    # Time & Energy
    hw_grid = ttk.Frame(hw_panel, style="Panel.TFrame")
    hw_grid.pack(fill="x")
    
    ttk.Label(hw_grid, text="Sim Duration (Ticks):", style="Dark.TLabel").grid(row=0, column=0, sticky="w", pady=5)
    tk.Entry(hw_grid, textvariable=vars["sim_time"], width=10, bg=COLOR_BG, fg=COLOR_ACCENT, insertbackground="white", relief="flat").grid(row=0, column=1, sticky="e")

    ttk.Checkbutton(hw_panel, text="Track Energy Consumption (Power Aware)", variable=vars["energy"], style="Dark.TCheckbutton").pack(anchor="w", pady=5)

    # 3. Source Panel
    src_panel = ttk.Frame(main_frame, style="Panel.TFrame", padding=15)
    src_panel.pack(fill="x", pady=(0, 15))

    ttk.Label(src_panel, text="APPLICATION SOURCE", style="Section.TLabel").pack(anchor="w", pady=(0, 5))

    status_frame = tk.Frame(src_panel, bg=COLOR_PANEL)
    status_frame.pack(fill="x", pady=5)

    if HAS_APP_FILE:
        tk.Label(status_frame, text="●", fg="#00FF00", bg=COLOR_PANEL).pack(side="left")
        tk.Label(status_frame, text="app.py loaded", fg=COLOR_TEXT, bg=COLOR_PANEL).pack(side="left", padx=5)
    else:
        tk.Label(status_frame, text="●", fg="#FF0000", bg=COLOR_PANEL).pack(side="left")
        tk.Label(status_frame, text="No app.py found (Using demo data)", fg=COLOR_TEXT, bg=COLOR_PANEL).pack(side="left", padx=5)

    def load_external():
        fp = filedialog.askopenfilename(filetypes=[("JSON/CSV", "*.json *.csv")])
        if fp:
            vars["file_path"].set(fp)
            vars["file_type"].set(fp.split('.')[-1])
            messagebox.showinfo("File Selected", f"Loaded: {os.path.basename(fp)}")

    # Custom styling for standard button using Frame workaround or just standard tk button with colors
    # Using standard Button for reliable BG color
    btn_load = tk.Button(src_panel, text="LOAD EXTERNAL JSON/CSV", command=load_external, 
                         bg=COLOR_PANEL, fg=COLOR_ACCENT, activebackground=COLOR_BG, activeforeground=COLOR_ACCENT,
                         relief="groove", borderwidth=1, font=("Segoe UI", 8))
    btn_load.pack(fill="x", pady=5)

    # --- Footer ---
    def submit():
        root.destroy()
    
    # Big styled start button
    btn_start = tk.Button(root, text="INITIALIZE SIMULATION >", command=submit,
                          bg=COLOR_BUTTON, fg="white", activebackground=COLOR_BUTTON_HOVER, activeforeground="white",
                          font=("Segoe UI", 12, "bold"), relief="flat", pady=10, cursor="hand2")
    btn_start.pack(fill="x", side="bottom")

    # Center window
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry('{}x{}+{}+{}'.format(width, height, x, y))

    root.mainloop()
    return {k: v.get() for k, v in vars.items()}

def load_tasks_from_dict(data_list: List[Dict]) -> List[TaskSpec]:
    specs = []
    for item in data_list:
        specs.append(TaskSpec(
            id=item.get("id"),
            wcet=int(item.get("wcet")),
            period=item.get("period"),
            deadline=item.get("deadline"),
            release=int(item.get("release", 0)),
            type=item.get("type", "aperiodic"),
            color=item.get("color"),
            category="Application", # Default for loaded tasks
            dependencies=item.get("dependencies", [])
        ))
    return specs

def get_scheduler(name):
    schedulers = {"EDF": EDF, "RM": RM, "DM": DM, "LLF": LLF, "FCFS": FCFS, "RR": RR}
    return schedulers.get(name, EDF)()

def main():
    config = gui_config_menu()
    if not config["scheduler"]: return

    # 1. Load Application Tasks
    app_specs = []
    if config["file_path"]:
        # Load CSV/JSON if provided override
        if config["file_type"] == "json":
             with open(config["file_path"]) as f: app_specs = load_tasks_from_dict(json.load(f))
        # Add CSV logic here if needed
    elif HAS_APP_FILE:
        print("Loading application tasks from app.py...")
        app_data = user_app.get_user_application_tasks()
        app_specs = load_tasks_from_dict(app_data)
    else:
        print("No app.py found. Using dummy data.")
        app_specs = [TaskSpec("DummyApp", 2, 10, category="Application", type="periodic")]

    # 2. Generate Background Tasks
    print(f"Generating background noise for: {config['device']}")
    bg_specs = get_background_tasks(config["device"], config["sim_time"])

    # 3. Merge
    all_specs = app_specs + bg_specs
    print(f"Total Tasks: {len(all_specs)} ({len(app_specs)} App, {len(bg_specs)} Background)")

    # 4. Run
    sim = Simulator(
        task_specs=all_specs,
        scheduler=get_scheduler(config["scheduler"]),
        sim_time=config["sim_time"],
        preemptive=config["preemption"],
        cores=config["cores"],
        energy_aware=config["energy"]
    )
    
    metrics = sim.run()
    
    print("\n--- Simulation Metrics ---")
    for k, v in metrics.items(): print(f"{k}: {v}")
    
    plot_gantt_and_animate(sim, f"Analysis: {config['device']} | {config['scheduler']}")

if __name__ == "__main__":
    main()