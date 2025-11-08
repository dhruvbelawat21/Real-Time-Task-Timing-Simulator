"""
Real-Time Task Timing Visualizer (Simulator)
Single-file Python implementation.

Dependencies:
- Python 3.8+
- numpy
- matplotlib
- pandas (optional, for CSV load)

MODIFIED:
- Added Energy-Aware simulation.
- Tracks active/idle power per core.
- Calculates total energy consumption.
- Visualizes total power consumption in a subplot below the Gantt chart.
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
import statistics
import os
import datetime

# ------------------------------
# Task & Job Models
# ------------------------------

@dataclass
class TaskSpec:
    """Specification for a task type (periodic or aperiodic)."""
    id: str
    wcet: int                      # worst-case execution time (time units)
    period: Optional[int] = None   # if periodic, the period (and implicit deadline)
    deadline: Optional[int] = None # relative deadline (if None and periodic -> equals period)
    release: int = 0               # initial release time
    instances: Optional[int] = None # number of instances for periodic tasks (None => run until sim end)
    color: Optional[str] = None
    type: str = "aperiodic"        # "periodic" or "aperiodic" or "interrupt"
    priority: Optional[int] = None # fixed priority (if used by fixed-priority schedulers)
    jitter: int = 0                # max jitter added to release for modeling jitter
    dependencies: List[str] = field(default_factory=list)

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
    """Abstract scheduler. Implement get_next_job(ready_jobs, current_time)."""
    preemptive: bool = True

    def __init__(self, name="BASE"):
        self.name = name

    def add_job(self, job: Job):
        pass

    def remove_job(self, job: Job):
        pass

    def get_next_job(self, ready_jobs: List[Job], current_time: int, running_job: Optional[Job]) -> Optional[Job]:
        raise NotImplementedError

class EDF(SchedulerBase):
    def __init__(self):
        super().__init__("EDF")
        self.preemptive = True

    def get_next_job(self, ready_jobs, current_time, running_job):
        if not ready_jobs:
            return None
        # earliest absolute deadline
        return min(ready_jobs, key=lambda j: j.absolute_deadline)

class RM(SchedulerBase):
    def __init__(self):
        super().__init__("RM")
        self.preemptive = True

    def get_next_job(self, ready_jobs, current_time, running_job):
        if not ready_jobs:
            return None
        # priority = smaller period (for tasks with period defined)
        # fallback: use absolute_deadline if no period
        return min(ready_jobs, key=lambda j: (math.inf if j.spec.period is None else j.spec.period, j.absolute_deadline))

class DM(SchedulerBase):
    def __init__(self):
        super().__init__("DM")
        self.preemptive = True

    def get_next_job(self, ready_jobs, current_time, running_job):
        if not ready_jobs:
            return None
        # priority = shorter relative deadline (deadline - release)
        return min(ready_jobs, key=lambda j: ((j.absolute_deadline - j.release_time) if (j.absolute_deadline - j.release_time) > 0 else math.inf, j.absolute_deadline))

class LLF(SchedulerBase):
    def __init__(self):
        super().__init__("LLF")
        self.preemptive = True

    def get_next_job(self, ready_jobs, current_time, running_job):
        if not ready_jobs:
            return None
        # laxity = (deadline - current_time - remaining)
        def laxity(j):
            return (j.absolute_deadline - current_time - j.remaining)
        # job with smallest laxity
        return min(ready_jobs, key=lambda j: (laxity(j), j.absolute_deadline))

class FCFS(SchedulerBase):
    def __init__(self):
        super().__init__("FCFS")
        self.preemptive = False

    def get_next_job(self, ready_jobs, current_time, running_job):
        if not ready_jobs:
            return None
        # first released first served -> min release_time then uid
        return min(ready_jobs, key=lambda j: (j.release_time, j.uid))

class SRTF(SchedulerBase):
    def __init__(self):
        super().__init__("SRTF")
        self.preemptive = True

    def get_next_job(self, ready_jobs, current_time, running_job):
        if not ready_jobs:
            return None
        return min(ready_jobs, key=lambda j: (j.remaining, j.absolute_deadline))

class RR(SchedulerBase):
    def __init__(self, quantum=2):
        super().__init__("RR")
        self.preemptive = True
        self.quantum = quantum
        self._queue: List[Job] = []
        self._time_slice_left = 0
        self._current: Optional[Job] = None

    def get_next_job(self, ready_jobs, current_time, running_job):
        # maintain internal queue in arrival order
        # when new job arrives add to queue
        # if current job exists and time slice left >0 and job still has remaining, continue it
        # otherwise rotate
        # Build a queue: include all ready jobs in arrival order (release_time, uid), but keep internal ordering across calls
        # Sync internal queue with ready_jobs
        ready_set = {j.uid for j in ready_jobs}
        # remove jobs not in ready
        self._queue = [j for j in self._queue if j.uid in ready_set]
        # add any new ones at end in arrival order
        existing_uids = {j.uid for j in self._queue}
        for j in sorted(ready_jobs, key=lambda x: (x.release_time, x.uid)):
            if j.uid not in existing_uids:
                self._queue.append(j)
                existing_uids.add(j.uid)

        # If running_job still has remaining and is same as current internal, continue if slice left.
        if self._current and self._current.remaining > 0 and self._current.uid in existing_uids and self._time_slice_left > 0:
            return self._current

        # else pick next from queue head
        if not self._queue:
            self._current = None
            self._time_slice_left = 0
            return None
        # rotate if current was at head
        if self._current and self._current in self._queue:
            # move current to end if it's finished or slice exhausted
            try:
                idx = self._queue.index(self._current)
            except ValueError:
                idx = None
            if idx == 0:
                # pop head if finished or exhausted
                self._queue.pop(0)
                # if still has remaining, append to end
                if self._current.remaining > 0:
                    self._queue.append(self._current)
        # pick head
        if not self._queue: # Queue might be empty after pop
             self._current = None
             self._time_slice_left = 0
             return None
        self._current = self._queue[0]
        self._time_slice_left = self.quantum
        return self._current

    def consume_quantum(self, amount=1):
        self._time_slice_left = max(0, self._time_slice_left - amount)


# ------------------------------
# Simulator core
# ------------------------------

class Simulator:
    def __init__(
        self,
        task_specs: List[TaskSpec],
        scheduler: SchedulerBase,
        sim_time: int = 200,
        time_unit: int = 1,
        preemptive: bool = True,
        cores: int = 1,
        # --- NEW: Energy parameters ---
        energy_aware: bool = False,
        power_active: float = 2.5,
        power_idle: float = 0.5
    ):
        self.specs = task_specs
        self.scheduler = scheduler
        self.sim_time = sim_time
        self.time_unit = time_unit # This is our 'dt' for energy calcs
        self.preemptive = preemptive
        self.cores = cores

        # --- NEW: Energy attributes ---
        self.energy_aware = energy_aware
        self.power_active = power_active
        self.power_idle = power_idle
        self.total_energy = 0.0
        self.cpu_power_log: List[List[float]] = [] # Log power (W) per core per tick
        if self.energy_aware:
            # Pre-allocate log, assuming idle state initially
            self.cpu_power_log = [[self.power_idle] * cores for _ in range(sim_time)]

        # State variables
        self.jobs: List[Job] = []
        self.pending_jobs: List[Job] = []
        self.jobs_to_release: List[Job] = []
        self.ready: List[Job] = []
        self.running: List[Optional[Job]] = [None] * cores  # track job per core
        self.timeline: List[List[Optional[int]]] = [
            [None] * cores for _ in range(sim_time)
        ]  # uid per core per tick

        self.gantt_segments: Dict[str, List[Tuple[int, int, int]]] = {}  # task_id -> list of (core, start, end)
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
            abs_deadline = rel + spec.wcet * 10  # default soft deadline

        job = Job(
            uid=self.uid_counter,
            task_id=spec.id,
            release_time=rel,
            absolute_deadline=abs_deadline,
            exec_time=spec.wcet,
            remaining=spec.wcet,
            spec=spec,
            instance_no=instance_no,
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
        """Instantiate all jobs and link their dependencies based on instance number."""
        releases = self.prepare_job_releases()
        instance_map: Dict[Tuple[str, int], Job] = {} # Key: (task_id, instance_no)

        # First pass: Instantiate all jobs
        for (rel_time, spec, inst_no) in releases:
            job = self.instantiate_job(spec, rel_time, inst_no)
            instance_map[(spec.id, inst_no)] = job

        # Second pass: Link dependencies
        # This assumes T2_inst_N depends on T1_inst_N
        for job in self.jobs:
            if not job.spec.dependencies:
                continue
            for dep_task_id in job.spec.dependencies:
                # Find the dependency with the *same instance number*
                dep_job = instance_map.get((dep_task_id, job.instance_no))
                if dep_job:
                    job.dependencies.append(dep_job.uid)
                else:
                    # Handle case where dependency instance might not exist (e.g., runs for fewer instances)
                    print(f"Warning: Could not find dependency {dep_task_id} (instance {job.instance_no}) for job {job.uid} ({job.task_id})")
        
        # Sort all jobs by release time for the main loop
        self.jobs_to_release = sorted(self.jobs, key=lambda j: j.release_time)


    def _dependencies_met(self, job: Job) -> bool:
        """Check if all dependencies for a job are finished."""
        if not job.dependencies:
            return True
        for dep_uid in job.dependencies:
            dep_job = self.job_records.get(dep_uid)
            # Fails if dependency job doesn't exist or is not finished
            if not dep_job or dep_job.finish_time is None:
                return False
        return True

    def run(self, realtime_visualize: bool = False, verbose: bool = False):
        
        # Reset state
        self.jobs = []
        self.pending_jobs = []
        self.jobs_to_release = []
        self.ready = []
        self.running = [None] * self.cores
        self.timeline = [[None] * self.cores for _ in range(self.sim_time)]
        self.gantt_segments = {}
        self.job_records = {}
        self.uid_counter = 1
        
        # --- NEW: Reset energy state ---
        self.total_energy = 0.0
        if self.energy_aware:
            self.cpu_power_log = [[self.power_idle] * self.cores for _ in range(self.sim_time)]
        
        
        self._pre_instantiate_and_link()
        release_idx = 0
        
        for t in range(self.sim_time):
            # 1. Release jobs -> pending
            while release_idx < len(self.jobs_to_release) and self.jobs_to_release[release_idx].release_time <= t:
                job = self.jobs_to_release[release_idx]
                self.pending_jobs.append(job)
                release_idx += 1

            # 2. Check dependencies -> pending to ready
            newly_ready = []
            for job in self.pending_jobs:
                if self._dependencies_met(job):
                    self.ready.append(job)
                    newly_ready.append(job)
            if newly_ready:
                self.pending_jobs = [j for j in self.pending_jobs if j not in newly_ready]

            # 3. Assign jobs to cores
            for core_id in range(self.cores):
                current = self.running[core_id]
                
                # --- Fix: If job finished last tick, 'current' should be None ---
                if current and current.remaining <= 0:
                    current.finish_time = t
                    current.response_time = current.finish_time - current.release_time
                    current = None
                    self.running[core_id] = None
                
                # Get jobs that are ready OR the 'current' job (if it exists)
                available_jobs = self.ready
                if current and current not in available_jobs:
                    available_jobs = self.ready + [current]
                
                # --- Fix: Exclude jobs running on *other* cores ---
                other_running = [self.running[c] for c in range(self.cores) if c != core_id and self.running[c] is not None]
                available_jobs = [j for j in available_jobs if j not in other_running]

                candidate = self.scheduler.get_next_job(available_jobs, t, current)
                
                if candidate is None:
                    # No candidate. If 'current' was running, it's preempted.
                    if current:
                        if current.remaining > 0 and current not in self.ready:
                            self.ready.append(current) # Put back in ready
                    self.running[core_id] = None
                    self.timeline[t][core_id] = None
                    # --- NEW: Energy logging for IDLE ---
                    if self.energy_aware:
                        self.cpu_power_log[t][core_id] = self.power_idle
                    continue # Go to next core

                if current is None:
                    # Core was idle
                    self.running[core_id] = candidate
                    if candidate.start_time is None:
                        candidate.start_time = t
                    if candidate in self.ready:
                        self.ready.remove(candidate)
                        
                elif candidate.uid != current.uid:
                    # Switch
                    if self.preemptive and self.scheduler.preemptive:
                        if current.remaining > 0 and current not in self.ready:
                            self.ready.append(current) # Put current back
                            
                        self.running[core_id] = candidate
                        if candidate.start_time is None:
                            candidate.start_time = t
                        if candidate in self.ready:
                            self.ready.remove(candidate)
                    else:
                        # Non-preemptive: 'current' continues
                        candidate = current
                        self.running[core_id] = current # Ensure it stays
                else:
                    # Candidate is same as current, do nothing
                    pass

                # Execute 1 tick
                job_to_run = self.running[core_id]
                if job_to_run:
                    job_to_run.remaining -= 1
                    self.timeline[t][core_id] = job_to_run.uid
                    
                    # --- NEW: Energy logging for ACTIVE ---
                    if self.energy_aware:
                        self.cpu_power_log[t][core_id] = self.power_active

                    # Round Robin: quantum control
                    if isinstance(self.scheduler, RR):
                        self.scheduler.consume_quantum(1)

                    # If job finishes *now*
                    if job_to_run.remaining <= 0:
                        job_to_run.finish_time = t + 1
                        job_to_run.response_time = job_to_run.finish_time - job_to_run.release_time
                        self.running[core_id] = None # Core will be idle next tick
                
                else: # Should have been caught by 'candidate is None' block, but as safeguard
                     # --- NEW: Energy logging for IDLE ---
                    if self.energy_aware:
                        self.cpu_power_log[t][core_id] = self.power_idle


            # Cleanup ready list
            self.ready = [j for j in self.ready if j.remaining > 0]
            
        # --- End of main 'run' loop ---

        # Build Gantt segments
        uid_to_task = {j.uid: j.task_id for j in self.jobs}
        for job in self.jobs:
            if job.task_id not in self.gantt_segments:
                self.gantt_segments[job.task_id] = []

        for core_id in range(self.cores):
            current_uid = None
            seg_start = 0
            for t in range(self.sim_time):
                uid = self.timeline[t][core_id]
                if uid != current_uid:
                    if current_uid is not None:
                        task_id = uid_to_task.get(current_uid, "IDLE")
                        self.gantt_segments.setdefault(task_id, []).append(
                            (core_id, seg_start, t)
                        )
                    current_uid = uid
                    seg_start = t
            if current_uid is not None:
                task_id = uid_to_task.get(current_uid, "IDLE")
                self.gantt_segments.setdefault(task_id, []).append(
                    (core_id, seg_start, self.sim_time)
                )

        return self._compute_metrics()

    def _compute_metrics(self):
        busy = sum(
            1
            for t in range(self.sim_time)
            for c in range(self.cores)
            if self.timeline[t][c] is not None
        )
        total_core_time = (self.sim_time * self.cores)
        cpu_util = busy / total_core_time if total_core_time > 0 else 0.0
        
        finished = [j for j in self.jobs if j.finish_time is not None]
        response_times = [j.response_time for j in finished if j.response_time]
        avg_response = statistics.mean(response_times) if response_times else None

        missed = sum(
            1
            for j in self.jobs
            if (j.finish_time and j.finish_time > j.absolute_deadline)
            or (j.finish_time is None and j.absolute_deadline < self.sim_time)
        )

        metrics = {
            "cpu_utilization": cpu_util,
            "avg_response_time": avg_response,
            "missed_deadlines": missed,
            "total_jobs": len(self.jobs),
            "finished_jobs": len(finished),
            "busy_time": busy,
        }
        
        # --- NEW: Calculate and add energy metrics ---
        if self.energy_aware:
            self.total_energy = 0.0
            for t in range(self.sim_time):
                for c in range(self.cores):
                    self.total_energy += self.cpu_power_log[t][c] * self.time_unit
            
            total_sim_duration = self.sim_time * self.time_unit
            avg_power = (self.total_energy / total_sim_duration) if total_sim_duration > 0 else 0.0
            
            metrics["total_energy_j"] = self.total_energy
            metrics["avg_power_w"] = avg_power

        return metrics

    def export_csv(self, filename="simulation_results.csv"):
        metrics = self._compute_metrics()
        now = datetime.datetime.now().isoformat()
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["sim_time", self.sim_time])
            writer.writerow(["scheduler", self.scheduler.name])
            writer.writerow(["cores", self.cores])
            writer.writerow(["preemptive", self.preemptive])
            writer.writerow(["generated_on", now])
            writer.writerow([])
            writer.writerow(["metrics"])
            for k, v in metrics.items():
                writer.writerow([k, v])
            writer.writerow([])
            writer.writerow(["jobs"])
            writer.writerow(
                [
                    "uid",
                    "task_id",
                    "instance_no",
                    "release",
                    "deadline",
                    "wcet",
                    "start",
                    "finish",
                    "response_time",
                    "missed",
                ]
            )
            for j in sorted(self.jobs, key=lambda x: x.uid):
                missed = int(
                    (j.finish_time and j.finish_time > j.absolute_deadline)
                    or (j.finish_time is None and j.absolute_deadline < self.sim_time)
                )
                writer.writerow(
                    [
                        j.uid,
                        j.task_id,
                        j.instance_no,
                        j.release_time,
                        j.absolute_deadline,
                        j.exec_time,
                        j.start_time,
                        j.finish_time,
                        j.response_time,
                        missed,
                    ]
                )
        return filename

# ------------------------------
# Visualization (Matplotlib Gantt)
# ------------------------------

def plot_gantt_and_animate(sim: Simulator, title: str = "RT Scheduling Gantt", playback_speed: float = 1.0):
    # Build list of tasks order (consistent display)
    task_ids = sorted(list(sim.gantt_segments.keys()))
    if "IDLE" in task_ids:
        task_ids.remove("IDLE")
        task_ids = ["IDLE"] + task_ids

    # map tasks to row index
    task_indices = {tid: idx for idx, tid in enumerate(task_ids)}

    num_tasks = len(task_ids)
    
    # --- MODIFIED: Add space for energy plot ---
    num_gantt_subplots = sim.cores
    num_total_subplots = sim.cores
    if sim.energy_aware:
        num_total_subplots += 1 # Add one subplot for the power graph

    # Calculate dynamic height
    fig_height = 1 + num_gantt_subplots * (num_tasks * 0.3)
    if sim.energy_aware:
        fig_height += 2 # Add 2 inches for the power plot

    if num_total_subplots > 0:
        fig, axes = plt.subplots(num_total_subplots, 1, figsize=(12, fig_height), sharex=True, squeeze=False)
        fig.suptitle(title + f"  (Scheduler: {sim.scheduler.name})")
    else:
        print("Error: No subplots to create.")
        return


    colors = plt.get_cmap('tab20').colors
    color_map = {}
    yticks = []
    ylabels = []
    for i, tid in enumerate(task_ids):
        yticks.append(i + 0.5)
        ylabels.append(tid)
        color_map[tid] = colors[i % len(colors)]

    # Prepare rectangles
    bars = []
    for tid, segs in sim.gantt_segments.items():
        row_idx = task_indices.get(tid, None)
        if row_idx is None:
            continue
        
        for (core_id, s, e) in segs:
            ax = axes[core_id][0] # Select the subplot for this core
            rect = ax.barh(row_idx + 0.1, e - s, left=s, height=0.8, align='center', color=color_map.get(tid, 'grey'), edgecolor='black')
            bars.append(rect)

    # Configure Gantt axes
    deadline_artists = []
    missed_artists = []
    
    for core_id in range(num_gantt_subplots):
        ax = axes[core_id][0]
        if sim.cores > 1:
            ax.set_ylabel(f"Core {core_id}", rotation=0, labelpad=25, ha='right')

        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels)
        ax.set_xlim(0, sim.sim_time)
        ax.set_ylim(0, len(task_ids))
        ax.grid(True, axis='x', linestyle='--', alpha=0.4)

        # draw deadlines: for each job, vertical line
        for job in sim.jobs:
            if 0 <= job.absolute_deadline <= sim.sim_time:
                dl = ax.axvline(job.absolute_deadline, color='red', linestyle='--', linewidth=0.6, alpha=0.6)
                if core_id == 0: deadline_artists.append(dl) # Add only once

        # annotate missed deadlines
        for job in sim.jobs:
            if job.finish_time and job.finish_time > job.absolute_deadline:
                tid = job.task_id
                row = task_indices.get(tid, None)
                if row is not None:
                    # Find which core it finished on
                    finish_core = -1
                    for c_id in range(sim.cores):
                        # Handle edge case where finish_time is sim_time
                        finish_tick = min(job.finish_time - 1, sim.sim_time - 1)
                        if sim.timeline[finish_tick][c_id] == job.uid:
                            finish_core = c_id
                            break
                    if finish_core == core_id:
                        mt = ax.text(job.finish_time + 0.1, row + 0.5, "✖", color='red', fontsize=10)
                        if core_id == 0: missed_artists.append(mt)
    
    # --- NEW: Configure and draw energy plot ---
    cursors = [] # List for all animated cursors
    
    if sim.energy_aware:
        power_ax = axes[sim.cores][0] # The last subplot
        times = np.arange(sim.sim_time)
        # Calculate total power at each time step
        total_power_log = [sum(sim.cpu_power_log[t]) for t in range(sim.sim_time)]
        
        power_ax.plot(times, total_power_log, color='green', label='Total Power', linestyle='-')
        power_ax.set_ylabel("Total Power (W)")
        power_ax.set_xlim(0, sim.sim_time)
        # Set Y-limit from 0 to max possible power + 10%
        max_power = (sim.power_active * sim.cores) * 1.1
        min_power = (sim.power_idle * sim.cores) * 0.9
        power_ax.set_ylim(min_power, max_power)
        power_ax.grid(True, linestyle='--', alpha=0.4)
        
        # Add cursor for this subplot
        power_cursor = power_ax.axvline(0, color='blue', linewidth=1.2)
        cursors.append(power_cursor)

    # Add cursors for Gantt charts
    for core_id in range(num_gantt_subplots):
        gantt_cursor = axes[core_id][0].axvline(0, color='blue', linewidth=1.2)
        cursors.append(gantt_cursor)

    # Set X-label on the very last subplot
    axes[-1][0].set_xlabel("Time (ticks)")


    # subplot for metrics (relative to the figure)
    metrics_text = fig.text(0.01, 0.01, "", fontsize=9, va="bottom", ha="left")

    # animation update function
    def update(frame):
        artists = []
        for c in cursors:
            c.set_xdata([frame])
            artists.append(c)
            
        metrics = sim._compute_metrics()
        mt = (
            f"Time: {frame}/{sim.sim_time}   "
            f"CPU util (total): {metrics['cpu_utilization']:.3f}   "
            f"Missed: {metrics['missed_deadlines']}   "
            f"Finished: {metrics['finished_jobs']}/{metrics['total_jobs']}"
        )
        if sim.energy_aware:
            mt += f"   Energy: {metrics.get('total_energy_j', 0.0):.2f} J"

        metrics_text.set_text(mt)
        artists.append(metrics_text)
        
        # We need to return all artists
        return artists + deadline_artists + missed_artists + [b[0] for b in bars if b]


    ani = FuncAnimation(fig, update, frames=range(0, sim.sim_time + 1), interval=max(1, int(500 / playback_speed)), blit=False, repeat=False)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # Adjust for suptitle and metrics
    plt.show()


# ------------------------------
# Helpers for loading tasks
# ------------------------------

def load_tasks_from_json(path: str) -> List[TaskSpec]:
    with open(path, "r") as f:
        data = json.load(f)
    specs = []
    for item in data:
        specs.append(TaskSpec(
            id=item.get("id"),
            wcet=int(item.get("wcet")),
            period=(None if item.get("period") is None else int(item.get("period"))),
            deadline=(None if item.get("deadline") is None else int(item.get("deadline"))),
            release=int(item.get("release", 0)),
            instances=(None if item.get("instances") is None else int(item.get("instances"))),
            color=item.get("color"),
            type=item.get("type", "aperiodic"),
            priority=item.get("priority"),
            jitter=int(item.get("jitter", 0)),
            dependencies=item.get("dependencies", [])
        ))
    return specs

def load_tasks_from_csv(path: str) -> List[TaskSpec]:
    specs = []
    with open(path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Handle comma-separated strings for dependencies
            dep_str = row.get("dependencies", "")
            dependencies = [] if not dep_str else [task.strip() for task in dep_str.split(',')]
            
            specs.append(TaskSpec(
                id=row.get("id"),
                wcet=int(row.get("wcet")),
                period=(None if not row.get("period") else int(row.get("period"))),
                deadline=(None if not row.get("deadline") else int(row.get("deadline"))),
                release=int(row.get("release", 0)),
                instances=(None if not row.get("instances") else int(row.get("instances"))),
                color=row.get("color"),
                type=row.get("type", "aperiodic"),
                priority=(None if not row.get("priority") else int(row.get("priority"))),
                jitter=(0 if not row.get("jitter") else int(row.get("jitter"))),
                dependencies=dependencies
            ))
    return specs

# ------------------------------
# Sample tasks for quick test
# ------------------------------

SAMPLE_TASKS = [
    # periodic tasks: id, wcet, period, (deadline default=period)
    TaskSpec(id="T1", wcet=1, period=4, release=0, type="periodic", instances=20),
    TaskSpec(id="T2", wcet=2, period=6, release=0, type="periodic", instances=15),
    TaskSpec(id="T3", wcet=1, period=8, release=0, type="periodic", instances=10, dependencies=["T1"]),

    # aperiodic
    TaskSpec(id="A1", wcet=3, release=5, deadline=12, type="aperiodic"),
    TaskSpec(id="A2", wcet=2, release=12, deadline=20, type="aperiodic", dependencies=["A1"]),

    # interrupt-style (one-shot arriving later)
    TaskSpec(id="IRQ", wcet=1, release=18, deadline=20, type="interrupt"),
]

# ------------------------------
# CLI / Runner
# ------------------------------

def get_scheduler_by_name(name: str, quantum: int = 2):
    name = name.upper()
    if name == "EDF":
        return EDF()
    if name == "RM":
        return RM()
    if name == "DM":
        return DM()
    if name == "LLF":
        return LLF()
    if name == "FCFS":
        return FCFS()
    if name == "SRTF":
        return SRTF()
    if name == "RR":
        return RR(quantum=quantum)
    raise ValueError("Unknown scheduler")

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import sys

def gui_config_menu():
    root = tk.Tk()
    root.title("Scheduler Configuration")
    # --- MODIFIED: Increased height for new options ---
    root.geometry("420x520")

    # Variables
    scheduler_var = tk.StringVar(value="EDF")
    preemption_var = tk.BooleanVar(value=True)
    cores_var = tk.IntVar(value=1)
    sim_time_var = tk.IntVar(value=50)
    quantum_var = tk.IntVar(value=2)
    file_path_var = tk.StringVar(value="")
    file_type_var = tk.StringVar(value="None")
    
    # --- NEW: Energy variables ---
    energy_aware_var = tk.BooleanVar(value=True) # Default to True for demo
    power_active_var = tk.DoubleVar(value=2.5)
    power_idle_var = tk.DoubleVar(value=0.5)

    # === Labels & Inputs ===
    ttk.Label(root, text="Scheduler Type:", font=("Arial", 11)).pack(pady=5)
    sched_combo = ttk.Combobox(root, textvariable=scheduler_var, 
                               values=["EDF", "RM", "DM", "LLF", "FCFS", "SRTF", "RR"], 
                               state="readonly")
    sched_combo.pack()

    ttk.Label(root, text="Preemption:", font=("Arial", 11)).pack(pady=5)
    ttk.Checkbutton(root, text="Enable Preemption (Global)", variable=preemption_var).pack()

    ttk.Label(root, text="Number of Cores:", font=("Arial", 11)).pack(pady=5)
    ttk.Spinbox(root, from_=1, to=16, textvariable=cores_var, width=5).pack()

    ttk.Label(root, text="Simulation Time (ticks):", font=("Arial", 11)).pack(pady=5)
    ttk.Entry(root, textvariable=sim_time_var, width=10).pack()

    ttk.Label(root, text="Quantum (for RR):", font=("Arial", 11)).pack(pady=5)
    ttk.Entry(root, textvariable=quantum_var, width=10).pack()

    # --- NEW: Energy options frame ---
    energy_frame = ttk.LabelFrame(root, text="Energy Simulation", padding=(10, 5))
    energy_frame.pack(pady=10, padx=10, fill="x")
    
    ttk.Checkbutton(energy_frame, text="Enable Energy Simulation", variable=energy_aware_var).pack()
    
    power_frame = ttk.Frame(energy_frame)
    ttk.Label(power_frame, text="Active Power (W):").pack(side=tk.LEFT, padx=5)
    ttk.Entry(power_frame, textvariable=power_active_var, width=5).pack(side=tk.LEFT)
    ttk.Label(power_frame, text="Idle Power (W):").pack(side=tk.LEFT, padx=5)
    ttk.Entry(power_frame, textvariable=power_idle_var, width=5).pack(side=tk.LEFT)
    power_frame.pack(pady=5)
    # --- End of new frame ---

    ttk.Label(root, text="Load Task File:", font=("Arial", 11)).pack(pady=5)
    file_frame = ttk.Frame(root)
    file_frame.pack()

    def browse_file():
        filepath = filedialog.askopenfilename(filetypes=[("JSON files", "*.json"), ("CSV files", "*.csv")])
        if filepath:
            file_path_var.set(filepath)
            if filepath.endswith(".json"):
                file_type_var.set("json")
            elif filepath.endswith(".csv"):
                file_type_var.set("csv")

    ttk.Button(file_frame, text="Browse...", command=browse_file).pack(side=tk.LEFT, padx=5)
    ttk.Label(file_frame, textvariable=file_path_var, wraplength=300).pack(side=tk.LEFT)

    def submit():
        if sim_time_var.get() <= 0:
            messagebox.showerror("Error", "Simulation time must be positive")
            return
        root.destroy()

    ttk.Button(root, text="Run Simulation", command=submit).pack(pady=20)

    root.mainloop()

    return {
        "scheduler": scheduler_var.get(),
        "preemption": preemption_var.get(),
        "cores": cores_var.get(),
        "sim_time": sim_time_var.get(),
        "quantum": quantum_var.get(),
        "file_path": file_path_var.get(),
        "file_type": file_type_var.get(),
        # --- NEW: Return energy config ---
        "energy_aware": energy_aware_var.get(),
        "power_active": power_active_var.get(),
        "power_idle": power_idle_var.get(),
    }


def main():
    # === Get config from GUI ===
    config = gui_config_menu()
    
    if not config["scheduler"]:
        print("Simulation cancelled.")
        return

    # === Load tasks ===
    if config["file_type"] == "json":
        print(f"Loading tasks from {config['file_path']}...")
        specs = load_tasks_from_json(config["file_path"])
    elif config["file_type"] == "csv":
        print(f"Loading tasks from {config['file_path']}...")
        specs = load_tasks_from_csv(config["file_path"])
    else:
        print("No file loaded, using sample tasks.")
        specs = SAMPLE_TASKS

    scheduler = get_scheduler_by_name(config["scheduler"], quantum=config["quantum"])
    
    sim = Simulator(
        task_specs=specs,
        scheduler=scheduler,
        sim_time=config["sim_time"],
        preemptive=config["preemption"],
        cores=config["cores"],
        # --- NEW: Pass energy config to Simulator ---
        energy_aware=config["energy_aware"],
        power_active=config["power_active"],
        power_idle=config["power_idle"]
    )

    print(f"Running simulation: {config['cores']} core(s), {scheduler.name} scheduler, Preemption={config['preemption']}, Energy Sim={config['energy_aware']}")
    metrics = sim.run()
    
    print("\n--- Simulation Complete: Metrics ---")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")
    print("--------------------------------------\n")

    fname = sim.export_csv("simulation_output.csv")
    print(f"Results exported to {fname}")

    print("Launching visualization...")
    plot_gantt_and_animate(sim, title=f"{scheduler.name} Visualization ({config['cores']} Core(s))", playback_speed=1.0)


if __name__ == "__main__":
    main()