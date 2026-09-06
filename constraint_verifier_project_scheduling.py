"""
Constraint verifier for the project-scheduling problem described in L'X.docx.

Each task contains:
    - start_time       : t_i^S, in hours from project time t0 = 0
    - finish_time      : t_i^F, in hours from project time t0 = 0
    - required_skill   : the single skill required by the task
    - working_days     : W_i, the days on which the task is actually worked
    - assignments      : {day: [employee_id, ...]}

Each employee contains:
    - skills: set of skills possessed by the employee

The verifier checks feasibility. It does NOT optimize the schedule.

Note on day indexing:
The explicit day constraints in the document imply
    d_i - 1 <= t_i / 12 < d_i
so the implementation uses the 1-based mapping:
    d_i = floor(t_i / 12) + 1
"""

from dataclasses import dataclass, field
from math import floor, ceil
from typing import Dict, List, Set, Tuple


WORKDAY_HOURS = 12.0


@dataclass
class Employee:
    employee_id: int
    skills: Set[str]


@dataclass
class Task:
    task_id: int
    start_time: float
    finish_time: float
    required_skill: str

    # W_i: actual days on which the task is worked.
    # Gaps are allowed, so interruptions are supported.
    working_days: Set[int] = field(default_factory=set)

    # assignments[day] = employees working on this task on that day.
    assignments: Dict[int, List[int]] = field(default_factory=dict)


@dataclass
class ProjectSchedule:
    tasks: List[Task]
    employees: List[Employee]

    # L: tasks starting at the original project timeline t0 = 0.
    start_tasks: Set[int] = field(default_factory=set)


def day_from_time(time_hours: float) -> int:
    """Return the 1-based project day associated with a time in hours."""
    return floor(time_hours / WORKDAY_HOURS) + 1


def time_day_consistent(time_hours: float, day: int) -> bool:
    """Check d-1 <= t/12 < d."""
    return (
        day - 1 <= time_hours / WORKDAY_HOURS
        and time_hours / WORKDAY_HOURS < day
    )


def verify_schedule(schedule: ProjectSchedule) -> Tuple[bool, List[str]]:
    """
    Check the scheduling constraints from the formalization.

    Returns:
        feasible  : True if no implemented constraint is violated.
        violations: list of violated constraints.
    """

    violations: List[str] = []
    tasks = schedule.tasks
    employees = {e.employee_id: e for e in schedule.employees}
    task_ids = {t.task_id for t in tasks}

    if not tasks:
        return False, ["The task set is empty."]

    if not employees:
        violations.append("The employee set is empty.")

    # Project completion time t_e and project length lambda.
    t_e = max((t.finish_time for t in tasks), default=0.0)
    lam = ceil(t_e / WORKDAY_HOURS)

    # ------------------------------------------------------------
    # L: tasks starting at t0 = 0
    # ------------------------------------------------------------
    L = set(schedule.start_tasks)

    unknown_L = L - task_ids
    if unknown_L:
        violations.append(
            f"L contains unknown task IDs: {sorted(unknown_L)}."
        )

    # If L was not explicitly supplied, infer it from tS = 0.
    if not L:
        L = {t.task_id for t in tasks if t.start_time == 0}

    if not L:
        violations.append(
            "L is empty: at least one task must start at t0 = 0."
        )

    # ------------------------------------------------------------
    # Per-task constraints
    # ------------------------------------------------------------
    for task in tasks:
        i = task.task_id
        ts = task.start_time
        tf = task.finish_time

        # t_i^S <= t_e
        if ts > t_e:
            violations.append(
                f"Task {i}: tS={ts} > project end t_e={t_e}."
            )

        # t_i^F <= t_e
        if tf > t_e:
            violations.append(
                f"Task {i}: tF={tf} > project end t_e={t_e}."
            )

        # t_i^S < t_i^F
        if not ts < tf:
            violations.append(
                f"Task {i}: tS={ts} must be smaller than tF={tf}."
            )

        # 0 < t_i^F
        if not tf > 0:
            violations.append(
                f"Task {i}: tF={tf} must satisfy 0 < tF."
            )

        # Tasks in L start at t0 = 0.
        if i in L and ts != 0:
            violations.append(
                f"Task {i}: belongs to L but tS={ts}, expected 0."
            )

        # Tasks not in L start after t0.
        if i not in L and not ts > 0:
            violations.append(
                f"Task {i}: not in L but tS={ts}, expected tS > 0."
            )

        # 1-based day indices.
        dS = day_from_time(ts)
        dF = day_from_time(tf)

        # 1 <= d_i^S <= lambda
        if not (1 <= dS <= lam):
            violations.append(
                f"Task {i}: dS={dS} is outside [1, lambda={lam}]."
            )

        # 1 <= d_i^F <= lambda
        if not (1 <= dF <= lam):
            violations.append(
                f"Task {i}: dF={dF} is outside [1, lambda={lam}]."
            )

        # d_i^S <= d_i^F
        if dS > dF:
            violations.append(
                f"Task {i}: dS={dS} > dF={dF}."
            )

        # d_i^S - 1 <= t_i^S/12 < d_i^S
        if not time_day_consistent(ts, dS):
            violations.append(
                f"Task {i}: tS={ts} is inconsistent with dS={dS}."
            )

        # d_i^F - 1 <= t_i^F/12 < d_i^F
        if not time_day_consistent(tf, dF):
            violations.append(
                f"Task {i}: tF={tf} is inconsistent with dF={dF}."
            )

        # D_i = d_i^F - d_i^S + 1 <= lambda
        D_i = dF - dS + 1
        if D_i > lam:
            violations.append(
                f"Task {i}: D_i={D_i} > lambda={lam}."
            )

        # --------------------------------------------------------
        # Interruption / working-day constraint:
        # sum h(T_i,k) <= D_i
        # --------------------------------------------------------
        invalid_days = [
            k for k in task.working_days
            if k < dS or k > dF
        ]

        if invalid_days:
            violations.append(
                f"Task {i}: working days {invalid_days} are outside "
                f"[{dS}, {dF}]."
            )

        h_sum = sum(
            1 for k in range(dS, dF + 1)
            if k in task.working_days
        )

        if h_sum > D_i:
            violations.append(
                f"Task {i}: worked days={h_sum} > D_i={D_i}."
            )

        if not task.working_days:
            violations.append(
                f"Task {i}: W_i is empty."
            )

        # --------------------------------------------------------
        # Single-skill constraint.
        # Each task has one skill for its whole duration.
        # --------------------------------------------------------
        if not isinstance(task.required_skill, str) or not task.required_skill:
            violations.append(
                f"Task {i}: exactly one required skill must be specified."
            )

        # C_k(T_i,j) <= h(T_i,k)
        # With one fixed task skill, the skill requirement is active
        # only on days where the task is actually worked.
        for k in range(dS, dF + 1):
            C_k = 1 if k in task.working_days else 0
            h_ik = 1 if k in task.working_days else 0

            if C_k > h_ik:
                violations.append(
                    f"Task {i}, day {k}: C_k <= h is violated."
                )

        # --------------------------------------------------------
        # Employee assignment constraints.
        # --------------------------------------------------------
        for k in range(dS, dF + 1):
            h_ik = 1 if k in task.working_days else 0
            assigned = task.assignments.get(k, [])

            # epsilon_k(T_i,p) <= h(T_i,k)
            if assigned and not h_ik:
                violations.append(
                    f"Task {i}, day {k}: employee assigned while "
                    "the task is not being worked."
                )

            # h(T_i,k) <= sum_p epsilon_k(T_i,p)
            if h_ik and not assigned:
                violations.append(
                    f"Task {i}, day {k}: task is worked but no employee "
                    "is assigned."
                )

            # epsilon_k(T_i,p) <= Psi(T_i,p)
            # Assigned employee must have the required skill.
            for employee_id in assigned:
                if employee_id not in employees:
                    violations.append(
                        f"Task {i}, day {k}: unknown employee {employee_id}."
                    )
                elif task.required_skill not in employees[employee_id].skills:
                    violations.append(
                        f"Task {i}, day {k}: employee {employee_id} lacks "
                        f"skill '{task.required_skill}'."
                    )

        # sum_k epsilon_k(T_i,p) <= card(W_i)
        for employee_id in employees:
            count = sum(
                employee_id in task.assignments.get(k, [])
                for k in range(dS, dF + 1)
            )

            if count > len(task.working_days):
                violations.append(
                    f"Task {i}, employee {employee_id}: assignments={count} "
                    f"> card(W_i)={len(task.working_days)}."
                )

    # ------------------------------------------------------------
    # Employee capacity:
    # sum_i epsilon_k(T_i,p) <= 1
    # An employee can work on at most one task per day.
    # ------------------------------------------------------------
    for employee_id in employees:
        for k in range(1, lam + 1):
            tasks_for_employee = [
                task.task_id
                for task in tasks
                if employee_id in task.assignments.get(k, [])
            ]

            if len(tasks_for_employee) > 1:
                violations.append(
                    f"Employee {employee_id}, day {k}: assigned to multiple "
                    f"tasks {tasks_for_employee}; maximum is one task/day."
                )

    # Assignment must belong to W_i and to the task's day interval.
    for task in tasks:
        dS = day_from_time(task.start_time)
        dF = day_from_time(task.finish_time)

        for k, assigned in task.assignments.items():
            if assigned and k not in task.working_days:
                violations.append(
                    f"Task {task.task_id}: assignment on day {k} but "
                    "day is not in W_i."
                )

            if k < dS or k > dF:
                violations.append(
                    f"Task {task.task_id}: assignment on day {k} is outside "
                    f"[{dS}, {dF}]."
                )

    violations = list(dict.fromkeys(violations))
    return len(violations) == 0, violations


if __name__ == "__main__":

    employees = [
        Employee(1, {"Python", "AI"}),
        Employee(2, {"Optimization"}),
        Employee(3, {"Python", "Optimization"}),
    ]

    tasks = [
        Task(
            task_id=1,
            start_time=0,
            finish_time=20,
            required_skill="Python",
            working_days={1, 2},
            assignments={
                1: [1],
                2: [3],
            },
        ),
        Task(
            task_id=2,
            start_time=12,
            finish_time=36,
            required_skill="Optimization",
            working_days={2, 3},
            assignments={
                2: [2],
                3: [3],
            },
        ),
    ]

    schedule = ProjectSchedule(
        tasks=tasks,
        employees=employees,
        start_tasks={1},
    )

    feasible, violations = verify_schedule(schedule)

    print("=" * 60)
    print("SCHEDULE FEASIBILITY")
    print("=" * 60)
    print("Feasible:", feasible)

    if violations:
        print("\nConstraint violations:")
        for violation in violations:
            print(" -", violation)
    else:
        print("All implemented constraints are satisfied.")
