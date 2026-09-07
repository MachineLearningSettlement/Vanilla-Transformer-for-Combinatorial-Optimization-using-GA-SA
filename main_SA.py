from __future__ import annotations

import copy
import importlib.util
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Set, Tuple

import numpy as np


# ============================================================================
# SIMULATED ANNEALING + VANILLA TRANSFORMER/MLP
# ============================================================================
#
# Pipeline implemented here:
#
#   Initial feasible solution S
#          |
#          v
#   Generate mutated neighbor S'
#          |
#          v
#   Vanilla Transformer + MLP
#          |
#          v
#   Predicted-time difference:
#       | predicted_time(S') - exact_time(S) |
#          |
#      threshold test
#       /       \
#     fail      pass
#      |          |
#   discard       v
#   S'        Exact constraint verification
#                  |
#              infeasible?
#               /      \
#             yes       no
#              |         |
#           discard      v
#                       Exact time f(S')
#                          |
#                          v
#                  Standard SA acceptance rule
#                          |
#              +-----------+-----------+
#              |                       |
#          better/equal             worse
#              |                       |
#          accept              exp(-Delta f / T)
#                                      |
#                                random test
#                                      |
#                              accept or reject
#                                      |
#                                      v
#                           Update temperature T
#                                      |
#                                      v
#                            Next SA iteration
#
# IMPORTANT:
# - Vanilla is ONLY a pre-filter for generated neighbors.
# - The SA acceptance rule uses the EXACT project duration after
#   constraint verification.
# - Temperature is updated after every SA iteration, independently
#   of whether the accepted neighbor was better, worse, or rejected.
# - The project-specific M, P, model architecture, thresholds, and
#   trained weights remain configuration placeholders.
# - Days are 1-based: day k corresponds to [(k-1)*12, k*12).
# ============================================================================


ROOT = Path(__file__).resolve().parent

VANILLA_FILE = ROOT / "Vanilla_MLP_for_time_prediction.py"
VERIFIER_FILE = ROOT / "constraint_verification_mechanism.py"

VANILLA_WEIGHTS = ROOT / "vanilla_transformer_weights.pt"
MLP_WEIGHTS = ROOT / "time_prediction_mlp_weights.pt"


# ============================================================================
# CONFIGURATION
# ============================================================================


INITIAL_TEMPERATURE = None
COOLING_RATE = None
PROMISING_THRESHOLD = None
MAX_ITERATIONS = None

"""
SET YOUR PARAMS BASED ON YOUR TAILORED PROJECT
"""

# Maximum number of generated neighbors examined inside one SA iteration
# before failing explicitly instead of silently looping forever.
MAX_NEIGHBOR_TRIALS = None

# Maximum attempts for finding the initial feasible solution.
MAX_INITIAL_TRIALS = None

M = None
P = None

D_MODEL = None
NHEAD = None
NUM_LAYERS = None
DIM_FEEDFORWARD = None
MLP_HIDDEN_DIMS = None


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Problem:
    alpha: List[List[int]]
    Psi: List[List[int]]
    lam: int
    O: Set[int]

    @property
    def N(self) -> int:
        return len(self.alpha)

    @property
    def M(self) -> int:
        return len(self.alpha[0]) if self.alpha else 0

    @property
    def P(self) -> int:
        return len(self.Psi[0]) if self.Psi else 0


@dataclass
class Schedule:
    tS: List[float]
    tF: List[float]
    h: List[List[int]]
    C: List[List[List[int]]]
    alpha: List[List[int]]
    eps: List[List[List[int]]]
    Psi: List[List[int]]
    dS: List[int]
    dF: List[int]
    W: List[Set[int]]
    te: float
    lam: int


# ============================================================================
# MODULE LOADING
# ============================================================================

def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)

    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


vanilla = load_module("vanilla_time_prediction", VANILLA_FILE)
verifier = load_module("constraint_verifier", VERIFIER_FILE)


# ============================================================================
# VALIDATION
# ============================================================================

def validate_problem(problem: Problem) -> None:
    if problem.N == 0:
        raise ValueError("The scheduling problem must contain at least one task.")

    if problem.lam <= 0:
        raise ValueError("Problem.lam must be positive.")

    if not problem.alpha:
        raise ValueError("alpha cannot be empty.")

    if not problem.Psi:
        raise ValueError("Psi cannot be empty.")

    if len(problem.alpha) != len(problem.Psi):
        raise ValueError("alpha and Psi must contain the same number of tasks.")

    m = problem.M
    p = problem.P

    if m <= 0:
        raise ValueError("The problem must contain at least one skill.")

    if p <= 0:
        raise ValueError("The problem must contain at least one employee.")

    if M != m or P != p:
        raise ValueError("M/P do not match the scheduling instance.")

    for i in range(problem.N):
        if len(problem.alpha[i]) != m:
            raise ValueError(f"alpha[{i}] has an incorrect skill dimension.")

        if len(problem.Psi[i]) != p:
            raise ValueError(f"Psi[{i}] has an incorrect employee dimension.")

        # The problem statement specifies one required skill per task.
        if sum(int(x) for x in problem.alpha[i]) != 1:
            raise ValueError(
                f"Task {i + 1} must require exactly one skill according "
                "to the scheduling problem definition."
            )

        # Every task must have at least one eligible employee.
        if not any(problem.Psi[i]):
            raise ValueError(
                f"Task {i + 1} has no employee with the required skill."
            )

    if not problem.O:
        raise ValueError("O must be non-empty because the project starts at t=0.")

    if not problem.O.issubset(set(range(problem.N))):
        raise ValueError("O contains an invalid task index.")


def validate_config(problem: Problem) -> None:
    values = {
        "INITIAL_TEMPERATURE": INITIAL_TEMPERATURE,
        "COOLING_RATE": COOLING_RATE,
        "PROMISING_THRESHOLD": PROMISING_THRESHOLD,
        "MAX_ITERATIONS": MAX_ITERATIONS,
        "MAX_NEIGHBOR_TRIALS": MAX_NEIGHBOR_TRIALS,
        "MAX_INITIAL_TRIALS": MAX_INITIAL_TRIALS,
        "M": M,
        "P": P,
        "D_MODEL": D_MODEL,
        "NHEAD": NHEAD,
        "NUM_LAYERS": NUM_LAYERS,
        "DIM_FEEDFORWARD": DIM_FEEDFORWARD,
        "MLP_HIDDEN_DIMS": MLP_HIDDEN_DIMS,
    }

    missing = [name for name, value in values.items() if value is None]
    if missing:
        raise ValueError(
            "Set these configuration values: " + ", ".join(missing)
        )

    if INITIAL_TEMPERATURE <= 0:
        raise ValueError("INITIAL_TEMPERATURE must be > 0.")

    if not (0 < COOLING_RATE < 1):
        raise ValueError("COOLING_RATE must satisfy 0 < COOLING_RATE < 1.")

    if PROMISING_THRESHOLD < 0:
        raise ValueError("PROMISING_THRESHOLD must be >= 0.")

    if MAX_ITERATIONS <= 0:
        raise ValueError("MAX_ITERATIONS must be > 0.")

    if MAX_NEIGHBOR_TRIALS <= 0:
        raise ValueError("MAX_NEIGHBOR_TRIALS must be > 0.")

    if MAX_INITIAL_TRIALS <= 0:
        raise ValueError("MAX_INITIAL_TRIALS must be > 0.")

    if D_MODEL <= 0 or NHEAD <= 0 or NUM_LAYERS <= 0 or DIM_FEEDFORWARD <= 0:
        raise ValueError("Transformer dimensions must be positive.")

    if not MLP_HIDDEN_DIMS:
        raise ValueError("MLP_HIDDEN_DIMS cannot be empty.")

    for hidden_dim in MLP_HIDDEN_DIMS:
        if hidden_dim <= 0:
            raise ValueError("All MLP hidden dimensions must be positive.")

    if D_MODEL % NHEAD != 0:
        raise ValueError("D_MODEL must be divisible by NHEAD.")


# ============================================================================
# SCHEDULE GENERATION
# ============================================================================

def _random_task_times(
    start_day: int,
    finish_day: int,
    starts_at_zero: bool,
) -> Tuple[float, float]:
    """
    Generate times satisfying the strict day-interval constraints:

        dS - 1 <= tS / 12 < dS
        dF - 1 <= tF / 12 < dF
        tS < tF

    O-tasks start at t=0, hence dS=1 is enforced by the caller.
    """

    # Strictly inside the finish-day interval.
    finish_time = (
        (finish_day - 1) * 12.0
        + random.uniform(1e-6, 12.0 - 1e-6)
    )

    if starts_at_zero:
        start_time = 0.0
    elif start_day < finish_day:
        start_time = (
            (start_day - 1) * 12.0
            + random.uniform(1e-6, 12.0 - 1e-6)
        )
    else:
        # Same day: explicitly enforce tS < tF.
        day_start = (start_day - 1) * 12.0
        start_time = random.uniform(
            day_start + 1e-6,
            finish_time - 1e-6,
        )

    if not start_time < finish_time:
        raise RuntimeError("Generated task times violate tS < tF.")

    return start_time, finish_time


def random_schedule(problem: Problem) -> Schedule:
    tS: List[float] = []
    tF: List[float] = []
    h: List[List[int]] = []
    C: List[List[List[int]]] = []
    eps: List[List[List[int]]] = []
    dS: List[int] = []
    dF: List[int] = []
    W: List[Set[int]] = []

    for i in range(problem.N):

        # A task in O must start at t=0. Under the explicit day
        # constraints this means dS=1.
        if i in problem.O:
            start_day = 1
        else:
            start_day = random.randint(1, problem.lam)

        finish_day = random.randint(start_day, problem.lam)

        days = sorted(
            random.sample(
                range(start_day, finish_day + 1),
                random.randint(1, finish_day - start_day + 1),
            )
        )

        hi = [0] * problem.lam
        for k in days:
            hi[k - 1] = 1

        Ci = [
            [0] * problem.lam
            for _ in range(problem.M)
        ]

        for j in range(problem.M):
            if problem.alpha[i][j]:
                for k in days:
                    Ci[j][k - 1] = 1

        epsi = [
            [0] * problem.lam
            for _ in range(problem.P)
        ]

        eligible = [
            employee
            for employee in range(problem.P)
            if problem.Psi[i][employee]
        ]

        for k in days:
            employee = random.choice(eligible)
            epsi[employee][k - 1] = 1

        start_time, finish_time = _random_task_times(
            start_day,
            finish_day,
            starts_at_zero=(i in problem.O),
        )

        tS.append(start_time)
        tF.append(finish_time)
        h.append(hi)
        C.append(Ci)
        eps.append(epsi)
        dS.append(start_day)
        dF.append(finish_day)
        W.append(set(days))

    te = max(tF)

    return Schedule(
        tS=tS,
        tF=tF,
        h=h,
        C=C,
        alpha=copy.deepcopy(problem.alpha),
        eps=eps,
        Psi=copy.deepcopy(problem.Psi),
        dS=dS,
        dF=dF,
        W=W,
        te=te,
        lam=problem.lam,
    )


# ============================================================================
# VANILLA INPUT REPRESENTATION
# ============================================================================

def schedule_tensor(schedule: Schedule) -> np.ndarray:
    """
    Each task-day token is:

        [h(T_i,k),
         C_k(T_i,1), ..., C_k(T_i,M),
         epsilon_k(T_i,1), ..., epsilon_k(T_i,P)]

    Therefore input_dim = 1 + M + P.
    """

    rows = []

    for i in range(len(schedule.tS)):
        for k in range(schedule.lam):
            rows.append(
                [
                    schedule.h[i][k],
                    *[
                        schedule.C[i][j][k]
                        for j in range(len(schedule.C[i]))
                    ],
                    *[
                        schedule.eps[i][p][k]
                        for p in range(len(schedule.eps[i]))
                    ],
                ]
            )

    return np.asarray(rows, dtype=np.float32)


def build_predictor():
    model = vanilla.SchedulingTimePredictor(
        input_dim=1 + M + P,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        mlp_hidden_dims=MLP_HIDDEN_DIMS,
    ).to(vanilla.DEVICE)

    vanilla.load_weights(
        model,
        VANILLA_WEIGHTS,
        MLP_WEIGHTS,
    )

    model.eval()
    return model


def predict_time(model, schedule: Schedule) -> float:
    return float(
        vanilla.predict_project_time(
            model,
            schedule_tensor(schedule),
        )
    )


# ============================================================================
# EXACT CONSTRAINT VERIFICATION
# ============================================================================

def verify(schedule: Schedule, problem: Problem) -> bool:
    result = verifier.verify_constraints(
        tS=schedule.tS,
        tF=schedule.tF,
        h=schedule.h,
        C=schedule.C,
        alpha=schedule.alpha,
        eps=schedule.eps,
        Psi=schedule.Psi,
        dS=schedule.dS,
        dF=schedule.dF,
        W=schedule.W,
        te=schedule.te,
        lam=schedule.lam,
        O=problem.O,
    )

    return bool(result["satisfied"])


def exact_time(schedule: Schedule) -> float:
    """
    Exact project duration/makespan for this formulation.

    Since O is non-empty and tasks in O start at t=0,
    the project end time is max_i tF_i.
    """

    schedule.te = max(schedule.tF)
    return float(schedule.te)


def find_initial_feasible_solution(problem: Problem) -> Schedule:
    for _ in range(MAX_INITIAL_TRIALS):
        candidate = random_schedule(problem)

        if verify(candidate, problem):
            candidate.te = exact_time(candidate)
            return candidate

    raise RuntimeError(
        "Could not generate a feasible initial solution within "
        "MAX_INITIAL_TRIALS."
    )


# ============================================================================
# SA NEIGHBOR / MUTATION OPERATOR
# ============================================================================

def mutate(schedule: Schedule, problem: Problem) -> Schedule:
    """
    Generate one neighboring scheduling solution by modifying one task.

    This is the SA neighborhood move. Conceptually it plays the role that
    mutation plays in GA, but SA works with one current solution and one
    generated neighbor at a time.
    """

    neighbor = copy.deepcopy(schedule)

    i = random.randrange(problem.N)

    if i in problem.O:
        start_day = 1
    else:
        start_day = random.randint(1, problem.lam)

    finish_day = random.randint(start_day, problem.lam)

    days = sorted(
        random.sample(
            range(start_day, finish_day + 1),
            random.randint(1, finish_day - start_day + 1),
        )
    )

    neighbor.dS[i] = start_day
    neighbor.dF[i] = finish_day
    neighbor.W[i] = set(days)

    neighbor.h[i] = [
        int(k in neighbor.W[i])
        for k in range(1, problem.lam + 1)
    ]

    neighbor.C[i] = [
        [
            int(
                neighbor.alpha[i][j] and
                k in neighbor.W[i]
            )
            for k in range(1, problem.lam + 1)
        ]
        for j in range(problem.M)
    ]

    neighbor.eps[i] = [
        [0] * problem.lam
        for _ in range(problem.P)
    ]

    eligible = [
        employee
        for employee in range(problem.P)
        if problem.Psi[i][employee]
    ]

    for k in days:
        employee = random.choice(eligible)
        neighbor.eps[i][employee][k - 1] = 1

    start_time, finish_time = _random_task_times(
        start_day,
        finish_day,
        starts_at_zero=(i in problem.O),
    )

    neighbor.tS[i] = start_time
    neighbor.tF[i] = finish_time

    neighbor.te = max(neighbor.tF)

    return neighbor


# ============================================================================
# VANILLA PRE-FILTER LOOP
# ============================================================================

def generate_promising_feasible_neighbor(
    model,
    current: Schedule,
    current_exact_time: float,
    problem: Problem,
) -> Tuple[Schedule, float, float, int]:
    """
    Generate neighbors until one passes BOTH:

    1. Vanilla predicted-difference threshold:
           |predicted_time(S') - exact_time(S)|
           <= PROMISING_THRESHOLD

    2. Exact constraint verification.

    Returns:
        promising_feasible_neighbor,
        predicted_time,
        exact_time,
        number_of_trials
    """

    for trial in range(1, MAX_NEIGHBOR_TRIALS + 1):

        neighbor = mutate(current, problem)

        predicted = predict_time(model, neighbor)

        predicted_delta = abs(
            predicted - current_exact_time
        )

        if predicted_delta > PROMISING_THRESHOLD:
            continue

        # Vanilla has only screened the neighbor.
        # Exact feasibility is checked before the SA acceptance rule.
        if not verify(neighbor, problem):
            continue

        neighbor_exact_time = exact_time(neighbor)

        return (
            neighbor,
            predicted,
            neighbor_exact_time,
            trial,
        )

    raise RuntimeError(
        "No promising feasible neighbor was found within "
        "MAX_NEIGHBOR_TRIALS."
    )


# ============================================================================
# STANDARD SA ACCEPTANCE RULE
# ============================================================================

def acceptance_probability(
    current_exact_time: float,
    neighbor_exact_time: float,
    temperature: float,
) -> float:
    """
    Standard SA acceptance probability for minimization.

    Better/equal neighbor:
        probability = 1

    Worse neighbor:
        probability = exp(-(f(S') - f(S)) / T)
    """

    delta_f = neighbor_exact_time - current_exact_time

    if delta_f <= 0:
        return 1.0

    probability = math.exp(
        -delta_f / temperature
    )

    # Numerical protection; mathematically this is already in (0, 1).
    return min(1.0, max(0.0, probability))


def accept_neighbor(
    current_exact_time: float,
    neighbor_exact_time: float,
    temperature: float,
) -> Tuple[bool, float]:
    """
    Apply the standard SA acceptance rule.

    Returns:
        accepted, acceptance_probability
    """

    probability = acceptance_probability(
        current_exact_time=current_exact_time,
        neighbor_exact_time=neighbor_exact_time,
        temperature=temperature,
    )

    if probability >= 1.0:
        return True, probability

    random_number = random.random()

    return random_number < probability, probability


# ============================================================================
# TEMPERATURE UPDATE
# ============================================================================

def update_temperature(temperature: float) -> float:
    """
    Geometric cooling:

        T_new = COOLING_RATE * T_old

    This is performed after every SA iteration, independently of
    the acceptance outcome.
    """

    return COOLING_RATE * temperature


# ============================================================================
# MAIN SA ALGORITHM
# ============================================================================

def run(problem: Problem):
    validate_problem(problem)
    validate_config(problem)

    model = build_predictor()

    # SA needs one current solution, not a population.
    current = find_initial_feasible_solution(problem)
    current_exact_time = exact_time(current)

    best = copy.deepcopy(current)
    best_exact_time = current_exact_time

    temperature = float(INITIAL_TEMPERATURE)

    history = []

    for iteration in range(1, MAX_ITERATIONS + 1):

        # --------------------------------------------------------------
        # 1. Generate S' repeatedly until Vanilla + exact feasibility
        #    conditions identify a promising feasible S*.
        # --------------------------------------------------------------
        (
            neighbor,
            predicted_time,
            neighbor_exact_time,
            trials,
        ) = generate_promising_feasible_neighbor(
            model=model,
            current=current,
            current_exact_time=current_exact_time,
            problem=problem,
        )

        # --------------------------------------------------------------
        # 2. Standard SA acceptance rule is now applied to S*.
        #    IMPORTANT: exact time is used here, NOT Vanilla prediction.
        # --------------------------------------------------------------
        accepted, probability = accept_neighbor(
            current_exact_time=current_exact_time,
            neighbor_exact_time=neighbor_exact_time,
            temperature=temperature,
        )

        if accepted:
            current = neighbor
            current_exact_time = neighbor_exact_time

        # --------------------------------------------------------------
        # 3. Maintain best solution found so far.
        # --------------------------------------------------------------
        if current_exact_time < best_exact_time:
            best = copy.deepcopy(current)
            best_exact_time = current_exact_time

        # --------------------------------------------------------------
        # 4. Update temperature after every iteration.
        # --------------------------------------------------------------
        temperature = update_temperature(temperature)

        history.append(
            {
                "iteration": iteration,
                "trials_to_promising_neighbor": trials,
                "predicted_neighbor_time": predicted_time,
                "exact_neighbor_time": neighbor_exact_time,
                "current_time": current_exact_time,
                "best_time": best_exact_time,
                "temperature": temperature,
                "acceptance_probability": probability,
                "accepted": accepted,
            }
        )

        print(
            f"Iteration {iteration}: "
            f"trials={trials}, "
            f"predicted={predicted_time:.6f}, "
            f"neighbor_exact={neighbor_exact_time:.6f}, "
            f"current={current_exact_time:.6f}, "
            f"best={best_exact_time:.6f}, "
            f"T={temperature:.6f}, "
            f"P_accept={probability:.6f}, "
            f"accepted={accepted}"
        )

    return best, history


# ============================================================================
# EXAMPLE CALL
# ============================================================================

if __name__ == "__main__":
    raise SystemExit(
        "Set the actual SA parameters, trained-model architecture/weights, "
        "and scheduling instance, then call run(problem)."
    )
