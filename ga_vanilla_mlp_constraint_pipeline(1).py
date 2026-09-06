from __future__ import annotations

import copy
import importlib.util
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Set

import numpy as np
import torch
from deap import base, creator, tools


ROOT = Path(__file__).resolve().parent
VANILLA_FILE = ROOT / "Vanilla_MLP_for_time_prediction.py"
VERIFIER_FILE = ROOT / "constraint_verification_mechanism.py"

VANILLA_WEIGHTS = ROOT / "vanilla_transformer_weights.pt"
MLP_WEIGHTS = ROOT / "time_prediction_mlp_weights.pt"


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

POPULATION_SIZE = None
N_BEST = None
P_OFFSPRING = None
Q_FEASIBLE = None
T_NEXT = None
PREDICTION_THRESHOLD = None
MAX_GENERATIONS = None

M = None
P = None
D_MODEL = None
NHEAD = None
NUM_LAYERS = None
DIM_FEEDFORWARD = None
MLP_HIDDEN_DIMS = None


@dataclass
class Problem:
    alpha: List[List[int]]
    Psi: List[List[int]]
    lam: int
    O: Set[int]

    @property
    def N(self):
        return len(self.alpha)

    @property
    def M(self):
        return len(self.alpha[0])

    @property
    def P(self):
        return len(self.Psi[0])


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


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


vanilla = load_module("vanilla_time_prediction", VANILLA_FILE)
verifier = load_module("constraint_verifier", VERIFIER_FILE)


def validate_config(problem: Problem):
    values = {
        "POPULATION_SIZE": POPULATION_SIZE,
        "N_BEST": N_BEST,
        "P_OFFSPRING": P_OFFSPRING,
        "Q_FEASIBLE": Q_FEASIBLE,
        "T_NEXT": T_NEXT,
        "PREDICTION_THRESHOLD": PREDICTION_THRESHOLD,
        "MAX_GENERATIONS": MAX_GENERATIONS,
        "M": M,
        "P": P,
        "D_MODEL": D_MODEL,
        "NHEAD": NHEAD,
        "NUM_LAYERS": NUM_LAYERS,
        "DIM_FEEDFORWARD": DIM_FEEDFORWARD,
        "MLP_HIDDEN_DIMS": MLP_HIDDEN_DIMS,
    }

    missing = [k for k, v in values.items() if v is None]
    if missing:
        raise ValueError("Set these configuration values: " + ", ".join(missing))

    if M != problem.M or P != problem.P:
        raise ValueError("M/P do not match the scheduling instance.")

    if not (0 < N_BEST <= POPULATION_SIZE):
        raise ValueError("Require 0 < N_BEST <= POPULATION_SIZE.")

    if not (0 < Q_FEASIBLE <= P_OFFSPRING):
        raise ValueError("Require 0 < Q_FEASIBLE <= P_OFFSPRING.")

    if not (0 < T_NEXT <= Q_FEASIBLE):
        raise ValueError("Require 0 < T_NEXT <= Q_FEASIBLE.")


def random_schedule(problem: Problem) -> Schedule:
    tS, tF, h, C, eps, dS, dF, W = [], [], [], [], [], [], [], []

    for i in range(problem.N):
        start = random.randint(1, problem.lam)
        finish = random.randint(start, problem.lam)

        days = sorted(random.sample(
            range(start, finish + 1),
            random.randint(1, finish - start + 1)
        ))

        hi = [0] * problem.lam
        for k in days:
            hi[k - 1] = 1

        Ci = [[0] * problem.lam for _ in range(problem.M)]
        for j in range(problem.M):
            if problem.alpha[i][j]:
                for k in days:
                    Ci[j][k - 1] = 1

        epsi = [[0] * problem.lam for _ in range(problem.P)]
        eligible = [p for p in range(problem.P) if problem.Psi[i][p]]
        for k in days:
            if eligible:
                p = random.choice(eligible)
                epsi[p][k - 1] = 1

        start_time = 0.0 if i in problem.O else (
            (start - 1) * 12 + random.uniform(0, 11.999999)
        )
        finish_time = finish * 12.0

        tS.append(start_time)
        tF.append(finish_time)
        h.append(hi)
        C.append(Ci)
        eps.append(epsi)
        dS.append(start)
        dF.append(finish)
        W.append(set(days))

    return Schedule(
        tS, tF, h, C, copy.deepcopy(problem.alpha), eps,
        copy.deepcopy(problem.Psi), dS, dF, W, max(tF), problem.lam
    )


def schedule_tensor(s: Schedule):
    rows = []
    for i in range(len(s.tS)):
        for k in range(s.lam):
            rows.append([
                s.h[i][k],
                *[s.C[i][j][k] for j in range(len(s.C[i]))],
                *[s.eps[i][p][k] for p in range(len(s.eps[i]))],
            ])
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

    vanilla.load_weights(model, VANILLA_WEIGHTS, MLP_WEIGHTS)
    model.eval()
    return model


def predict_time(model, schedule):
    return vanilla.predict_project_time(model, schedule_tensor(schedule))


def verify(schedule, problem):
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
    return result["satisfied"]


def crossover(a: Schedule, b: Schedule):
    a, b = copy.deepcopy(a), copy.deepcopy(b)
    if len(a.tS) < 2:
        return a, b

    left, right = sorted(random.sample(range(len(a.tS)), 2))

    for name in ("tS", "tF", "h", "C", "eps", "dS", "dF", "W"):
        x, y = getattr(a, name), getattr(b, name)
        x[left:right], y[left:right] = (
            copy.deepcopy(y[left:right]),
            copy.deepcopy(x[left:right]),
        )

    a.te, b.te = max(a.tF), max(b.tF)
    return a, b


def mutate(s: Schedule, problem: Problem):
    s = copy.deepcopy(s)
    i = random.randrange(problem.N)

    start = random.randint(1, problem.lam)
    finish = random.randint(start, problem.lam)
    days = sorted(random.sample(
        range(start, finish + 1),
        random.randint(1, finish - start + 1)
    ))

    s.dS[i], s.dF[i], s.W[i] = start, finish, set(days)
    s.h[i] = [int(k in s.W[i]) for k in range(1, problem.lam + 1)]

    s.C[i] = [
        [int(s.alpha[i][j] and k in s.W[i])
         for k in range(1, problem.lam + 1)]
        for j in range(problem.M)
    ]

    s.eps[i] = [[0] * problem.lam for _ in range(problem.P)]
    eligible = [p for p in range(problem.P) if problem.Psi[i][p]]
    for k in days:
        if eligible:
            s.eps[i][random.choice(eligible)][k - 1] = 1

    s.tS[i] = 0.0 if i in problem.O else (
        (start - 1) * 12 + random.uniform(0, 11.999999)
    )
    s.tF[i] = finish * 12.0
    s.te = max(s.tF)
    return s


def promising(model, candidates):
    scored = []
    for s in candidates:
        prediction = predict_time(model, s)
        if prediction < PREDICTION_THRESHOLD:
            scored.append((s, prediction))
    return sorted(scored, key=lambda x: x[1])


def feasible(problem, scored):
    return [(s, t) for s, t in scored if verify(s, problem)]


def initial_stage(model, population, problem):
    candidates = [copy.deepcopy(x) for x in population]
    return [
        s for s, _ in feasible(
            problem,
            promising(model, candidates)
        )
    ][:N_BEST]


def crossover_loop(model, parents, problem):
    accepted = []

    while len(accepted) < Q_FEASIBLE:
        offspring = []

        while len(offspring) < P_OFFSPRING:
            p1, p2 = random.choices(parents, k=2)
            c1, c2 = crossover(p1, p2)
            offspring.append(c1)
            if len(offspring) < P_OFFSPRING:
                offspring.append(c2)

        candidates = feasible(
            problem,
            promising(model, offspring)
        )

        accepted.extend(candidates)
        accepted.sort(key=lambda x: x[1])
        accepted = accepted[:Q_FEASIBLE]

    return [s for s, _ in accepted]


def mutation_loop(q_candidates, problem):
    accepted = []

    while len(accepted) < T_NEXT:
        for q in q_candidates:
            m = mutate(q, problem)
            if verify(m, problem):
                m.te = max(m.tF)
                accepted.append(m)

        accepted.sort(key=lambda s: s.te)
        accepted = accepted[:T_NEXT]

    return accepted


def make_deap_individual(schedule):
    return creator.ProjectScheduleIndividual(
        tS=copy.deepcopy(schedule.tS),
        tF=copy.deepcopy(schedule.tF),
        h=copy.deepcopy(schedule.h),
        C=copy.deepcopy(schedule.C),
        alpha=copy.deepcopy(schedule.alpha),
        eps=copy.deepcopy(schedule.eps),
        Psi=copy.deepcopy(schedule.Psi),
        dS=copy.deepcopy(schedule.dS),
        dF=copy.deepcopy(schedule.dF),
        W=copy.deepcopy(schedule.W),
        te=schedule.te,
        lam=schedule.lam,
    )


def run(problem: Problem):
    validate_config(problem)

    if not hasattr(creator, "FitnessProjectScheduling"):
        creator.create("FitnessProjectScheduling", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "ProjectSchedulingIndividual"):
        creator.create(
            "ProjectSchedulingIndividual",
            Schedule,
            fitness=creator.FitnessProjectScheduling,
        )

    model = build_predictor()

    population = [
        make_deap_individual(random_schedule(problem))
        for _ in range(POPULATION_SIZE)
    ]

    for generation in range(MAX_GENERATIONS):

        # Initial population -> Vanilla + MLP -> threshold -> constraints -> N
        n_candidates = initial_stage(model, population, problem)

        # N -> crossover -> P -> Vanilla + MLP -> constraints -> Q
        q_candidates = crossover_loop(model, n_candidates, problem)

        # Q -> mutation -> constraints + execution time -> T
        t_candidates = mutation_loop(q_candidates, problem)

        # T -> next population
        population = [
            make_deap_individual(s)
            for s in t_candidates
        ]

        print(
            f"Generation {generation + 1}: "
            f"N={len(n_candidates)}, "
            f"Q={len(q_candidates)}, "
            f"T={len(t_candidates)}, "
            f"best_time={min(s.te for s in t_candidates):.6f}"
        )

    return population


if __name__ == "__main__":
    raise SystemExit(
        "Set the actual GA parameters, trained-model architecture/weights, "
        "and scheduling instance before calling run(problem)."
    )
