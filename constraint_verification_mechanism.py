"""
The verifier checks the constraints one-by-one and reports every
violation rather than stopping at the first failure.

Day convention used by the explicit constraints in the document:
    d = floor(t / 12) + 1
so day 1 corresponds to t in [0, 12).

Variables represented here:
    tS[i], tF[i]        task start/finish times (hours)
    h[i][k]             task-work indicator
    C[i][j][k]          skill-requirement indicator
    alpha[i][j]         task-skill indicator
    eps[i][p][k]        employee-task-day assignment
    Psi[i][p]            employee p has the correct skill for task i
    dS[i], dF[i]        start/finish work days
    W[i]                worked days of task i
    lam                 project length in days
    te                  project end time in hours

IMPORTANT:
O = the set of tasks that start at time 0.
"""

from math import ceil, isclose
from typing import Any, Dict, List, Set


TOL = 1e-9


def _violation(code: str, message: str) -> Dict[str, str]:
    return {"constraint": code, "message": message}


def verify_constraints(
    tS: List[float],
    tF: List[float],
    h: List[List[int]],
    C: List[List[List[int]]],
    alpha: List[List[int]],
    eps: List[List[List[int]]],
    Psi: List[List[int]],
    dS: List[int],
    dF: List[int],
    W: List[Set[int]],
    te: float,
    lam: int,
    O: Set[int],
) -> Dict[str, Any]:
    """
    Verify all displayed constraints from the document.

    Indexing:
        tasks      i = 0..N-1
        skills     j = 0..M-1
        employees  p = 0..P-1
        days       k = 1..lambda

    O uses 0-based task indices in Python.

    Returns:
        {
            "satisfied": bool,
            "violations": [...],
            "checked_constraints": 21
        }
    """

    violations = []

    N = len(tS)
    if not (
        len(tF) == N
        and len(h) == N
        and len(C) == N
        and len(alpha) == N
        and len(eps) == N
        and len(dS) == N
        and len(dF) == N
        and len(W) == N
        and len(Psi) == N
    ):
        return {
            "satisfied": False,
            "violations": [
                _violation(
                    "INPUT",
                    "Task-dependent arrays do not all have the same number of tasks."
                )
            ],
            "checked_constraints": 0,
        }

    M = len(alpha[0]) if N else 0
    P = len(Psi[0]) if N else 0

    # Basic dimensions.
    for i in range(N):
        if len(alpha[i]) != M:
            violations.append(
                _violation("INPUT", f"alpha[{i}] has incorrect skill dimension.")
            )
        if len(Psi[i]) != P:
            violations.append(
                _violation("INPUT", f"Psi[{i}] has incorrect employee dimension.")
            )
        if len(h[i]) != lam:
            violations.append(
                _violation("INPUT", f"h[{i}] must contain lambda={lam} days.")
            )
        if len(C[i]) != M:
            violations.append(
                _violation("INPUT", f"C[{i}] has incorrect skill dimension.")
            )
        else:
            for j in range(M):
                if len(C[i][j]) != lam:
                    violations.append(
                        _violation(
                            "INPUT",
                            f"C[{i}][{j}] must contain lambda={lam} days."
                        )
                    )

        if len(eps[i]) != P:
            violations.append(
                _violation("INPUT", f"eps[{i}] has incorrect employee dimension.")
            )
        else:
            for p in range(P):
                if len(eps[i][p]) != lam:
                    violations.append(
                        _violation(
                            "INPUT",
                            f"eps[{i}][{p}] must contain lambda={lam} days."
                        )
                    )

    # -----------------------------
    # Constraint 1
    # forall i: t_i^S <= t_e
    # -----------------------------
    for i in range(N):
        if tS[i] > te + TOL:
            violations.append(
                _violation("C1", f"Task {i+1}: tS={tS[i]} > te={te}.")
            )

    # -----------------------------
    # Constraint 2
    # forall i: t_i^F <= t_e
    # -----------------------------
    for i in range(N):
        if tF[i] > te + TOL:
            violations.append(
                _violation("C2", f"Task {i+1}: tF={tF[i]} > te={te}.")
            )

    # -----------------------------
    # Constraint 3
    # forall i: t_i^S < t_i^F
    # -----------------------------
    for i in range(N):
        if not (tS[i] < tF[i] - TOL):
            violations.append(
                _violation("C3", f"Task {i+1}: tS={tS[i]} is not < tF={tF[i]}.")
            )

    # -----------------------------
    # Constraint 4
    # forall i: 0 < t_i^F
    # -----------------------------
    for i in range(N):
        if not (tF[i] > TOL):
            violations.append(
                _violation("C4", f"Task {i+1}: tF={tF[i]} is not > 0.")
            )

    # -----------------------------
    # Constraint 5
    # forall i in O: t_i^S = 0
    # -----------------------------
    for i in O:
        if i < 0 or i >= N:
            violations.append(
                _violation("INPUT", f"O contains invalid task index {i}.")
            )
        elif not isclose(tS[i], 0.0, abs_tol=TOL):
            violations.append(
                _violation("C5", f"Task {i+1} in O: tS={tS[i]} != 0.")
            )

    # -----------------------------
    # Constraint 6
    # forall i not in O: 0 < t_i^S
    # -----------------------------
    for i in range(N):
        if i not in O and not (tS[i] > TOL):
            violations.append(
                _violation(
                    "C6",
                    f"Task {i+1} not in O: tS={tS[i]} is not > 0."
                )
            )

    # -----------------------------
    # Constraint 7
    # forall i: D_i <= lambda
    # D_i = dF_i - dS_i + 1
    # -----------------------------
    for i in range(N):
        D_i = dF[i] - dS[i] + 1
        if D_i > lam:
            violations.append(
                _violation("C7", f"Task {i+1}: D={D_i} > lambda={lam}.")
            )

    # -----------------------------
    # Constraint 8
    # forall i: 1 <= dS_i <= lambda
    # -----------------------------
    for i in range(N):
        if not (1 <= dS[i] <= lam):
            violations.append(
                _violation(
                    "C8",
                    f"Task {i+1}: dS={dS[i]} outside [1,{lam}]."
                )
            )

    # -----------------------------
    # Constraint 9
    # forall i: 1 <= dF_i <= lambda
    # -----------------------------
    for i in range(N):
        if not (1 <= dF[i] <= lam):
            violations.append(
                _violation(
                    "C9",
                    f"Task {i+1}: dF={dF[i]} outside [1,{lam}]."
                )
            )

    # -----------------------------
    # Constraint 10
    # forall i: dS_i <= dF_i
    # -----------------------------
    for i in range(N):
        if dS[i] > dF[i]:
            violations.append(
                _violation(
                    "C10",
                    f"Task {i+1}: dS={dS[i]} > dF={dF[i]}."
                )
            )

    # -----------------------------
    # Constraint 11
    # forall i:
    # dS_i - 1 <= tS_i / 12 < dS_i
    # -----------------------------
    for i in range(N):
        x = tS[i] / 12.0
        if not (dS[i] - 1 - TOL <= x < dS[i] - TOL):
            violations.append(
                _violation(
                    "C11",
                    f"Task {i+1}: dS-1 <= tS/12 < dS violated "
                    f"({dS[i]-1} <= {x} < {dS[i]})."
                )
            )

    # -----------------------------
    # Constraint 12
    # forall i:
    # dF_i - 1 <= tF_i / 12 < dF_i
    # -----------------------------
    for i in range(N):
        x = tF[i] / 12.0
        if not (dF[i] - 1 - TOL <= x < dF[i] - TOL):
            violations.append(
                _violation(
                    "C12",
                    f"Task {i+1}: dF-1 <= tF/12 < dF violated "
                    f"({dF[i]-1} <= {x} < {dF[i]})."
                )
            )

    # -----------------------------
    # Constraint 13
    # forall i:
    # sum_{k=dS_i}^{dF_i} h(i,k) <= D_i
    # -----------------------------
    for i in range(N):
        D_i = dF[i] - dS[i] + 1
        total = sum(h[i][k - 1] for k in range(dS[i], dF[i] + 1))
        if total > D_i:
            violations.append(
                _violation(
                    "C13",
                    f"Task {i+1}: sum h={total} > D={D_i}."
                )
            )

    # -----------------------------
    # Constraint 14
    # forall i,j,k: C(i,j,k) <= h(i,k)
    # -----------------------------
    for i in range(N):
        for j in range(M):
            for k in range(1, lam + 1):
                if C[i][j][k - 1] > h[i][k - 1]:
                    violations.append(
                        _violation(
                            "C14",
                            f"Task {i+1}, skill {j+1}, day {k}: "
                            f"C={C[i][j][k-1]} > h={h[i][k-1]}."
                        )
                    )

    # -----------------------------
    # Constraint 15
    # forall i,p,k: eps(i,p,k) <= h(i,k)
    # -----------------------------
    for i in range(N):
        for p in range(P):
            for k in range(1, lam + 1):
                if eps[i][p][k - 1] > h[i][k - 1]:
                    violations.append(
                        _violation(
                            "C15",
                            f"Task {i+1}, employee {p+1}, day {k}: "
                            f"eps={eps[i][p][k-1]} > h={h[i][k-1]}."
                        )
                    )

    # -----------------------------
    # Constraint 16
    # forall i,j,k: C(i,j,k) <= alpha(i,j)
    # -----------------------------
    for i in range(N):
        for j in range(M):
            for k in range(1, lam + 1):
                if C[i][j][k - 1] > alpha[i][j]:
                    violations.append(
                        _violation(
                            "C16",
                            f"Task {i+1}, skill {j+1}, day {k}: "
                            f"C={C[i][j][k-1]} > alpha={alpha[i][j]}."
                        )
                    )

    # -----------------------------
    # Constraint 17
    # forall i,j:
    # sum_{k in W_i} C(i,j,k)
    # <= card(W_i) * alpha(i,j)
    # -----------------------------
    for i in range(N):
        Wi = set(W[i])
        for j in range(M):
            total = sum(C[i][j][k - 1] for k in Wi)
            rhs = len(Wi) * alpha[i][j]
            if total > rhs:
                violations.append(
                    _violation(
                        "C17",
                        f"Task {i+1}, skill {j+1}: "
                        f"sum_W C={total} > |W|*alpha={rhs}."
                    )
                )

    # -----------------------------
    # Constraint 18
    # forall i,p,k: eps(i,p,k) <= Psi(i,p)
    # -----------------------------
    for i in range(N):
        for p in range(P):
            for k in range(1, lam + 1):
                if eps[i][p][k - 1] > Psi[i][p]:
                    violations.append(
                        _violation(
                            "C18",
                            f"Task {i+1}, employee {p+1}, day {k}: "
                            f"eps={eps[i][p][k-1]} > Psi={Psi[i][p]}."
                        )
                    )

    # -----------------------------
    # Constraint 19
    # forall i,k:
    # h(i,k) <= sum_p eps(i,p,k)
    # -----------------------------
    for i in range(N):
        for k in range(1, lam + 1):
            assigned = sum(eps[i][p][k - 1] for p in range(P))
            if h[i][k - 1] > assigned:
                violations.append(
                    _violation(
                        "C19",
                        f"Task {i+1}, day {k}: h={h[i][k-1]} > "
                        f"assigned employees={assigned}."
                    )
                )

    # -----------------------------
    # Constraint 20
    # forall i,p:
    # sum_{k=dS_i}^{dF_i} eps(i,p,k) <= card(W_i)
    # -----------------------------
    for i in range(N):
        Wi = set(W[i])
        for p in range(P):
            total = sum(
                eps[i][p][k - 1]
                for k in range(dS[i], dF[i] + 1)
            )
            rhs = len(Wi)
            if total > rhs:
                violations.append(
                    _violation(
                        "C20",
                        f"Task {i+1}, employee {p+1}: "
                        f"assignment sum={total} > |W|={rhs}."
                    )
                )

    # -----------------------------
    # Constraint 21
    # forall p,k:
    # sum_i eps(i,p,k) <= 1
    # -----------------------------
    for p in range(P):
        for k in range(1, lam + 1):
            total = sum(eps[i][p][k - 1] for i in range(N))
            if total > 1:
                violations.append(
                    _violation(
                        "C21",
                        f"Employee {p+1}, day {k}: "
                        f"assigned to {total} tasks (>1)."
                    )
                )

    return {
        "satisfied": len(violations) == 0,
        "violations": violations,
        "checked_constraints": 21,
    }


def print_verification_report(result: Dict[str, Any]) -> None:
    """Print a complete human-readable verification report."""
    print("=" * 70)
    print("PROJECT-SCHEDULING CONSTRAINT VERIFICATION")
    print("=" * 70)

    if result["satisfied"]:
        print("RESULT: ALL CHECKED CONSTRAINTS ARE SATISFIED")
    else:
        print("RESULT: CONSTRAINT VIOLATIONS FOUND")
        print(f"Number of violations: {len(result['violations'])}")
        print()

        for v in result["violations"]:
            print(f"[{v['constraint']}] {v['message']}")

    print("=" * 70)


if __name__ == "__main__":
    # ------------------------------------------------------------
    # Minimal example showing how to call the verifier.
    # Replace this data with a GA-generated scheduling candidate.
    # ------------------------------------------------------------

    N = 2
    M = 2
    P = 2
    lam = 2

    # Task start/finish times in hours.
    tS = [0.0, 12.0]
    tF = [12.0, 24.0]
    te = 24.0

    # Start/finish days (1-based).
    dS = [1, 2]
    dF = [1, 2]

    # Worked-day indicator h[i][k], k=1..lambda.
    h = [
        [1, 0],
        [0, 1],
    ]

    # alpha[i][j]: task i requires skill j.
    alpha = [
        [1, 0],
        [0, 1],
    ]

    # C[i][j][k]: skill j is required for task i on day k.
    C = [
        [
            [1, 0],
            [0, 0],
        ],
        [
            [0, 0],
            [0, 1],
        ],
    ]

    # Psi[i][p]: employee p has the correct skill for task i.
    Psi = [
        [1, 0],
        [0, 1],
    ]

    # eps[i][p][k]: employee p works on task i on day k.
    eps = [
        [
            [1, 0],
            [0, 0],
        ],
        [
            [0, 0],
            [0, 1],
        ],
    ]

    # W_i: worked days. 1-based day numbers.
    W = [
        {1},
        {2},
    ]

    # O must come from the scheduling model.
    O = {0}

    result = verify_constraints(
        tS=tS,
        tF=tF,
        h=h,
        C=C,
        alpha=alpha,
        eps=eps,
        Psi=Psi,
        dS=dS,
        dF=dF,
        W=W,
        te=te,
        lam=lam,
        O=O,
    )

    print_verification_report(result)
