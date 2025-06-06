
import time
import numpy as np
from pyomo.environ import *

#---- Two Turn Damage Optimization Code with Pets and Guests----

qp_problems = {('Vampire', 'Melee'): {'Q': np.array([[3.0e-05, 0.0e+00, 0.0e+00, 0.0e+00, 3.0e-05, 4.5e-06],
       [0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00],
       [0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00],
       [0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00],
       [3.0e-05, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 4.5e-06],
       [4.5e-06, 0.0e+00, 0.0e+00, 0.0e+00, 4.5e-06, 0.0e+00]]), 'c': np.array([0.265  , 0.     , 0.     , 0.     , 0.     , 0.03975])}, ('Vampire', 'Ranged'): {'Q': np.array([[0.0e+00, 2.4e-05, 0.0e+00, 0.0e+00, 2.4e-05, 0.0e+00],
       [2.4e-05, 1.2e-05, 0.0e+00, 0.0e+00, 1.2e-05, 4.5e-06],
       [0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00],
       [0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00],
       [2.4e-05, 1.2e-05, 0.0e+00, 0.0e+00, 0.0e+00, 4.5e-06],
       [0.0e+00, 4.5e-06, 0.0e+00, 0.0e+00, 4.5e-06, 0.0e+00]]), 'c': np.array([0.212  , 0.106  , 0.     , 0.     , 0.     , 0.03975])}, ('Vampire', 'Magic'): {'Q': np.array([[0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00],
       [0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00],
       [0.0e+00, 0.0e+00, 1.2e-04, 0.0e+00, 4.0e-05, 1.8e-05],
       [0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00],
       [0.0e+00, 0.0e+00, 4.0e-05, 0.0e+00, 0.0e+00, 6.0e-06],
       [0.0e+00, 0.0e+00, 1.8e-05, 0.0e+00, 6.0e-06, 0.0e+00]]), 'c': np.array([0.    , 0.    , 0.29  , 0.    , 0.    , 0.0435])}, ('Vampire', 'Spell'): {'Q': np.array([[0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00],
       [0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00],
       [0.0e+00, 0.0e+00, 1.2e-04, 0.0e+00, 4.0e-05, 1.8e-05],
       [0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00],
       [0.0e+00, 0.0e+00, 4.0e-05, 0.0e+00, 0.0e+00, 6.0e-06],
       [0.0e+00, 0.0e+00, 1.8e-05, 0.0e+00, 6.0e-06, 0.0e+00]]), 'c': np.array([0.    , 0.    , 0.29  , 0.    , 0.    , 0.0435])}, ('Werewolf', 'Melee'): {'Q': np.array([[3.0e-05, 0.0e+00, 0.0e+00, 0.0e+00, 3.0e-05, 4.5e-06],
       [0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00],
       [0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00],
       [0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00],
       [3.0e-05, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 4.5e-06],
       [4.5e-06, 0.0e+00, 0.0e+00, 0.0e+00, 4.5e-06, 0.0e+00]]), 'c': np.array([0.265  , 0.     , 0.     , 0.     , 0.     , 0.03975])}, ('Werewolf', 'Ranged'): {'Q': np.array([[0.0e+00, 2.4e-05, 0.0e+00, 0.0e+00, 2.4e-05, 0.0e+00],
       [2.4e-05, 6.0e-06, 0.0e+00, 0.0e+00, 6.0e-06, 4.5e-06],
       [0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00],
       [0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00],
       [2.4e-05, 6.0e-06, 0.0e+00, 0.0e+00, 0.0e+00, 4.5e-06],
       [0.0e+00, 4.5e-06, 0.0e+00, 0.0e+00, 4.5e-06, 0.0e+00]]), 'c': np.array([0.212  , 0.053  , 0.     , 0.     , 0.     , 0.03975])}, ('Werewolf', 'Magic'): {'Q': np.array([[0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00],
       [0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00],
       [0.0e+00, 0.0e+00, 1.2e-04, 0.0e+00, 4.0e-05, 1.8e-05],
       [0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00],
       [0.0e+00, 0.0e+00, 4.0e-05, 0.0e+00, 0.0e+00, 6.0e-06],
       [0.0e+00, 0.0e+00, 1.8e-05, 0.0e+00, 6.0e-06, 0.0e+00]]), 'c': np.array([0.    , 0.    , 0.29  , 0.    , 0.    , 0.0435])}, ('Werewolf', 'Spell'): {'Q': np.array([[0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00],
       [0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00],
       [0.0e+00, 0.0e+00, 1.2e-04, 0.0e+00, 4.0e-05, 1.8e-05],
       [0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00],
       [0.0e+00, 0.0e+00, 4.0e-05, 0.0e+00, 0.0e+00, 6.0e-06],
       [0.0e+00, 0.0e+00, 1.8e-05, 0.0e+00, 6.0e-06, 0.0e+00]]), 'c': np.array([0.    , 0.    , 0.29  , 0.    , 0.    , 0.0435])}, ('Werepyre', 'Melee'): {'Q': np.array([[7.5e-05, 0.0e+00, 7.5e-06, 0.0e+00, 7.5e-06, 4.5e-06],
       [0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00],
       [7.5e-06, 0.0e+00, 0.0e+00, 0.0e+00, 7.5e-06, 0.0e+00],
       [0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00],
       [7.5e-06, 0.0e+00, 7.5e-06, 0.0e+00, 0.0e+00, 4.5e-06],
       [4.5e-06, 0.0e+00, 0.0e+00, 0.0e+00, 4.5e-06, 0.0e+00]]), 'c': np.array([0.06625, 0.     , 0.06625, 0.     , 0.     , 0.03975])}, ('Werepyre', 'Ranged'): {'Q': np.array([[0.0e+00, 0, 0.0e+00, 0.0e+00, 0, 0],
       [0.0e+00, 1.5e-05, 0.0e+00, 0.0e+00, 1.5e-05, 4.5e-06],
       [0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00],
       [0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00],
       [0.0e+00, 1.5e-05, 0.0e+00, 0.0e+00, 0.0e+00, 4.5e-06],
       [0.0e+00, 4.5e-06, 0.0e+00, 0.0e+00, 4.5e-06, 0.0e+00]]), 'c': np.array([0.     , 0.1325 , 0.     , 0.     , 0.     , 0.03975])}, ('Werepyre', 'Magic'): {'Q': np.array([[0.0e+00, 0.0e+00, 3.6e-05, 0.0e+00, 1.2e-05, 0.0e+00],
       [0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00],
       [3.6e-05, 0.0e+00, 3.6e-05, 0.0e+00, 1.2e-05, 1.2e-05],
       [0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00],
       [1.2e-05, 0.0e+00, 1.2e-05, 0.0e+00, 0.0e+00, 1.2e-05],
       [0.0e+00, 0.0e+00, 1.2e-05, 0.0e+00, 1.2e-05, 0.0e+00]]), 'c': np.array([0.087 , 0.    , 0.087 , 0.    , 0.    , 0.0725])}, ('Werepyre', 'Spell'): {'Q': np.array([[0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00],
       [0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00],
       [0.0e+00, 0.0e+00, 1.2e-04, 0.0e+00, 4.0e-05, 1.8e-05],
       [0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00],
       [0.0e+00, 0.0e+00, 4.0e-05, 0.0e+00, 0.0e+00, 6.0e-06],
       [0.0e+00, 0.0e+00, 1.8e-05, 0.0e+00, 6.0e-06, 0.0e+00]]), 'c': np.array([0.    , 0.    , 0.29  , 0.    , 0.    , 0.0435])}}

stat_labels = ["STR", "DEX", "INT", "END", "CHA", "LUK"]
optimal_stats_qp = {}

#---- QIP Problem ----
def solve_integer_qip(Q, c):
    model = ConcreteModel()
    model.I = RangeSet(1, 6)
    model.x = Var(model.I, domain=NonNegativeIntegers, bounds=(0, 250))
    model.y = Var(model.I, domain=NonNegativeIntegers, bounds=(0, 50)) #integer multiples of 5
    model.mult_of_5 = ConstraintList()
    for i in model.I:
        model.mult_of_5.add(model.x[i] == 5 * model.y[i])
    model.total = Constraint(expr=sum(model.x[i] for i in model.I) <= 750)

    def obj_rule(m):
        return sum(Q[i-1, j-1] * m.x[i] * m.x[j] for i in m.I for j in m.I) + sum(c[i-1] * m.x[i] for i in m.I)
    model.obj = Objective(rule=obj_rule, sense=maximize)

    start = time.time()
    SolverFactory("gurobi").solve(model)
    elapsed = time.time() - start

    x_vals = [model.x[i]() for i in model.I]
    obj_val = model.obj()
    return x_vals, obj_val, elapsed

#----QP Problem----
def solve_relaxed_qp(Q, c):
    model = ConcreteModel()
    model.I = RangeSet(1, 6)
    model.x = Var(model.I, domain=NonNegativeReals, bounds=(0, 250))
    model.total = Constraint(expr=sum(model.x[i] for i in model.I) <= 750)

    def obj_rule(m):
        return sum(Q[i-1, j-1] * m.x[i] * m.x[j] for i in m.I for j in m.I) + sum(c[i-1] * m.x[i] for i in m.I)
    model.obj = Objective(rule=obj_rule, sense=maximize)

    start = time.time()
    SolverFactory("gurobi").solve(model)
    elapsed = time.time() - start

    x_vals = [model.x[i]() for i in model.I]
    obj_val = model.obj()
    return x_vals, obj_val, elapsed

#Print QIP results then QP results
for (subrace, attack), data in qp_problems.items():
    Q, c = data["Q"], data["c"]

    x_relaxed, obj_relaxed, t_relaxed = solve_relaxed_qp(Q, c)
    x_integer, obj_integer, t_integer = solve_integer_qip(Q, c)

    print(f"{'Subrace':<10} {'Attack':<8} " + " ".join(f"{label:<6}" for label in stat_labels) + "  Damage  Time")

    #QIP 
    print(f"{'':10} {'':8} " + 
        " ".join(f"{int(x):<6}" for x in x_integer) + 
        f"  {obj_integer:<9.4f}  {t_integer:.3f}s")

    #QP
    print(f"{subrace:<10} {attack:<8} " + 
        " ".join(f"{x:<6.1f}" for x in x_relaxed) + 
        f"  {obj_relaxed:<9.4f}  {t_relaxed:.3f}s")

    optimal_stats_qp[(subrace, attack)] = x_relaxed

#---- Applying Ability Multipliers----
def evaluate_damage(ability_mult, obj_func, toa_mult, x):
    return ability_mult * obj_func(x) * toa_mult

#Dictionary for each subrace/attack type and the formula: Ability Multipliers, objective function, Attack type
damage_cases_qp = {
    ("Vampire", "Melee"): (
        0.765625,
        lambda x: (1.06 + (0.03 / 250) * x[0] + (0.03 / 250) * (x[4]+50)) * (x[0] / 4 + 3 * x[5] / 80),
        1
    ),
    ("Vampire", "Ranged"): (
        0.765625,
        lambda x: (1.06 + (0.03 / 250) * x[1] + (0.03 / 250) * (x[4]+50)) * (x[0] / 5 + x[1] / 20 + 3 * x[5] / 80),
        1
    ),
    ("Vampire", "Magic"): (
        0.65625,
        lambda x: (1.16 + (0.12 / 250) * (x[2] + 50) + (0.04 / 250) * (x[4] + 50)) * ((x[2] + 50) / 4 + 3 * x[5] / 80),
        0.75
    ),
    ("Vampire", "Spell"): (
        1.25,
        lambda x: (1.16 + (0.12 / 250) * (x[2] + 50) + (0.04 / 250) * (x[4] + 50)) * ((x[2] + 50) / 4 + 3 * x[5] / 80),
        2
    ),

    ("Werewolf", "Melee"): (
        0.765625,
        lambda x: (1.06 + (0.03 / 250) * (x[0] + 50) + (0.03 / 250) * x[4]) * ((x[0] + 50) / 4 + 3 * x[5] / 80),
        1
    ),
    ("Werewolf", "Ranged"): (
        0.765625,
        lambda x: (1.06 + (0.03 / 250) * (x[1] + 50) + (0.03 / 250) * x[4]) * ((x[0] + 50) / 5 + (x[1] + 50) / 20 + 3 * x[5] / 80),
        1
    ),
    ("Werewolf", "Magic"): (
        0.65625,
        lambda x: (1.16 + (0.12 / 250) * x[2] + (0.04 / 250) * x[4]) * (x[2] / 4 + 3 * x[5] / 80),
        0.75
    ),
    ("Werewolf", "Spell"): (
        1.375,
        lambda x: (1.16 + (0.12 / 250) * x[2] + (0.04 / 250) * x[4]) * (x[2] / 4 + 3 * x[5] / 80),
        2
    ),

    ("Werepyre", "Melee"): (
        1,
        lambda x: (1.06 + (0.03 / 250) * (x[0] + 35) + (0.03 / 250) * x[4]) * ((x[0] + 35) / 16 + (x[2] + 35) / 16 + 3 * x[5] / 80),
        1
    ),
    ("Werepyre", "Ranged"): (
        1,
        lambda x: (1.06 + (0.03 / 250) * x[1] + (0.03 / 250) * x[4]) * (x[1] / 8 + 3 * x[5] / 80),
        1
    ),
    ("Werepyre", "Magic"): (
        1,
        lambda x: (1.16 + (0.12 / 250) * (x[2] + 35) + (0.04 / 250) * x[4]) * (3 * (x[0] + 35) / 40 + 3 * (x[2] + 35) / 40 + 5 * x[5] / 80),
        0.9975
    ),
    ("Werepyre", "Spell"): (
        1.5,
        lambda x: (1.16 + (0.12 / 250) * (x[2] + 35) + (0.04 / 250) * x[4]) * (3 * (x[0] + 35) / 40 + 3 * (x[2] + 35) / 40 + 5 * x[5] / 80),
        2
    ),
}

print("")
print("Total Damage from QP Solution after Abilities")
for (race, atype), (ab_mult, formula, toa_mult) in damage_cases_qp.items():
    x = optimal_stats_qp[(race, atype)]
    dmg = evaluate_damage(ab_mult, formula, toa_mult, x)
    print(f"{race:<10} {atype:<8} Damage: {dmg:.4f}")
