from pyomo.environ import *
import numpy as np
import time

#---- Single Turn Damage Optimization Code----

#Define the stat weight vectors: [STR, DEX, INT, END, CHA, LUK]
damage_map = {
    'vampire': {
        1: [1/4, 0, 0, 0, 0, 3/80],  #melee
        2: [1/5, 1/10, 0, 0, 0, 3/80],  #ranged
        3: [0, 0, 1/4, 0, 0, 3/80],  #magic
        4: [0, 0, 1/4, 0, 0, 3/80],  #spell
    },
    'werewolf': {
        1: [1/4, 0, 0, 0, 0, 3/80],
        2: [1/5, 1/20, 0, 0, 0, 3/80],
        3: [0, 0, 1/4, 0, 0, 3/80],
        4: [0, 0, 1/4, 0, 0, 3/80],
    },
    'werepyre': {
        1: [1/16, 0, 1/16, 0, 0, 3/80],  #hybrid melee
        2: [0, 1/8, 0, 0, 0, 3/80],
        3: [3/40, 0, 3/40, 0, 0, 5/80],  #hybrid magic  
        4: [3/40, 0, 3/40, 0, 0, 5/80], 
    }
}
#---- ILP Problem ----
stat_labels = ['STR', 'DEX', 'INT', 'END', 'CHA', 'LUK']
print("ILP Solutions")
print(f"{'Subrace':<10} {'Type':<8} " + " ".join(f"{s:<5}" for s in stat_labels) + " Damage")

attack_names = {1: "Melee", 2: "Ranged", 3: "Magic", 4: "Spell"}
optimal_stats = {}

for race in damage_map:
    for atype in range(1, 5):
        w = damage_map[race][atype]

        model = ConcreteModel()
        model.I = RangeSet(1, 6)
        model.y = Var(model.I, domain=NonNegativeIntegers, bounds=(0, 50))
        model.x = Expression(model.I, rule=lambda m, i: 5 * m.y[i])
        model.obj = Objective(expr=sum(w[i-1] * model.x[i] for i in model.I), sense=maximize)
        model.total = Constraint(expr=sum(model.x[i] for i in model.I) <= 750)

        start = time.time()
        SolverFactory('glpk').solve(model, tee=False)
        end = time.time()
        elapsed = end - start

        x_vals = [int(model.x[i]()) for i in model.I]
        damage = model.obj()

        print(f"{race:<10} {atype:<6} " + " ".join(f"{x:<6}" for x in x_vals) + f"{damage:.4f}  Time: {elapsed:.3f}s")

        #Storing the optimal stat distributions
        optimal_stats[(race.capitalize(), attack_names[atype])] = [model.x[i]() for i in model.I]

#---- LP Relaxation ----
print()
print("LP Relaxation Solutions")
print(f"{'Subrace':<10} {'Type':<6} " + " ".join(f"{s:<6}" for s in stat_labels) + "Damage")

for race in damage_map:
    for atype in range(1, 5):
        w = damage_map[race][atype]

        relaxed_model = ConcreteModel()
        relaxed_model.I = RangeSet(1, 6)
        relaxed_model.x = Var(relaxed_model.I, domain=NonNegativeReals, bounds=(0, 250))
        relaxed_model.obj = Objective(expr=sum(w[i-1] * relaxed_model.x[i] for i in relaxed_model.I), sense=maximize)
        relaxed_model.total = Constraint(expr=sum(relaxed_model.x[i] for i in relaxed_model.I) <= 750)
        
        start = time.time()
        SolverFactory('glpk').solve(relaxed_model)
        end = time.time()
        elapsed = end - start

        x_vals = [relaxed_model.x[i]() for i in relaxed_model.I]
        damage = relaxed_model.obj()
        print(f"{race:<10} {atype:<6} " + " ".join(f"{x:<6.2f}" for x in x_vals) + f"  {damage:.4f}  Time: {elapsed:.3f}s")

#----Applying Ability Multipliers----

x = [model.x[i]() for i in range(1, 7)]  #Stats x1 ... x6

def evaluate_damage(ability_mult, obj_function, toa_mult):
    return ability_mult * obj_function(x) * toa_mult

#Dictionary for each subrace/attack type and the formula: Ability Multipliers, objective function, Attack type
damage_cases = {
    ("Vampire", "Melee"):      (0.765625, lambda x: x[0]/4 + 3*x[5]/80, 1),
    ("Vampire", "Ranged"):     (0.765625, lambda x: x[0]/5 + x[1]/20 + 3*x[5]/80, 1),
    ("Vampire", "Magic"):      (0.65625, lambda x: (x[2] + 50)/4 + 3*x[5]/80, 0.75),
    ("Vampire", "Spell"):      (1.25, lambda x: (x[2] + 50)/4 + 3*x[5]/80, 2),

    ("Werewolf", "Melee"):     (0.765625, lambda x: (x[0] + 50)/4 + 3*x[5]/80, 1),
    ("Werewolf", "Ranged"):    (0.765625, lambda x: (x[0] + 50)/5 + (x[1] + 50)/20 + 3*x[5]/80, 1),
    ("Werewolf", "Magic"):     (0.65625, lambda x: x[2]/4 + 3*x[5]/80, 0.75),
    ("Werewolf", "Spell"):     (1.375, lambda x: x[2]/4 + 3*x[5]/80, 2),

    ("Werepyre", "Melee"):     (1, lambda x: (x[0] + 35)/16 + (x[2] + 35)/16 + 3*x[5]/80, 1),
    ("Werepyre", "Ranged"):    (1, lambda x: x[1]/8 + 3*x[5]/80, 1),
    ("Werepyre", "Magic"):     (1, lambda x: 3*(x[0] + 35)/40 + 3*(x[2] + 35)/40 + 5*x[5]/80, 0.9975),
    ("Werepyre", "Spell"):     (1.5, lambda x: 3*(x[0] + 35)/40 + 3*(x[2] + 35)/40 + 5*x[5]/80, 2),
}

for (race, atype), (ab_mult, formula, toa_mult) in damage_cases.items():
    x = optimal_stats[(race, atype)]  
    dmg = evaluate_damage(ab_mult, formula, toa_mult)
    print(f"{race} {atype} Damage: {dmg:.4f}")

#---- Single Turn Damage Optimization Dual Code----

damage_map = {
    'vampire': {
        1: [1/4, 0, 0, 0, 0, 3/80],     #melee
        2: [1/5, 1/10, 0, 0, 0, 3/80],  #ranged
        3: [0, 0, 1/4, 0, 0, 3/80],     #magic
        4: [0, 0, 1/4, 0, 0, 3/80],     #spell
    },
    'werewolf': {
        1: [1/4, 0, 0, 0, 0, 3/80],
        2: [1/5, 1/20, 0, 0, 0, 3/80],
        3: [0, 0, 1/4, 0, 0, 3/80],
        4: [0, 0, 1/4, 0, 0, 3/80],
    },
    'werepyre': {
        1: [1/16, 0, 1/16, 0, 0, 3/80],  #hybrid melee
        2: [0, 1/8, 0, 0, 0, 3/80],
        3: [3/40, 0, 3/40, 0, 0, 5/80],  #hybrid magic
        4: [3/40, 0, 3/40, 0, 0, 5/80], 
    }
}

print(f"{'Race':<10} {'Type':<6} {'Objective':<10} {'Time (s)'}")
print("-" * 40)

for race in damage_map:
    for atype in range(1, 5):
        c = damage_map[race][atype]

        model = ConcreteModel()
        model.I = RangeSet(1, 6)
        model.J = RangeSet(1, 13)  # 1 total, 6 lower, 6 upper

        model.lmbda = Var(model.J, domain=NonNegativeReals)

        b = [750] + [0]*6 + [250]*6

        model.obj = Objective(expr=sum(b[j-1] * model.lmbda[j] for j in model.J), sense=minimize)

        def dual_con_rule(model, i):
            return (
                model.lmbda[1]                # total constraint
                - model.lmbda[1 + i]          # x_i ≥ 0
                + model.lmbda[7 + i]          # x_i ≤ 250
                >= c[i - 1]
            )
        model.dual_constraints = Constraint(model.I, rule=dual_con_rule)

        start = time.time()
        result = SolverFactory('glpk').solve(model)
        end = time.time()

        print(f"{race:<10} {atype:<6} {value(model.obj):<10.4f} {end - start:.3f}")
        for j in model.J:
            print(f"  lambda[{j}] = {value(model.lmbda[j]):.6f}")

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
