from pyomo.environ import *
import time

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
