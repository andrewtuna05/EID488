from pyomo.environ import *

# Define the stat weight vectors: [STR, DEX, INT, END, CHA, LUK]
damage_map = {
    'vampire': {
        1: [1/8, 0, 0, 0, 0, 3/80],  # melee
        2: [0, 1/8, 0, 0, 0, 3/80],  # ranged
        3: [0, 0, 3/32, 0, 0, 3/80],  # magic
        4: [0, 0, 1/4, 0, 0, 3/80],  # spell
    },
    'werewolf': {
        1: [1/8, 0, 0, 0, 0, 3/80],
        2: [0, 1/8, 0, 0, 0, 3/80],
        3: [0, 0, 3/32, 0, 0, 3/80],
        4: [0, 0, 1/4, 0, 0, 3/80],
    },
    'werepyre': {
        1: [1/16, 0, 1/16, 0, 0, 3/80],  # hybrid melee
        2: [0, 1/8, 0, 0, 0, 3/80],    # ranged
        3: [3/40, 0, 3/40, 0, 0, 5/80],  # hybrid magic
        4: [0, 0, 1/4, 0, 0, 3/80],  # hybrid spell
    }
}

stat_labels = ['STR', 'DEX', 'INT', 'END', 'CHA', 'LUK']
print("ILP Solutions")
print(f"{'Subrace':<10} {'Type':<8} " + " ".join(f"{s:<5}" for s in stat_labels) + " Damage")

for race in damage_map:
    for atype in range(1, 5):
        w = damage_map[race][atype]

        model = ConcreteModel()
        model.I = RangeSet(1, 6)
        model.y = Var(model.I, domain=NonNegativeIntegers, bounds=(0, 50))
        model.x = Expression(model.I, rule=lambda m, i: 5 * m.y[i])
        model.obj = Objective(expr=sum(w[i-1] * model.x[i] for i in model.I), sense=maximize)
        model.total = Constraint(expr=sum(model.x[i] for i in model.I) <= 750)

        SolverFactory('glpk').solve(model, tee=False)

        x_vals = [int(model.x[i]()) for i in model.I]
        damage = model.obj()

        print(f"{race:<10} {atype:<6} " + " ".join(f"{x:<6}" for x in x_vals) + f"{damage:.4f}")
print()
print("LP Relaxation Solutions")
print(f"{'Subrace':<10} {'Type':<6} " + " ".join(f"{s:<6}" for s in stat_labels) + "Damage")

for race in damage_map:
    for atype in range(1, 5):
        w = damage_map[race][atype]

        relaxed_model = ConcreteModel()
        relaxed_model.I = RangeSet(1, 6)
        relaxed_model.y = Var(relaxed_model.I, domain=NonNegativeReals, bounds=(0, 50))
        relaxed_model.x = Expression(relaxed_model.I, rule=lambda m, i: 5 * m.y[i])
        relaxed_model.obj = Objective(expr=sum(w[i-1] * relaxed_model.x[i] for i in relaxed_model.I), sense=maximize)
        relaxed_model.total = Constraint(expr=sum(relaxed_model.x[i] for i in relaxed_model.I) <= 750)
        SolverFactory('glpk').solve(relaxed_model)

        x_vals = [relaxed_model.x[i]() for i in relaxed_model.I]
        damage = relaxed_model.obj()
        print(f"{race:<10} {atype:<6} " + " ".join(f"{x:<6.2f}" for x in x_vals) + f" {damage:.4f}")
