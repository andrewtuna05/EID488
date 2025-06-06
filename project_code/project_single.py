from pyomo.environ import *
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

