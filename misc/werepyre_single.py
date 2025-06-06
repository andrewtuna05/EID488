import sys
from pyomo.environ import *

""""
Werepyre: Choose an Attack type:
1 = Melee
2 = Ranged
3 = Magic
4 = Spell
"""
choice = int(sys.argv[1])
if choice not in [1, 2, 3, 4]:
    raise ValueError("Usage: python3 werepyre_single.py [1-4]")

model = ConcreteModel()
model.I = RangeSet(1, 6)
model.y = Var(model.I, domain=NonNegativeIntegers, bounds=(0, 50))

def x_rule(model, i):
    return 5 * model.y[i]
model.x = Expression(model.I, rule=x_rule)

#Use corresponding formula based on input 
def obj_expression(model):
    if choice == 1:
        return 2 * ((model.x[1] + 35 + model.x[3] + 35) / 16 + 3 * model.x[6] / 80)
    elif choice == 2:
        return 2 * ((model.x[2])/ 8 + 3 * model.x[6] / 80)
    elif choice == 3:
        return (8/3) * ((model.x[1] + 35 + model.x[3] + 35) / 16 + 3 * model.x[6] / 80)
    elif choice == 4:
        return 3 * ((model.x[1] + 35 + model.x[3] + 35) / 16 + 3 * model.x[6] / 80)

model.obj = Objective(rule=obj_expression, sense=maximize)

def total_points_rule(model):
    return sum(model.x[i] for i in model.I) <= 750
model.total_points = Constraint(rule=total_points_rule)

SolverFactory('glpk').solve(model)

for i in model.I:
    print(f"x[{i}] = {model.x[i]():.0f}")
print(f"Objective value (Damage) = {model.obj():.5f}")
