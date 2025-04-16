from pyomo.environ import *

model = ConcreteModel()

# Index set for stat variables
model.I = RangeSet(1, 6)

# Integer variables for multiples of 5: x_i = 5 * y_i
model.y = Var(model.I, domain=NonNegativeIntegers, bounds=(0, 50))  # 0 to 250 → 0 to 50 in steps of 5

# Reconstruct x_i = 5 * y_i
def x_rule(model, i):
    return 5 * model.y[i]

model.x = Expression(model.I, rule=x_rule)

# Objective function
def obj_expression(model):
    return 1.03125 * (3 * (model.x[3] + 50) / 32 + 3 * model.x[6] / 80)

model.obj = Objective(rule=obj_expression, sense=maximize)

# Constraint: total allocated stat points ≤ 750
def total_points_rule(model):
    return sum(model.x[i] for i in model.I) <= 750

model.total_points = Constraint(rule=total_points_rule)

# Solve
solver = SolverFactory('glpk')
solver.solve(model)

# Output results
for i in model.I:
    print(f"x[{i}] = {model.x[i]():.0f} (y[{i}] = {model.y[i]()} → x = 5*y)")

print("Objective value (Damage) =", model.obj())
