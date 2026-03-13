
# app.py — tiny API for your pricing model
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# 1) Load your trained pipeline (preprocessing + model)
pipe = joblib.load("demand_model_oranges.joblib")

# 2) Define the request shapes (just the fields the model needs)
class Context(BaseModel):
    promo: int
    comp_price: float
    temp_c: float
    dow: int      # 0=Mon ... 6=Sun
    month: int    # 1..12
    lag_1: float
    lag_7: float
    roll7_mean: float

class PredictRequest(Context):
    price: float

class OptimizeRequest(Context):
    unit_cost: float
    pmin: float = 0.85
    pmax: float = 1.30
    step: float = 0.01

app = FastAPI(title="Profit Optimizer (mini)")

# 3) Helper: demand at a given price (recompute derived features)
def predict_demand_at_price(ctx: dict, price: float) -> float:
    ctx = ctx.copy()
    comp = float(ctx["comp_price"])
    ctx["price"] = float(price)
    ctx["price_gap"] = float(price) - comp
    ctx["price_ratio"] = (float(price) / comp) if comp else 1.0
    row = pd.DataFrame([ctx])
    q = float(pipe.predict(row)[0])
    return max(0.0, q)

# 4) Endpoint A: predict units at a chosen price
@app.post("/predict_demand")
def predict_demand(req: PredictRequest):
    qhat = predict_demand_at_price(req.dict(), req.price)
    return {"predicted_units": qhat}

# 5) Endpoint B: get the best price for profit
@app.post("/optimize_price")
def optimize_price(req: OptimizeRequest):
    ctx = req.dict()
    unit_cost = float(ctx.pop("unit_cost"))
    pmin = float(ctx.pop("pmin"))
    pmax = float(ctx.pop("pmax"))
    step = float(ctx.pop("step"))

    grid = np.round(np.arange(pmin, pmax + 1e-6, step), 2)
    grid = grid[grid <= pmax]  # strict cap

    best_p, best_pi = None, -1e18
    for p in grid:
        q = predict_demand_at_price(ctx, p)
        profit = (p - unit_cost) * q
        if profit > best_pi:
            best_p, best_pi = float(p), float(profit)

    return {"p_star": best_p, "profit_star": best_pi}