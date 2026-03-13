Brief on the model: 

Oranges Pricing Optimizer — I built a lightweight, end‑to‑end pricing system that learns a product’s demand curve and recommends 
the price that maximizes expected daily profit. I started with oranges because I grew up around a large orange orchard, which made 
pricing problems feel real and worth solving. I defined the problem myself and justified the solution from first principles: identify 
the signals that move demand (price, promotions, competitor price, seasonality, weather, recent sales), train a Gradient Boosting 
model in a reproducible scikit‑learn pipeline, and validate it with a time‑aware split to mimic how pricing would work in practice.

To make the work usable, I deployed the trained model behind a FastAPI service and added a tiny HTML interface so anyone can input market 
conditions and instantly see the recommended price and expected profit. Although the first demo uses orange sales, the approach is product‑agnostic—it
can be retrained for any produce as long as historical sales and pricing data are available. I used Copilot as an assistant for iteration and debugging, 
but the problem framing, design choices, and trade‑offs were mine. Note: the dataset in this demo is synthetic and Copilot‑generated, used purely for 
learning and prototyping.
