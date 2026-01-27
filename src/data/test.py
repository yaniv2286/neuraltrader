from data.preprocess import build_model_input

df = build_model_input("AAPL", timeframes=["1d", "1h"], start="2023-01-01", end="2023-12-31")
print(df.head())
