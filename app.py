from flask import Flask, render_template, request, jsonify
from utils.stock_data import get_stock_data, get_historical_data, get_candlestick_data
from utils.model import predict_next_price
from utils.forecast_monitor import (
    seed_initial_history_if_empty,
    record_forecast,
    match_observed_prices,
    calculate_rolling_accuracy,
    get_deterioration_alerts,
    get_forecast_history_records
)

app = Flask(__name__)
seed_initial_history_if_empty()

# Default watchlist / popular tickers
SYMBOLS = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Google",
    "TSLA": "Tesla",
    "AMZN": "Amazon",
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
}

@app.route("/")
def home():
    indices = {}
    for sym, name in [("^GSPC", "S&P 500"), ("^DJI", "Dow Jones"), ("^IXIC", "Nasdaq"), ("BTC-USD", "Bitcoin")]:
        try:
            data = get_stock_data(sym)
            indices[name] = {
                "price": data["price"],
                "change": data["price_change"],
                "pct": data["pct_change"],
                "symbol": sym
            }
        except Exception:
            indices[name] = {
                "price": 0.0,
                "change": 0.0,
                "pct": 0.0,
                "symbol": sym
            }
    return render_template("home.html", indices=indices, symbols=SYMBOLS)

@app.route("/dashboard")
def dashboard():
    symbol = request.args.get("symbol", "AAPL").upper().strip()
    display_symbols = SYMBOLS.copy()
    error_msg = None

    try:
        stock = get_stock_data(symbol)
        if symbol not in display_symbols:
            display_symbols[symbol] = stock["name"]
    except Exception as e:
        error_msg = f"Error loading '{symbol}': {str(e)}"
        symbol = "AAPL"
        stock = get_stock_data(symbol)

    return render_template(
        "dashboard.html",
        stock=stock,
        selected_symbol=symbol,
        symbols=display_symbols,
        error=error_msg
    )

@app.route("/prediction")
def prediction():
    symbol = request.args.get("symbol", "AAPL").upper().strip()
    selected_model = request.args.get("model", "linear_regression")
    display_symbols = SYMBOLS.copy()
    error_msg = None

    try:
        stock = get_stock_data(symbol)
        model_results = predict_next_price(symbol)
        if symbol not in display_symbols:
            display_symbols[symbol] = stock["name"]
    except Exception as e:
        error_msg = f"Error processing predictions for '{symbol}': {str(e)}"
        symbol = "AAPL"
        stock = get_stock_data(symbol)
        model_results = predict_next_price(symbol)

    model_data = model_results.get(selected_model, model_results["linear_regression"])
    pred_price = model_data["predicted_price"]
    diff = round(pred_price - stock["price"], 2)
    pct_diff = round((diff / stock["price"]) * 100, 2)

    # Match any due pending forecasts and record new live published forecasts
    try:
        match_observed_prices(symbol)
        for m_name, m_res in model_results.items():
            record_forecast(
                symbol=symbol,
                model_name=m_name,
                predicted_price=m_res["predicted_price"],
                base_price=stock["price"],
                validation_metrics={"mae": m_res["mae"], "rmse": m_res["rmse"], "r2": m_res["r2"]}
            )
    except Exception as e:
        print(f"Monitor record warning: {e}")

    live_metrics = calculate_rolling_accuracy(symbol=symbol, model_name=selected_model)
    deterioration_alerts = get_deterioration_alerts(symbol=symbol)

    if pct_diff > 1.5:
        signal = "Strong Buy 🟢🟢"
        badge_class = "success"
    elif pct_diff > 0.2:
        signal = "Buy 🟢"
        badge_class = "success"
    elif pct_diff < -1.5:
        signal = "Strong Sell 🔴🔴"
        badge_class = "danger"
    elif pct_diff < -0.2:
        signal = "Sell 🔴"
        badge_class = "danger"
    else:
        signal = "Hold 🟡"
        badge_class = "warning"

    return render_template(
        "prediction.html",
        stock=stock,
        selected_symbol=symbol,
        symbols=display_symbols,
        model_results=model_results,
        model_data=model_data,
        selected_model=selected_model,
        predicted_price=pred_price,
        prediction_difference=diff,
        prediction_pct=pct_diff,
        signal=signal,
        badge_class=badge_class,
        live_metrics=live_metrics,
        deterioration_alerts=deterioration_alerts,
        error=error_msg
    )

@app.route("/forecast-monitor")
def forecast_monitor():
    seed_initial_history_if_empty()
    symbol = request.args.get("symbol", "AAPL").upper().strip()
    model_filter = request.args.get("model", "all")
    display_symbols = SYMBOLS.copy()

    try:
        match_observed_prices(symbol if symbol != "ALL" else None)
    except Exception:
        pass

    sym_filter = symbol if symbol != "ALL" else None
    lr_stats = calculate_rolling_accuracy(symbol=sym_filter, model_name="linear_regression")
    rf_stats = calculate_rolling_accuracy(symbol=sym_filter, model_name="random_forest")
    alerts = get_deterioration_alerts(symbol=sym_filter)
    history = get_forecast_history_records(
        symbol=sym_filter,
        model_name=model_filter if model_filter != "all" else None,
        limit=60
    )

    return render_template(
        "forecast_monitor.html",
        selected_symbol=symbol,
        model_filter=model_filter,
        symbols=display_symbols,
        lr_stats=lr_stats,
        rf_stats=rf_stats,
        alerts=alerts,
        history=history
    )

@app.route("/api/forecast-monitor/match", methods=["POST", "GET"])
def api_forecast_match():
    symbol = request.args.get("symbol")
    sym = None if not symbol or symbol == "ALL" else symbol
    resolved = match_observed_prices(sym)
    return jsonify({"status": "success", "resolved_count": resolved})

@app.route("/analytics")
def analytics():
    symbol = request.args.get("symbol", "AAPL").upper().strip()
    display_symbols = SYMBOLS.copy()
    error_msg = None

    try:
        dates, prices, indicators = get_historical_data(symbol, period="6mo")
        model_results = predict_next_price(symbol)
        stock_info = get_stock_data(symbol)
        if symbol not in display_symbols:
            display_symbols[symbol] = stock_info["name"]
    except Exception as e:
        error_msg = f"Error loading analytics for '{symbol}': {str(e)}"
        symbol = "AAPL"
        dates, prices, indicators = get_historical_data(symbol, period="6mo")
        model_results = predict_next_price(symbol)

    return render_template(
        "analytics.html",
        selected_symbol=symbol,
        symbols=display_symbols,
        dates=dates,
        prices=prices,
        indicators=indicators,
        model_results=model_results,
        error=error_msg
    )

@app.route("/about")
def about():
    return render_template("about.html")

# API validation endpoint for client-side search & watchlist validation
@app.route("/api/validate")
def validate_ticker():
    symbol = request.args.get("symbol", "").upper().strip()
    if not symbol:
        return jsonify({"valid": False, "error": "Please enter a symbol."})
    try:
        data = get_stock_data(symbol)
        return jsonify({
            "valid": True,
            "symbol": data["symbol"],
            "name": data["name"],
            "price": data["price"]
        })
    except Exception as e:
        return jsonify({"valid": False, "error": str(e)})

# API endpoint for historical candlestick and volume data
@app.route("/api/history/<symbol>")
def api_history(symbol):
    period = request.args.get("period", "6mo")
    try:
        data = get_candlestick_data(symbol, period=period)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# API endpoint for live price of a single ticker
@app.route("/api/live-price/<symbol>")
def api_live_price(symbol):
    try:
        data = get_stock_data(symbol)
        return jsonify({
            "symbol": data["symbol"],
            "price": data["price"],
            "high": data["high"],
            "low": data["low"],
            "volume": data["volume"],
            "price_change": data["price_change"],
            "pct_change": data["pct_change"],
            "news": data["news"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# API endpoint for batch updates of multiple tickers (e.g. for the watchlist sidebar)
@app.route("/api/batch-prices")
def api_batch_prices():
    symbols_param = request.args.get("symbols", "")
    if not symbols_param:
        return jsonify([])
        
    symbols = [s.strip().upper() for s in symbols_param.split(",") if s.strip()]
    results = []
    
    for sym in symbols:
        try:
            data = get_stock_data(sym)
            results.append({
                "symbol": data["symbol"],
                "name": data["name"],
                "price": data["price"],
                "change": data["price_change"],
                "pct": data["pct_change"]
            })
        except Exception:
            # Return placeholder if fetch fails to avoid breaking the batch response
            results.append({
                "symbol": sym,
                "name": sym,
                "price": 0.0,
                "change": 0.0,
                "pct": 0.0
            })
            
    return jsonify(results)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
