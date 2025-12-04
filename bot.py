import time
import json
import re
import math
import os
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional

import requests
from flask import Flask, render_template_string

# =========================
# 1. CONFIG
# =========================

TELEGRAM_BOT_TOKEN = os.getenv("BOT_TOKEN")
if not TELEGRAM_BOT_TOKEN:
    print("ERROR: TELEGRAM_BOT_TOKEN is not set. Check Render environment variables.")

GAMMA_BASE = "https://gamma-api.polymarket.com"
DATA_BASE = "https://data-api.polymarket.com"

POLL_INTERVAL_SEC = 2
RECENT_TRADES_LIMIT = 300

# Hypothetical stake used in profit examples
INVEST_AMOUNT_USDC = 50.0

# When AI vs market difference is larger than this, we call it an “edge”
EDGE_THRESHOLD = 0.07  # 7 percentage points

# Whale definition (in USDC notional: size * price)
MIN_WHALE_USDC = 1000.0  # you can change this

# Markets to include in /arb scan
ARBITRAGE_WATCHLIST: List[str] = [
    "will-polymarket-us-go-live-in-2025",
]

# For event support: remember which event's markets a user can pick from
PENDING_EVENT_SELECTION: Dict[int, List[Dict[str, Any]]] = {}

# Auto-alert watchlist:
# WATCHLIST[chat_id][slug] = {threshold, last_edge, last_alert_ts}
WATCHLIST: Dict[int, Dict[str, Dict[str, Any]]] = {}

# How often to check watched markets (in seconds)
ALERT_CHECK_INTERVAL_SEC = 900  # 15 minutes
MIN_ALERT_GAP_SEC = 1800        # min 30 min between alerts per market

# For pretty formatting and dashboard
SECTION_DIVIDER = "────────────────────────"
LAST_ANALYSES: List[Dict[str, Any]] = []  # store recent analyses for web UI

# =========================
# 2. TELEGRAM HELPERS
# =========================

def telegram_api(method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
    if params is None:
        params = {}
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/{method}"
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def send_message(chat_id: int, text: str) -> None:
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True,
    }
    try:
        resp = requests.post(url, data=payload, timeout=15)
        if not resp.ok:
            print("[ERROR] Telegram send failed:", resp.text)
    except Exception as e:
        print("[ERROR] Telegram exception:", e)


# =========================
# 3. POLYMARKET HELPERS
# =========================

def fetch_market_by_slug(slug: str) -> Dict[str, Any]:
    url = f"{GAMMA_BASE}/markets/slug/{slug}"
    resp = requests.get(url, timeout=10)
    if resp.status_code == 404:
        raise ValueError("MARKET_NOT_FOUND")
    resp.raise_for_status()
    return resp.json()


def fetch_event_by_slug(slug: str) -> Dict[str, Any]:
    url = f"{GAMMA_BASE}/events/slug/{slug}"
    resp = requests.get(url, timeout=10)
    if resp.status_code == 404:
        raise ValueError("EVENT_NOT_FOUND")
    resp.raise_for_status()
    return resp.json()


def fetch_recent_trades(condition_id: str, limit: int = RECENT_TRADES_LIMIT) -> List[Dict[str, Any]]:
    params = {"limit": limit, "market": condition_id}
    resp = requests.get(f"{DATA_BASE}/trades", params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def parse_outcome_prices(market: Dict[str, Any]) -> List[float]:
    raw = market.get("outcomePrices")
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return []
    try:
        return [float(x) for x in raw]
    except Exception:
        return []


def parse_outcomes(market: Dict[str, Any]) -> List[str]:
    outcomes = market.get("outcomes") or []
    if isinstance(outcomes, str):
        try:
            outcomes = json.loads(outcomes)
        except json.JSONDecodeError:
            outcomes = []
    return [str(o) for o in outcomes]


# =========================
# 4. FEATURES, AI MODEL, WHALES
# =========================

def compute_outcome_stats(trades: List[Dict[str, Any]], n_outcomes: int) -> List[Dict[str, float]]:
    stats = [
        {"buy_vol": 0.0, "sell_vol": 0.0, "total_vol": 0.0,
         "trade_count": 0, "avg_trade_price": 0.0}
        for _ in range(n_outcomes)
    ]

    for t in trades:
        idx = t.get("outcomeIndex")
        if idx is None:
            continue
        try:
            idx = int(idx)
        except Exception:
            continue
        if not (0 <= idx < n_outcomes):
            continue

        try:
            size = float(t.get("size", 0) or 0.0)
            price = float(t.get("price", 0) or 0.0)
        except Exception:
            size, price = 0.0, 0.0

        side = (t.get("side") or "").upper()

        if side == "BUY":
            stats[idx]["buy_vol"] += size
        elif side == "SELL":
            stats[idx]["sell_vol"] += size

        stats[idx]["total_vol"] += size

        c = stats[idx]["trade_count"]
        if c == 0:
            stats[idx]["avg_trade_price"] = price
        else:
            stats[idx]["avg_trade_price"] = (
                stats[idx]["avg_trade_price"] * c + price
            ) / (c + 1)

        stats[idx]["trade_count"] += 1

    return stats


def sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def ai_model_probs(prices: List[float], stats: List[Dict[str, float]]) -> List[float]:
    """
    Conservative regression-style AI:
      features:
        - market_prob
        - sentiment  = (buy - sell) / total
        - log_volume = log10(1 + total_vol)
        - momentum   = price - avg_trade_price
    """
    n = len(prices)
    if n == 0:
        return []

    w0 = -0.5
    w1 = 4.0   # market price
    w2 = 1.5   # sentiment
    w3 = 0.7   # volume
    w4 = 1.0   # momentum

    raw_scores: List[float] = []

    for i in range(n):
        p = max(0.001, min(0.999, float(prices[i])))

        total_vol = stats[i]["total_vol"]
        buy_vol = stats[i]["buy_vol"]
        sell_vol = stats[i]["sell_vol"]
        avg_trade_price = stats[i]["avg_trade_price"]

        if total_vol > 0:
            sentiment = (buy_vol - sell_vol) / total_vol  # [-1,1]
        else:
            sentiment = 0.0

        log_vol = math.log10(1.0 + total_vol)
        momentum = (p - avg_trade_price) if avg_trade_price > 0 else 0.0

        logit = w0 + w1 * p + w2 * sentiment + w3 * log_vol + w4 * momentum
        raw = sigmoid(logit)
        raw_scores.append(raw)

    total_raw = sum(raw_scores)
    if total_raw <= 0:
        return prices[:]

    return [r / total_raw for r in raw_scores]


def detect_whales(trades: List[Dict[str, Any]], min_usdc: float) -> List[Dict[str, Any]]:
    whales: List[Dict[str, Any]] = []
    for t in trades:
        try:
            size = float(t.get("size", 0) or 0.0)
            price = float(t.get("price", 0) or 0.0)
            notional = size * price
        except Exception:
            continue

        if notional >= min_usdc:
            t2 = dict(t)
            t2["notional"] = notional
            whales.append(t2)

    whales.sort(key=lambda x: x.get("notional", 0.0), reverse=True)
    return whales[:5]


def aggregate_whale_flow(whales: List[Dict[str, Any]], n_outcomes: int) -> List[Dict[str, float]]:
    flow = [
        {"buy_notional": 0.0, "sell_notional": 0.0}
        for _ in range(n_outcomes)
    ]
    for w in whales:
        idx = w.get("outcomeIndex")
        try:
            idx = int(idx)
        except Exception:
            continue
        if not (0 <= idx < n_outcomes):
            continue

        side = (w.get("side") or "").upper()
        notional = float(w.get("notional", 0.0) or 0.0)
        if side == "BUY":
            flow[idx]["buy_notional"] += notional
        elif side == "SELL":
            flow[idx]["sell_notional"] += notional
    return flow


# =========================
# 5. TEXT & SIGNAL HELPERS
# =========================

def extract_polymarket_slug(text: str) -> Optional[str]:
    """
    Extract slug even if URL has ?tid=, ?ref=, #fragment, etc.
    Works for /event/... and /market/...
    """
    m = re.search(r"https?://polymarket\.com/[^\s]+", text)
    if not m:
        return None

    url = m.group(0).rstrip("/")

    if "?" in url:
        url = url.split("?", 1)[0]
    if "#" in url:
        url = url.split("#", 1)[0]

    slug = url.split("/")[-1]
    return slug or None


def recommendation_text(market_prob: float, ai_prob: float) -> str:
    diff = ai_prob - market_prob

    if abs(diff) < EDGE_THRESHOLD:
        return "No strong edge. Probably *hold / avoid*."

    if diff > 0:
        return "AI sees this as *undervalued*. Consider **BUY** (if you agree)."
    else:
        return "AI sees this as *overpriced*. Consider **SELL/short or opposite side** (if possible)."


def trading_signal(market_prob: float, ai_prob: float) -> str:
    edge = ai_prob - market_prob
    abs_edge = abs(edge)

    if abs_edge < EDGE_THRESHOLD / 2:
        return "😐 Signal: HOLD / AVOID (tiny edge)"

    if edge > 0:
        if abs_edge > EDGE_THRESHOLD * 2:
            return "🔥 Signal: STRONG BUY (AI thinks heavily undervalued)"
        else:
            return "✅ Signal: BUY (AI sees some undervaluation)"
    else:
        if abs_edge > EDGE_THRESHOLD * 2:
            return "⚠️ Signal: STRONG SELL / TAKE OPPOSITE SIDE (if possible)"
        else:
            return "⚠️ Signal: SELL / TRIM (AI sees mild overpricing)"


def profit_scenario_text(price: float) -> str:
    p = max(0.01, min(0.99, price))
    stake = INVEST_AMOUNT_USDC
    payoff_if_win = stake / p
    profit_if_win = payoff_if_win - stake

    return (
        f"If you invest ~{stake:.2f} USDC at price {p:.3f}:\n"
        f"- If it *wins*: you get ~{payoff_if_win:.2f} (profit ≈ {profit_if_win:.2f})\n"
        f"- If it *loses*: you lose your stake ({stake:.2f})."
    )


def build_ai_summary(outcomes: List[str], prices: List[float], ai_probs: List[float]) -> str:
    if not ai_probs or len(ai_probs) != len(prices):
        return "I couldn't build an AI summary for this market."

    best_idx = max(range(len(ai_probs)), key=lambda i: ai_probs[i])
    best_ai = ai_probs[best_idx]
    best_mkt = prices[best_idx]
    best_name = outcomes[best_idx] if best_idx < len(outcomes) else f"Outcome {best_idx}"

    sorted_probs = sorted(ai_probs, reverse=True)
    second_ai = sorted_probs[1] if len(sorted_probs) > 1 else 0.0
    margin = best_ai - second_ai

    if margin > 0.15:
        confidence = "high"
    elif margin > 0.07:
        confidence = "medium"
    else:
        confidence = "low"

    return (
        f"*AI overall prediction:*\n"
        f"Most likely outcome: *{best_name}*\n"
        f"- AI prob: ~{best_ai*100:.2f}% (confidence: {confidence})\n"
        f"- Market prob: ~{best_mkt*100:.2f}%\n"
        f"- Difference (edge): {(best_ai - best_mkt)*100:+.2f}%"
    )


# =========================
# 6. CORE ANALYSIS (MARKET)
# =========================

def analyse_market_object(chat_id: int, market: Dict[str, Any]) -> None:
    question = market.get("question") or market.get("title") or "Unknown question"
    outcomes = parse_outcomes(market)
    prices = parse_outcome_prices(market)
    if not prices:
        send_message(chat_id, "I couldn’t read prices for that market.")
        return

    condition_id = market.get("conditionId")
    trades: List[Dict[str, Any]] = []
    stats: List[Dict[str, float]] = []

    if condition_id:
        try:
            trades = fetch_recent_trades(condition_id, RECENT_TRADES_LIMIT)
            stats = compute_outcome_stats(trades, len(prices))
        except Exception as e:
            print("[WARN] trades fetch failed:", e)
            stats = [{"buy_vol": 0.0, "sell_vol": 0.0,
                      "total_vol": 0.0, "trade_count": 0,
                      "avg_trade_price": prices[i]}
                     for i in range(len(prices))]
    else:
        stats = [{"buy_vol": 0.0, "sell_vol": 0.0,
                  "total_vol": 0.0, "trade_count": 0,
                  "avg_trade_price": prices[i]}
                 for i in range(len(prices))]

    ai_probs = ai_model_probs(prices, stats)

    total_market = sum(prices)
    total_ai = sum(ai_probs)

    lines: List[str] = []
    lines.append(f"*Question:*\n{question}\n")
    lines.append(f"`{SECTION_DIVIDER}`")
    lines.append(f"*Stake for examples:* `{INVEST_AMOUNT_USDC:.2f}` USDC\n")
    lines.append("*Outcome | Market % | AI % | Edge | Sentiment*")

    for i, price in enumerate(prices):
        name = outcomes[i] if i < len(outcomes) else f"Outcome {i}"
        m_pct = round(price * 100, 2)
        a_pct = round(ai_probs[i] * 100, 2)
        edge = a_pct - m_pct

        total_vol = stats[i]["total_vol"]
        buy_vol = stats[i]["buy_vol"]
        sell_vol = stats[i]["sell_vol"]
        if total_vol > 0:
            sentiment_val = (buy_vol - sell_vol) / total_vol
        else:
            sentiment_val = 0.0

        if sentiment_val > 0.25:
            sent_label = "📈 strong buy flow"
        elif sentiment_val > 0.05:
            sent_label = "🙂 mild buy flow"
        elif sentiment_val < -0.25:
            sent_label = "📉 strong sell flow"
        elif sentiment_val < -0.05:
            sent_label = "🙁 mild sell flow"
        else:
            sent_label = "😐 neutral flow"

        lines.append(
            f"- *{name}*: `{m_pct:.2f}%` → `AI {a_pct:.2f}%` "
            f"(edge: {edge:+.2f}%)  {sent_label}"
        )

        sig_line = trading_signal(price, ai_probs[i])
        lines.append(f"  ↳ {sig_line}")

        rec = recommendation_text(price, ai_probs[i])
        lines.append(f"  ↳ {rec}")

        lines.append("  ↳ " + profit_scenario_text(price))
        lines.append("")

    lines.append(f"Market prob sum: `{total_market:.3f}`")
    lines.append(f"AI prob sum: `{total_ai:.3f}`\n")

    lines.append(build_ai_summary(outcomes, prices, ai_probs))

    whales = detect_whales(trades, MIN_WHALE_USDC)
    if whales:
        lines.append(f"\n*Whale trades (>{MIN_WHALE_USDC:.0f} USDC per trade)*")
        for w in whales:
            outcome_idx = w.get("outcomeIndex")
            if isinstance(outcome_idx, int) and 0 <= outcome_idx < len(outcomes):
                outcome_name = outcomes[outcome_idx]
            else:
                outcome_name = f"Outcome {outcome_idx}"
            lines.append(
                f"- {w.get('side')} {outcome_name}: "
                f"size {w.get('size')} @ {float(w.get('price', 0)):.3f} "
                f"(≈{w['notional']:.0f} USDC)"
            )

        whale_flow = aggregate_whale_flow(whales, len(prices))
        lines.append(f"\n*Whale flow by outcome (>{MIN_WHALE_USDC:.0f} USDC per trade)*")
        for i, flow in enumerate(whale_flow):
            if flow["buy_notional"] == 0 and flow["sell_notional"] == 0:
                continue
            name = outcomes[i] if i < len(outcomes) else f"Outcome {i}"
            lines.append(
                f"- {name}: whales BUY ≈ {flow['buy_notional']:.0f} USDC, "
                f"SELL ≈ {flow['sell_notional']:.0f} USDC"
            )
    else:
        lines.append(
            f"\n_No trades above {MIN_WHALE_USDC:.0f} USDC detected (no whales under current threshold)._"
        )

    text = "\n".join(lines)
    send_message(chat_id, text)

    # store for dashboard
    LAST_ANALYSES.append({
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "question": question,
        "text": text
    })
    if len(LAST_ANALYSES) > 50:
        LAST_ANALYSES.pop(0)


# =========================
# 7. HANDLE LINKS & EVENTS
# =========================

def handle_polymarket_link(chat_id: int, text: str) -> None:
    slug = extract_polymarket_slug(text)
    if not slug:
        send_message(chat_id, "I couldn’t find a Polymarket link in your message.")
        return

    send_message(chat_id, f"🔍 Reading Polymarket slug:\n`{slug}`")

    try:
        market = fetch_market_by_slug(slug)
        analyse_market_object(chat_id, market)
        return
    except ValueError as e:
        if str(e) != "MARKET_NOT_FOUND":
            print("[ERROR] market fetch failed:", e)
            send_message(chat_id, "❌ Error fetching market from Polymarket.")
            return
    except Exception as e:
        print("[ERROR] market fetch exception:", e)

    # Try as event
    try:
        event = fetch_event_by_slug(slug)
    except ValueError as e:
        print("[ERROR] event not found:", e)
        send_message(chat_id, "❌ Could not fetch market or event from Polymarket.")
        return
    except Exception as e:
        print("[ERROR] event fetch failed:", e)
        send_message(chat_id, "❌ Error fetching event from Polymarket.")
        return

    markets = event.get("markets") or []
    if not markets:
        send_message(chat_id, "I found the event, but it doesn’t have any markets yet.")
        return

    PENDING_EVENT_SELECTION[chat_id] = markets

    lines: List[str] = []
    title = event.get("title") or slug
    lines.append(f"*Event:* {title}")
    lines.append("I found these markets in this event:\n")

    max_list = min(15, len(markets))
    for i in range(max_list):
        m = markets[i]
        q = m.get("question") or m.get("groupItemTitle") or m.get("slug") or f"Market {i+1}"
        lines.append(f"{i+1}. {q}")

    if len(markets) > max_list:
        lines.append(f"... and {len(markets) - max_list} more markets not listed here.")

    lines.append("\nReply like: `pick 1` or `pick 2` and I’ll analyse that specific market.")
    send_message(chat_id, "\n".join(lines))


def handle_pick_command(chat_id: int, text: str) -> None:
    m = re.match(r"pick\s+(\d+)", text.strip().lower())
    if not m:
        return

    if chat_id not in PENDING_EVENT_SELECTION:
        send_message(chat_id, "You don’t have any event selection active. Send me a Polymarket event link first.")
        return

    idx = int(m.group(1)) - 1
    markets = PENDING_EVENT_SELECTION[chat_id]
    if not (0 <= idx < len(markets)):
        send_message(chat_id, f"Invalid choice. Please pick a number between 1 and {len(markets)}.")
        return

    market = markets[idx]
    q = market.get("question") or market.get("groupItemTitle") or market.get("slug") or f"Market {idx+1}"
    send_message(chat_id, f"✅ Analysing market #{idx+1}:\n*{q}*")
    analyse_market_object(chat_id, market)


# =========================
# 8. WATCHLIST & AUTO ALERTS
# =========================

def handle_watch_command(chat_id: int, text: str) -> None:
    """
    /watch <polymarket link> [threshold]
    Example:
      /watch https://polymarket.com/... 0.10
    """
    parts = text.split()
    if len(parts) < 2:
        send_message(chat_id, "Usage: `/watch <polymarket link> [edge_threshold]` (example: 0.10 for 10%)")
        return

    slug = extract_polymarket_slug(text)
    if not slug:
        send_message(chat_id, "I couldn’t find a Polymarket link in that message.")
        return

    threshold = EDGE_THRESHOLD
    if len(parts) >= 3:
        try:
            threshold = float(parts[-1])
        except Exception:
            pass

    if chat_id not in WATCHLIST:
        WATCHLIST[chat_id] = {}
    WATCHLIST[chat_id][slug] = {
        "threshold": threshold,
        "last_edge": None,
        "last_alert_ts": 0.0,
    }

    send_message(chat_id, f"🔔 Watching `{slug}` with edge threshold `{threshold:.2f}` (i.e. {threshold*100:.1f}%).")


def handle_unwatch_command(chat_id: int) -> None:
    if chat_id in WATCHLIST:
        WATCHLIST.pop(chat_id)
        send_message(chat_id, "🧹 Cleared all watched markets for this chat.")
    else:
        send_message(chat_id, "You have no watched markets.")


def handle_watches_command(chat_id: int) -> None:
    markets = WATCHLIST.get(chat_id)
    if not markets:
        send_message(chat_id, "You’re not watching any markets.\nUse `/watch <link> [threshold]` to start.")
        return

    lines = ["*Watched markets:*"]
    for slug, info in markets.items():
        lines.append(f"- `{slug}` (edge threshold: {info['threshold']:.2f})")
    send_message(chat_id, "\n".join(lines))


def alert_loop() -> None:
    """
    Background loop that periodically checks all watched markets
    and sends alerts when AI edge crosses the threshold.
    """
    while True:
        try:
            if not WATCHLIST:
                time.sleep(ALERT_CHECK_INTERVAL_SEC)
                continue

            now = time.time()

            for chat_id, markets in list(WATCHLIST.items()):
                for slug, info in list(markets.items()):
                    try:
                        market = fetch_market_by_slug(slug)
                    except Exception as e:
                        print(f"[ALERT] failed to fetch {slug}:", e)
                        continue

                    prices = parse_outcome_prices(market)
                    outcomes = parse_outcomes(market)
                    condition_id = market.get("conditionId")

                    trades: List[Dict[str, Any]] = []
                    stats: List[Dict[str, float]] = []

                    if condition_id:
                        try:
                            trades = fetch_recent_trades(condition_id, RECENT_TRADES_LIMIT)
                            stats = compute_outcome_stats(trades, len(prices))
                        except Exception as e:
                            print("[ALERT] trades fetch failed:", e)
                            stats = [{"buy_vol": 0.0, "sell_vol": 0.0,
                                      "total_vol": 0.0, "trade_count": 0,
                                      "avg_trade_price": prices[i]}
                                     for i in range(len(prices))]
                    else:
                        stats = [{"buy_vol": 0.0, "sell_vol": 0.0,
                                  "total_vol": 0.0, "trade_count": 0,
                                  "avg_trade_price": prices[i]}
                                 for i in range(len(prices))]

                    ai_probs = ai_model_probs(prices, stats)
                    if not ai_probs:
                        continue

                    best_idx = max(range(len(ai_probs)), key=lambda i: abs(ai_probs[i] - prices[i]))
                    best_ai = ai_probs[best_idx]
                    best_mkt = prices[best_idx]
                    best_name = outcomes[best_idx] if best_idx < len(outcomes) else f"Outcome {best_idx}"
                    edge = best_ai - best_mkt
                    abs_edge = abs(edge)

                    threshold = info["threshold"]
                    last_edge = info.get("last_edge")
                    last_alert_ts = info.get("last_alert_ts", 0.0)

                    should_alert = (
                        abs_edge >= threshold and
                        (last_edge is None or abs(last_edge) < threshold * 0.9 or (edge * last_edge) < 0) and
                        (now - last_alert_ts) >= MIN_ALERT_GAP_SEC
                    )

                    if should_alert:
                        question = market.get("question") or market.get("title") or slug
                        sig_line = trading_signal(best_mkt, best_ai)
                        msg = (
                            f"📢 *Auto alert for watched market*\n"
                            f"`{slug}`\n\n"
                            f"*Question:*\n{question}\n\n"
                            f"Top edge outcome: *{best_name}*\n"
                            f"- Market: `{best_mkt*100:.2f}%`\n"
                            f"- AI: `{best_ai*100:.2f}%`\n"
                            f"- Edge: `{(best_ai - best_mkt)*100:+.2f}%`\n\n"
                            f"{sig_line}\n"
                            f"_Threshold: {threshold*100:.1f}% | Auto alerts every ~{ALERT_CHECK_INTERVAL_SEC//60} min_"
                        )
                        send_message(chat_id, msg)
                        info["last_edge"] = edge
                        info["last_alert_ts"] = now
                    else:
                        info["last_edge"] = edge

        except Exception as e:
            print("[ALERT LOOP ERROR]", e)

        time.sleep(ALERT_CHECK_INTERVAL_SEC)


# =========================
# 9. ARBITRAGE SCAN
# =========================

def check_market_arbitrage(slug: str) -> Optional[str]:
    try:
        m = fetch_market_by_slug(slug)
    except Exception as e:
        print(f"[ARB] failed for {slug}:", e)
        return None

    question = m.get("question") or m.get("title") or slug
    prices = parse_outcome_prices(m)
    if not prices:
        return None

    total_prob = sum(prices)
    if 0.98 <= total_prob <= 1.02:
        return None

    outcomes = parse_outcomes(m)
    msg_lines = [f"`{slug}`", question, f"Total probs sum: {total_prob:.3f}"]
    for i, p in enumerate(prices):
        name = outcomes[i] if i < len(outcomes) else f"Outcome {i}"
        msg_lines.append(f"- {name}: {p*100:.2f}%")

    msg_lines.append(
        "_Sum far from 1.0 → maybe mispricing or fees. Do your own check before trading._"
    )
    return "\n".join(msg_lines)


def handle_arb_command(chat_id: int) -> None:
    if not ARBITRAGE_WATCHLIST:
        send_message(chat_id, "Arbitrage watchlist is empty. Edit the code and add some slugs.")
        return

    send_message(chat_id, "🔎 Scanning watchlist for internal arbitrage...")

    hits: List[str] = []
    for slug in ARBITRAGE_WATCHLIST:
        res = check_market_arbitrage(slug)
        if res:
            hits.append(res)

    if not hits:
        send_message(chat_id, "No obvious internal arbitrage detected in watchlist (sum ≈ 1.0 everywhere).")
    else:
        send_message(chat_id, "*Possible internal mispricings:*\n\n" + "\n\n".join(hits))


# =========================
# 10. SIMPLE WEB DASHBOARD (Flask)
# =========================

app = Flask(__name__)

DASHBOARD_TEMPLATE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Polymarket Bot Dashboard</title>
  <style>
    body { font-family: system-ui, -apple-system, sans-serif; background:#0b0f19; color:#f5f5f5; padding:20px; }
    h1 { color:#7ad7ff; }
    .card { border:1px solid #333; padding:12px 16px; margin-bottom:16px; border-radius:8px; background:#111827; }
    .small { font-size:12px; color:#9ca3af; }
    pre { white-space:pre-wrap; word-wrap:break-word; font-size:13px; }
  </style>
</head>
<body>
  <h1>Polymarket AI Bot Dashboard</h1>
  <p class="small">Live summaries of recent analyses and watched markets.</p>

  <h2>Watched Markets</h2>
  {% if not watchlist %}
    <p class="small">No active watches. Use /watch in Telegram.</p>
  {% else %}
    {% for chat_id, mkts in watchlist.items() %}
      <div class="card">
        <div class="small">Chat ID: {{ chat_id }}</div>
        <ul>
        {% for slug, info in mkts.items() %}
          <li><code>{{ slug }}</code> — threshold: {{ '%.2f'|format(info.threshold) }}</li>
        {% endfor %}
        </ul>
      </div>
    {% endfor %}
  {% endif %}

  <h2>Recent Analyses</h2>
  {% if not analyses %}
    <p class="small">No analyses yet.</p>
  {% else %}
    {% for a in analyses %}
      <div class="card">
        <div class="small">{{ a.timestamp }}</div>
        <div><strong>{{ a.question }}</strong></div>
        <pre>{{ a.text }}</pre>
      </div>
    {% endfor %}
  {% endif %}
</body>
</html>
"""

@app.route("/")
def index():
    # create simple structures for template
    watch_display = {}
    for cid, mkts in WATCHLIST.items():
        watch_display[cid] = {}
        for slug, info in mkts.items():
            watch_display[cid][slug] = type("Info", (), info)

    return render_template_string(
        DASHBOARD_TEMPLATE,
        watchlist=watch_display,
        analyses=LAST_ANALYSES[-10:][::-1],
    )


def run_flask():
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)


# =========================
# 11. MAIN LOOP
# =========================

def main() -> None:
    print("Advanced Polymarket bot (with events, AI, alerts, dashboard) started.")
    last_update_id: Optional[int] = None

    # Start alert loop in background
    threading.Thread(target=alert_loop, daemon=True).start()

    while True:
        try:
            params: Dict[str, Any] = {"timeout": 25}
            if last_update_id is not None:
                params["offset"] = last_update_id + 1

            data = telegram_api("getUpdates", params)
        except Exception as e:
            print("[ERROR] getUpdates failed:", e)
            time.sleep(POLL_INTERVAL_SEC)
            continue

        for update in data.get("result", []):
            last_update_id = update["update_id"]
            message = update.get("message")
            if not message:
                continue

            chat_id = message["chat"]["id"]
            text = message.get("text", "")

            if not text:
                continue

            lower = text.strip().lower()

            if lower in ("/start", "start"):
                send_message(
                    chat_id,
                    "Hi! 👋\n"
                    "- Send me any *Polymarket* link (event or market) and I will analyse it.\n"
                    "- If it’s an *event*, I’ll list the markets and you can reply `pick 1`, `pick 2`, etc.\n"
                    "- `/watch <link> [threshold]` → auto alerts when AI edge is big.\n"
                    "- `/watches` → see your watched markets.\n"
                    "- `/unwatch` → clear your watches.\n"
                    "- `/arb` → scan a small watchlist for internal arbitrage.\n\n"
                    "⚠️ This is *not* financial advice. Markets are risky."
                )
                continue

            if lower.startswith("/watch"):
                handle_watch_command(chat_id, text)
                continue

            if lower == "/watches":
                handle_watches_command(chat_id)
                continue

            if lower == "/unwatch":
                handle_unwatch_command(chat_id)
                continue

            if lower == "/arb":
                handle_arb_command(chat_id)
                continue

            if lower.startswith("pick "):
                handle_pick_command(chat_id, lower)
                continue

            if "polymarket.com" in text:
                handle_polymarket_link(chat_id, text)
                continue

            send_message(
                chat_id,
                "Send me a *Polymarket* link (event or market), or use `/watch`, `/arb`, `/start`."
            )

        time.sleep(POLL_INTERVAL_SEC)


if __name__ == "__main__":
    # Run web dashboard + bot loop together
    threading.Thread(target=run_flask, daemon=True).start()
    main()
