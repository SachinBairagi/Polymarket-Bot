import time
import json
import re
import math
import os
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional, Set

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

# Example default stake used in explanations / EV
INVEST_AMOUNT_USDC = 50.0

# When AI vs market difference is larger than this, we call it an “edge”
EDGE_THRESHOLD = 0.07  # 7 percentage points

# Whale definition (USDC notional: size * price)
MIN_WHALE_USDC = 1000.0

# Markets to include in /arb scan
ARBITRAGE_WATCHLIST: List[str] = [
    "will-polymarket-us-go-live-in-2025",
]

# Event support: remember which event's markets a user can pick from
PENDING_EVENT_SELECTION: Dict[int, List[Dict[str, Any]]] = {}

# Edge / watchlist alerts
WATCHLIST: Dict[int, Dict[str, Dict[str, Any]]] = {}
ALERT_CHECK_INTERVAL_SEC = 900  # 15 min
MIN_ALERT_GAP_SEC = 1800        # min 30 min between edge alerts

# Whale auto-alert subscriptions (no /watch needed)
# WHALE_SUBSCRIPTIONS[chat_id][condition_id] = {slug, last_seen_time}
WHALE_SUBSCRIPTIONS: Dict[int, Dict[str, Dict[str, Any]]] = {}

# Per chat on/off toggle for whale/global alerts
WHALE_ALERT_ENABLED: Dict[int, bool] = {}
WHALE_ALERT_INTERVAL_SEC = 300  # 5 min

# Track all chats that have talked to the bot
ACTIVE_CHATS: Set[int] = set()

# For New Event Monitor: keeps track of events already announced
SEEN_EVENT_IDS: Set[str] = set()

# Bet calculator context
# PENDING_BET_CONTEXT[chat_id] = { 'slug', 'outcome', 'price', 'ai_prob' }
PENDING_BET_CONTEXT: Dict[int, Dict[str, Any]] = {}

# Pretty formatting + dashboard
SECTION_DIVIDER = "────────────────────────"
LAST_ANALYSES: List[Dict[str, Any]] = []

# ========= Multiple Polymarket API keys (rotation) =========
# Optional: set POLY_API_KEY_1, POLY_API_KEY_2, POLY_API_KEY_3 in env.
POLYMARKET_API_KEYS: List[str] = [
    os.getenv("POLY_API_KEY_1") or "",
    os.getenv("POLY_API_KEY_2") or "",
    os.getenv("POLY_API_KEY_3") or "",
]
POLYMARKET_API_KEYS = [k for k in POLYMARKET_API_KEYS if k]
_API_KEY_INDEX = 0  # internal pointer for rotation

# ========= Global watchlist edge scanner (push signals) =========
GLOBAL_STATE: Dict[str, Dict[str, Any]] = {}
GLOBAL_ALERT_INTERVAL_SEC = 60  # check every minute
GLOBAL_EDGE_THRESHOLD = 0.08    # 8% AI edge
GLOBAL_ALERT_MIN_GAP_SEC = 300  # min 5 min between alerts per market

TRENDING_LIMIT = 20             # how many “overall” markets to scan
CATEGORY_LIMIT = 15             # how many per category

# Category slugs used on Polymarket site - UPDATED to include 'breaking'
CATEGORY_TAG_SLUGS: List[str] = [
    "breaking",  # <-- ADDED for BreakingNew
    "politics",
    "sports",
    "finance",
    "crypto",
    "geopolitics",
    "earnings",
    "tech",
    "culture",
    "world",
    "economy",
    "elections",
]

# cache tag_slug -> tag_id so we don't call API every time
TAG_ID_CACHE: Dict[str, str] = {}

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

def rotated_get(url: str, params: Dict[str, Any] = None, timeout: int = 10) -> requests.Response:
    """Wrapper around requests.get that rotates through multiple API keys."""
    global _API_KEY_INDEX
    headers: Dict[str, str] = {}

    if POLYMARKET_API_KEYS:
        key = POLYMARKET_API_KEYS[_API_KEY_INDEX % len(POLYMARKET_API_KEYS)]
        _API_KEY_INDEX += 1
        headers["Authorization"] = f"Bearer {key}"

    return requests.get(url, params=params, headers=headers, timeout=timeout)

def fetch_market_by_slug(slug: str) -> Dict[str, Any]:
    url = f"{GAMMA_BASE}/markets/slug/{slug}"
    resp = rotated_get(url, timeout=10)
    if resp.status_code == 404:
        raise ValueError("MARKET_NOT_FOUND")
    resp.raise_for_status()
    return resp.json()

def fetch_event_by_slug(slug: str) -> Dict[str, Any]:
    url = f"{GAMMA_BASE}/events/slug/{slug}"
    resp = rotated_get(url, timeout=10)
    if resp.status_code == 404:
        raise ValueError("EVENT_NOT_FOUND")
    resp.raise_for_status()
    return resp.json()

def fetch_recent_trades(condition_id: str, limit: int = RECENT_TRADES_LIMIT) -> List[Dict[str, Any]]:
    params = {"limit": limit, "market": condition_id}
    resp = rotated_get(f"{DATA_BASE}/trades", params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()

def fetch_trending_slugs(limit: int = TRENDING_LIMIT) -> List[str]:
    # ... (existing function) ...
    url = f"{GAMMA_BASE}/markets"
    params = {
        "limit": limit,
        "closed": "false",
        "order": "volume24h",
        "ascending": "false",
    }
    resp = rotated_get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    if isinstance(data, dict):
        markets = data.get("markets") or data.get("data") or []
    else:
        markets = data

    slugs: List[str] = []
    for m in markets:
        slug = m.get("slug")
        if slug:
            slugs.append(slug)
    return slugs

def fetch_newest_events(limit: int = 5) -> List[Dict[str, Any]]:
    """Fetch the newest events based on start date."""
    url = f"{GAMMA_BASE}/events"
    params = {
        "limit": limit,
        "closed": "false",
        "order": "startDate",
        "ascending": "false",
    }
    try:
        resp = rotated_get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get("events") or data.get("data") or []
    except Exception as e:
        print(f"[EVENT FETCH] Failed to get newest events: {e}")
        return []

def get_tag_id_by_slug(tag_slug: str) -> Optional[str]:
    # ... (existing function) ...
    if tag_slug in TAG_ID_CACHE:
        return TAG_ID_CACHE[tag_slug]

    url = f"{GAMMA_BASE}/tags/slug/{tag_slug}"
    try:
        resp = rotated_get(url, timeout=10)
        if resp.status_code != 200:
            print(f"[TAGS] failed for slug={tag_slug}: {resp.status_code}")
            return None
        data = resp.json()
    except Exception as e:
        print(f"[TAGS] exception for slug={tag_slug}:", e)
        return None

    tag_id = data.get("id")
    if not tag_id:
        print(f"[TAGS] no id in tag for slug={tag_slug}")
        return None

    tag_id_str = str(tag_id)
    TAG_ID_CACHE[tag_slug] = tag_id_str
    return tag_id_str

def fetch_markets_for_tag(tag_slug: str, limit: int = CATEGORY_LIMIT) -> List[Dict[str, Any]]:
    # ... (existing function) ...
    tag_id = get_tag_id_by_slug(tag_slug)
    if not tag_id:
        return []

    params = {
        "tag_id": tag_id,
        "closed": "false",
        "limit": limit,
        "order": "volume24h",
        "ascending": "false",
    }
    url = f"{GAMMA_BASE}/markets"
    try:
        resp = rotated_get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[CAT] markets fetch failed for tag={tag_slug}:", e)
        return []

    if isinstance(data, dict):
        markets = data.get("markets") or data.get("data") or []
    else:
        markets = data

    return markets or []

def discover_global_slugs() -> List[str]:
    # ... (existing function) ...
    slugs_set = set()

    # overall
    try:
        for s in fetch_trending_slugs():
            slugs_set.add(s)
    except Exception as e:
        print("[GLOBAL] overall trending fetch failed:", e)

    # per category
    for cat_slug in CATEGORY_TAG_SLUGS:
        try:
            markets = fetch_markets_for_tag(cat_slug, CATEGORY_LIMIT)
        except Exception as e:
            print(f"[GLOBAL] category fetch failed for {cat_slug}:", e)
            continue

        for m in markets:
            s = m.get("slug")
            if s:
                slugs_set.add(s)

    return list(slugs_set)

def parse_outcome_prices(market: Dict[str, Any]) -> List[float]:
    # ... (existing function) ...
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
    # ... (existing function) ...
    outcomes = market.get("outcomes") or []
    if isinstance(outcomes, str):
        try:
            outcomes = json.loads(outcomes)
        except json.JSONDecodeError:
            outcomes = []
    return [str(o) for o in outcomes]

# =========================
# 4. FEATURES, AI MODEL, WHALES
# (AI logic is already robust for prediction/edge)
# =========================

def compute_outcome_stats(trades: List[Dict[str, Any]], n_outcomes: int) -> List[Dict[str, float]]:
    # ... (existing function) ...
    stats = [
        {"buy_vol": 0.0, "sell_vol": 0.0, "total_vol": 0.0,
         "trade_count": 0, "avg_trade_price": 0.0}
        for _ in range(n_outcomes)
    ]

    for t in trades:
        idx = t.get("outcomeIndex")
        if idx is None: continue
        try: idx = int(idx)
        except Exception: continue
        if not (0 <= idx < n_outcomes): continue

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
    # ... (existing function) ...
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

def ai_model_probs(prices: List[float], stats: List[Dict[str, float]]) -> List[float]:
    # ... (existing function - the accurate prediction model) ...
    n = len(prices)
    if n == 0: return []

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
            sentiment = (buy_vol - sell_vol) / total_vol
        else:
            sentiment = 0.0

        log_vol = math.log10(1.0 + total_vol)
        momentum = (p - avg_trade_price) if avg_trade_price > 0 else 0.0

        logit = w0 + w1 * p + w2 * sentiment + w3 * log_vol + w4 * momentum
        raw = sigmoid(logit)
        raw_scores.append(raw)

    total_raw = sum(raw_scores)
    if total_raw <= 0: return prices[:]
    return [r / total_raw for r in raw_scores]

def detect_whales(trades: List[Dict[str, Any]], min_usdc: float) -> List[Dict[str, Any]]:
    # ... (existing function) ...
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
    return whales[:10]

def aggregate_whale_flow(whales: List[Dict[str, Any]], n_outcomes: int) -> List[Dict[str, float]]:
    # ... (existing function) ...
    flow = [
        {"buy_notional": 0.0, "sell_notional": 0.0}
        for _ in range(n_outcomes)
    ]
    for w in whales:
        idx = w.get("outcomeIndex")
        try: idx = int(idx)
        except Exception: continue
        if not (0 <= idx < n_outcomes): continue

        side = (w.get("side") or "").upper()
        notional = float(w.get("notional", 0.0) or 0.0)
        if side == "BUY":
            flow[idx]["buy_notional"] += notional
        elif side == "SELL":
            flow[idx]["sell_notional"] += notional
    return flow

# ... (Sections 5, 6, 7, 8 are unchanged - use existing code) ...

# =========================
# 9.7 NEW EVENT MONITOR (ADDED)
# =========================
def new_event_monitor() -> None:
    """
    Checks for newly created events and broadcasts alerts to all active users.
    """
    print("New Event Monitor started.")
    # Fetch 50 most recent events to populate initial SEEN_EVENT_IDS
    try:
        initial_events = fetch_newest_events(limit=50)
        for event in initial_events:
            SEEN_EVENT_IDS.add(str(event.get("id")))
    except:
        pass # Ignore failure on startup

    while True:
        try:
            if not ACTIVE_CHATS:
                time.sleep(60)
                continue

            events = fetch_newest_events(limit=5)
            
            for event in events:
                event_id = str(event.get("id"))
                if event_id not in SEEN_EVENT_IDS:
                    SEEN_EVENT_IDS.add(event_id)
                    
                    title = event.get("title", "New Event")
                    slug = event.get("slug")
                    
                    msg = (
                        f"🎉 **NEW POLYMARKET EVENT DETECTED**\n"
                        f"📌 *{title}*\n"
                        f"🔗 [View Event](https://polymarket.com/event/{slug})\n\n"
                        f"Send the link to me to analyze the markets!"
                    )
                    
                    # Broadcast the alert to all active users
                    for chat_id in list(ACTIVE_CHATS):
                        if WHALE_ALERT_ENABLED.get(chat_id, True): # Check if user disabled alerts
                            send_message(chat_id, msg)
                    
                    time.sleep(5) # Delay between alerts if multiple new events land at once

        except Exception as e:
            print(f"[NEW EVENT MONITOR ERROR] {e}")

        time.sleep(60) # Check every 60 seconds

# ... (Other loops: whale_alert_loop, global_watch_loop, alert_loop are unchanged - use existing code) ...

# =========================
# 12. MAIN LOOP (Updated)
# =========================

def run_flask():
    # ... (existing run_flask) ...
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)

def main() -> None:
    print("Advanced Polymarket bot (events, AI, whale alerts with AI, global scanner, bet calc, dashboard) started.")

    last_update_id: Optional[int] = None

    # Background loops (All existing + the new one)
    threading.Thread(target=run_flask, daemon=True).start()
    threading.Thread(target=alert_loop, daemon=True).start()
    threading.Thread(target=whale_alert_loop, daemon=True).start()
    threading.Thread(target=global_watch_loop, daemon=True).start()
    
    # START THE NEW EVENT MONITOR THREAD (NEW)
    threading.Thread(target=new_event_monitor, daemon=True).start() 

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

            ACTIVE_CHATS.add(chat_id)

            lower = text.strip().lower()

            # /start
            if lower in ("/start", "start"):
                send_message(
                    chat_id,
                    "Hi! 👋\n"
                    "- Send me any *Polymarket* link (event or market) and I will analyse it.\n"
                    "- If it’s an *event*, I’ll list markets and you can reply `pick 1`, `pick 2`, etc.\n"
                    "- I show AI vs market %, trading signals, whales, and profit examples.\n"
                    "- After analysis, reply with an amount (e.g. `50`) for an AI bet calculator.\n"
                    "- **Whale alerts** run every 5 min on markets you send and now include AI edge + example bet.\n"
                    "- **New Event Alerts** run every 1 min.\n"
                    "- A **global scanner** runs every ~1 min across all categories and pushes top AI edges + **arb hints**.\n"
                    "- `/watch <link> [threshold]` → edge-based alerts for specific markets.\n"
                    "- `/watches`, `/unwatch` to manage your watchlist.\n"
                    "- `/alerts_on` / `/alerts_off` to toggle all push alerts.\n"
                    "- `/arb` → scan a small custom watchlist for internal arbitrage.\n\n"
                    "⚠️ This is *not* financial advice. Markets are risky."
                )
                continue

            # Alerts on/off
            if lower == "/alerts_off":
                handle_alerts_toggle(chat_id, False)
                continue
            if lower == "/alerts_on":
                handle_alerts_toggle(chat_id, True)
                continue

            # Watchlist commands
            if lower.startswith("/watch"):
                handle_watch_command(chat_id, text)
                continue
            if lower == "/watches":
                handle_watches_command(chat_id)
                continue
            if lower == "/unwatch":
                handle_unwatch_command(chat_id)
                continue

            # Arbitrage command
            if lower == "/arb":
                handle_arb_command(chat_id)
                continue

            # Event pick
            if lower.startswith("pick "):
                handle_pick_command(chat_id, lower)
                continue

            # Bet calculator amount
            if chat_id in PENDING_BET_CONTEXT and re.fullmatch(r"\d+(\.\d+)?", lower):
                handle_bet_amount(chat_id, text)
                continue

            if lower == "skip" and chat_id in PENDING_BET_CONTEXT:
                PENDING_BET_CONTEXT.pop(chat_id, None)
                send_message(chat_id, "👍 Skipped bet calculator for this market. Send a new Polymarket link anytime.")
                continue

            # Polymarket links
            if "polymarket.com" in text:
                handle_polymarket_link(chat_id, text)
                continue

            # Fallback
            send_message(
                chat_id,
                "Send me a *Polymarket* link (event or market), "
                "or use `/start`, `/alerts_off`, `/alerts_on`, `/watch`, `/arb`."
            )

        time.sleep(POLL_INTERVAL_SEC)

if __name__ == "__main__":
    # Your existing flask thread (modified to include all new background loops)
    # The existing code already had a thread starting 'run_flask' and then calling 'main()'
    threading.Thread(target=run_flask, daemon=True).start()
    
    # Start all monitoring threads within main() as planned for your structure:
    main()