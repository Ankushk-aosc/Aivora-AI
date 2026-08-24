"""Live Market Data (spec Part 1 preferred-provider philosophy: "actually
accessible", never fabricated).

Uses Yahoo Finance's public chart endpoint, which requires no API key.
This is an unofficial, undocumented public endpoint - not a paid/
credentialed integration - so it can break or rate-limit without notice.
Every failure surfaces as "Current market data is not available.", never
a fabricated price, consistent with the earlier explicit design.
"""

import time
import urllib.error
import urllib.request

QUOTE_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
_HEADERS = {"User-Agent": "Mozilla/5.0"}

# Small, explicit name->ticker map for common companies people refer to by
# name rather than symbol. Not exhaustive by design - unmapped names fall
# through to cashtag/bare-ticker extraction, and if that also fails, the
# caller reports "not available" rather than guessing a symbol.
COMPANY_TICKERS = {
    "apple": "AAPL", "tesla": "TSLA", "microsoft": "MSFT", "google": "GOOGL",
    "alphabet": "GOOGL", "amazon": "AMZN", "meta": "META", "facebook": "META",
    "nvidia": "NVDA", "netflix": "NFLX", "intel": "INTC", "amd": "AMD",
    "ibm": "IBM", "oracle": "ORCL", "salesforce": "CRM", "adobe": "ADBE",
    "walmart": "WMT", "disney": "DIS", "coca-cola": "KO", "coca cola": "KO",
    "pepsi": "PEP", "boeing": "BA", "visa": "V", "mastercard": "MA",
    "jpmorgan": "JPM", "goldman sachs": "GS",
}


def extract_symbol(text: str) -> str:
    """Best-effort ticker extraction: a $CASHTAG, a bare 1-5 letter
    uppercase token, or a known company name. Returns None (not a guess)
    if nothing matches."""
    import re
    cashtag = re.search(r"\$([A-Za-z]{1,5})\b", text)
    if cashtag:
        return cashtag.group(1).upper()

    lowered = text.lower()
    for name, ticker in COMPANY_TICKERS.items():
        if name in lowered:
            return ticker

    bare = re.findall(r"\b[A-Z]{2,5}\b", text)
    if bare:
        return bare[0]
    return None


class LiveDataError(RuntimeError):
    pass


def get_quote(symbol: str, timeout: float = 8.0) -> dict:
    """Fetch a real-time-ish quote for `symbol`. Raises LiveDataError on
    any failure (network, unknown symbol, malformed response) - callers
    must catch this and report unavailability, never guess a price."""
    symbol = symbol.strip().upper()
    if not symbol or not symbol.replace(".", "").replace("-", "").isalnum():
        raise LiveDataError(f"'{symbol}' does not look like a valid ticker symbol")

    url = QUOTE_URL.format(symbol=symbol)
    req = urllib.request.Request(url, headers=_HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            import json
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError) as e:
        raise LiveDataError(f"Network error fetching quote for {symbol}: {e}")
    except Exception as e:
        raise LiveDataError(f"Could not parse response for {symbol}: {e}")

    result = (data.get("chart") or {}).get("result")
    if not result:
        error = (data.get("chart") or {}).get("error")
        raise LiveDataError(f"No data for symbol '{symbol}'"
                             + (f": {error}" if error else ""))

    meta = result[0].get("meta", {})
    price = meta.get("regularMarketPrice")
    if price is None:
        raise LiveDataError(f"Response for '{symbol}' had no price field")

    return {
        "symbol": meta.get("symbol", symbol),
        "price": price,
        "currency": meta.get("currency"),
        "exchange": meta.get("fullExchangeName"),
        "previous_close": meta.get("chartPreviousClose"),
        "market_time_unix": meta.get("regularMarketTime"),
        "fetched_at_unix": int(time.time()),
        "source": "Yahoo Finance public chart endpoint (unofficial, no API key)",
    }
