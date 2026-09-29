"""Curated answers for diagnostic questions - "why is X happening?"

The glossary answers "what is X". The baseline's remaining failures included a
second kind of question it cannot serve, and was answering with a definition:

    "A company is profitable but keeps running out of cash. How?"
        -> answered with the definition of EPS
    "EBITDA increased while EBIT decreased. What could explain that?"
        -> answered with the definition of operating margin

These are standard analyst diagnostics with settled explanations, so they are
curated here rather than left to a 101M model that scores 0.88% on its own.

A pattern fires only when EVERY trigger group matches, each group being a list
of interchangeable wordings - the same all-of/any-of shape the evaluation
rubrics use. Requiring two or three independent signals to co-occur is what
keeps precision high: a single keyword can appear in an unrelated question, but
"profit" AND "no cash" together is the situation itself.

The set is written to cover the diagnostic space, not one benchmark's questions:
the pairs an analyst actually checks (accrual vs cash, leverage vs returns,
margin layers, working-capital drag, coverage, dilution). Patterns are matched
in order and the first hit wins, so the more specific ones come first.
"""

import re

# (trigger groups, answer). Every group must match; within a group any wording.
ANALYSIS_PATTERNS = [
    (
        [["profit", "profitable", "earning"], ["out of cash", "running out", "no cash",
                                               "short of cash", "cash problem",
                                               "cash shortage"]],
        "Profit is measured on an accrual basis, so a company can report earnings while "
        "the cash sits somewhere else. The usual culprits are working capital and "
        "financing: receivables the customers have not paid, inventory bought but not "
        "sold, suppliers paid faster than customers pay, capital expenditure, debt "
        "repayments and interest. Check the cash flow statement against net income - "
        "the gap between them is the answer.",
    ),
    (
        [["negative net income", "loss", "losing money", "negative earnings",
          "negative profit"],
         ["positive operating cash flow", "positive cash flow", "cash flow is positive",
          "generate cash", "generating cash"]],
        "Large non-cash charges reduce reported profit without any cash leaving the "
        "business. Depreciation, amortization, impairments, write-downs and share-based "
        "compensation are the common ones, so a company can post a loss and still "
        "collect more cash than it spends on operations.",
    ),
    (
        [["operating cash flow", "cash flow"],
         ["below net income", "less than net income", "lower than net income",
          "far below", "well below"]],
        "Look at working capital movements and non-cash items: receivables and inventory "
        "rising tie cash up, payables falling release it, and accruals or capitalised "
        "costs can flatter profit. Persistent divergence between operating cash flow and "
        "net income is the standard warning sign for earnings quality.",
    ),
    (
        [["inventory", "stock"], ["rising", "increasing", "growing", "building", "up"],
         ["sales are flat", "flat sales", "sales flat", "sales are unchanged",
          "sales have not", "no sales growth", "declining sales", "sales fell"]],
        "Inventory growing faster than sales usually means the goods are not selling. "
        "That risks write-downs or discounting if the stock becomes obsolete, and in the "
        "meantime it ties up cash that has already been spent on production. Check "
        "inventory turnover and days inventory outstanding against prior periods.",
    ),
    (
        [["gross margin"], ["operating margin"],
         ["but", "while", "whereas", "yet", "however"]],
        "The gap between the two margins is everything below the gross line: operating "
        "expenses - selling, general and administrative costs, research, marketing, "
        "depreciation. A high gross margin with a thin operating margin means overheads "
        "are consuming nearly all of the gross profit, so the problem is the cost base "
        "rather than pricing or production.",
    ),
    (
        [["ebitda"], ["ebit", "operating income"],
         ["increase", "increased", "rose", "grew", "higher", "up"],
         ["decrease", "decreased", "fell", "declined", "lower", "down"]],
        "EBITDA excludes depreciation and amortization and EBIT does not, so the two "
        "move apart exactly when those charges change. EBITDA up with EBIT down means "
        "D&A rose - typically after heavy capital expenditure, an acquisition adding "
        "intangibles, or an impairment.",
    ),
    (
        [["roe", "return on equity"],
         ["margin", "margins", "profitability"],
         ["stable", "unchanged", "flat", "same", "constant", "doubled", "rose", "jumped"]],
        "ROE can move without any change in profitability, because it divides by equity. "
        "A smaller equity base lifts it: buybacks, dividends paid out of reserves, or "
        "losses written against equity. More debt does the same thing through leverage. "
        "Decompose it with DuPont - margin x asset turnover x equity multiplier - and the "
        "leverage term will show the cause.",
    ),
    (
        [["revenue", "sales"], ["grew", "growth", "rose", "increased", "up"],
         ["net income fell", "profit fell", "income fell", "earnings fell",
          "net income declined", "profit declined", "net income decreased",
          "less profit", "lower profit"]],
        "Costs grew faster than revenue. Work down the income statement to find where: "
        "cost of goods sold (margin pressure or a worse sales mix), operating expenses "
        "(hiring, marketing, one-off charges), interest (more debt or higher rates), or "
        "tax. A single line usually accounts for most of it.",
    ),
    (
        [["current ratio"], ["0.7", "0.8", "0.9", "below 1", "less than 1", "under 1"]],
        "A current ratio below 1 means current liabilities exceed current assets: the "
        "obligations due within a year are larger than the assets available to meet "
        "them. It is a liquidity warning rather than a verdict - a business with fast "
        "inventory turnover and a reliable credit line can run this way - but it needs "
        "either operating cash flow or new financing to bridge the gap.",
    ),
    (
        [["days sales outstanding", "dso", "receivable", "receivables"],
         ["rose", "rising", "increased", "grew", "up", "higher", "longer"]],
        "Customers are taking longer to pay. Cash that the income statement has already "
        "recognised as revenue has not arrived, which strains liquidity and raises the "
        "risk of bad debts. It can also signal looser credit terms used to win sales, or "
        "revenue recognised too early.",
    ),
    (
        [["share count", "shares outstanding", "number of shares", "share issuance"],
         ["rising", "rose", "increasing", "increased", "growing", "up", "more"]],
        "More shares over the same earnings means earnings per share falls - dilution. "
        "Even with revenue and profit flat, each share represents a smaller claim. The "
        "usual sources are share-based compensation, equity raises and conversions.",
    ),
    (
        [["operating margin"], ["gross margin"],
         ["improved", "rose", "increased", "better", "up"],
         ["fell", "declined", "lower", "down", "worse"]],
        "Operating expenses fell by more than gross profit did. Cost control below the "
        "gross line - overheads, headcount, marketing - can lift the operating margin "
        "even while production economics or sales mix worsen above it.",
    ),
    (
        [["cash", "cash balance"], ["rose", "increased", "grew", "up", "higher"],
         ["free cash flow", "fcf"],
         ["negative", "below zero", "outflow"]],
        "If the business consumed cash and the balance still rose, the money came from "
        "somewhere other than operations: new debt, an equity issue, or selling assets. "
        "Read the financing and investing sections of the cash flow statement - a rising "
        "cash balance funded by borrowing is not the same as one funded by trading.",
    ),
    (
        [["debt", "leverage", "debt/ebitda", "debt to ebitda"],
         ["2x", "5x", "3x", "4x", "doubled", "rose", "increased", "higher"]],
        "Rising debt-to-EBITDA means the borrowings are large relative to the earnings "
        "that service them - either debt grew or earnings fell. Repayment capacity is "
        "weaker, covenants come closer, refinancing gets dearer, and equity holders sit "
        "behind more debt. Check interest coverage alongside it.",
    ),
    (
        [["interest coverage", "times interest earned"],
         ["fell", "falling", "declined", "dropped", "lower", "down"]],
        "Falling interest coverage means earnings cover interest by a smaller margin, so "
        "default risk rises and the balance sheet is materially weaker: less room remains "
        "before the company cannot service its debt from operations. "
        "Coverage near 1x leaves no headroom for a downturn and usually breaches "
        "covenants before it breaches the payment itself.",
    ),
    (
        [["gross margin"], ["fell", "declined", "lower", "down", "fallen"],
         ["revenue rose", "revenue grew", "sales rose", "sales grew",
          "revenue increased", "higher revenue"]],
        "Selling more at a worse margin: input or production costs rose faster than "
        "prices, discounting bought the extra volume, or the sales mix shifted toward "
        "lower-margin products. Check cost of goods sold per unit and the mix by product "
        "line.",
    ),
    (
        [["bond", "bonds", "debt"], ["instead of", "rather than", "not issue", "over"],
         ["share", "shares", "equity", "stock"]],
        "Borrowing raises money without giving away ownership: existing shareholders "
        "keep their proportion of the profits and their votes. Interest is usually "
        "deductible against tax, which makes debt cheaper than equity for a profitable "
        "company, and debt has a defined end - it is repaid, whereas a share is "
        "permanent. The cost is fixed obligations that must be met in bad years.",
    ),
    (
        [["p/e", "pe ratio", "price to earnings", "price-to-earnings"],
         ["different", "differ", "vary", "higher", "lower"],
         ["same industry", "same sector", "competitor", "peers", "two companies"]],
        "A P/E is a statement about the future, not the past. Two similar businesses "
        "trade apart when the market expects different growth, sees different risk "
        "(leverage, customer concentration, regulation), or trusts one company's "
        "earnings more than the other's. Accounting differences and one-off items in the "
        "denominator explain much of the rest.",
    ),
    (
        [["inventory accounting", "fifo", "lifo", "inventory method", "inventory methods"],
         ["profit", "income", "earnings", "cost"]],
        "FIFO and LIFO assign different costs to the goods that were sold. When input "
        "prices are rising, FIFO charges the older, cheaper stock to cost of goods sold "
        "and reports higher profit, while LIFO charges the newest, dearest stock and "
        "reports lower profit with a lower tax bill. The physical inventory is identical; "
        "only the cost assignment differs.",
    ),
    (
        [["quarterly reporting", "quarterly report", "quarterly results",
          "report quarterly", "quarterly disclosure"],
         ["why", "require", "required", "regulator", "mandate", "obliged"]],
        "Frequent reporting keeps investors informed on a timely basis and narrows the "
        "information gap between management and the market, which supports fair pricing "
        "and makes selective disclosure or insider advantage harder. The trade-off, long "
        "argued, is cost and a pull toward short-term decisions.",
    ),
]


def _score(groups, text):
    """Evidence for one pattern, or None when a required group is missing.

    Declaration order used to decide ties, which let a general pattern shadow a
    specific one: "Operating margin improved while gross margin fell" matched the
    general gross-vs-operating pattern rather than the one written for exactly
    that divergence. Scoring replaces order.

    Returns (groups_required, options_matched, longest_option) - more required
    groups means a more specific pattern, more matched options means more
    corroborating signals, and a longer matched phrase is more particular than a
    single word.
    """
    options_matched, longest = 0, 0
    for group in groups:
        hits = [option for option in group if option in text]
        if not hits:
            return None
        options_matched += len(hits)
        longest = max(longest, max(len(option) for option in hits))
    return (len(groups), options_matched, longest)


# A pattern needs at least this many groups before it may answer, so that one
# keyword can never trigger a diagnostic: "profit" alone must not select a
# profitability-deterioration explanation.
MIN_REQUIRED_GROUPS = 2


def match_pattern(question: str, min_groups: int = MIN_REQUIRED_GROUPS):
    """The best-supported curated diagnostic answer, or None.

    Every candidate is scored and the strongest wins, so adding a pattern cannot
    silently shadow an existing one by being declared earlier.
    """
    if not question:
        return None
    text = re.sub(r"\s+", " ", question.lower())

    best = None
    for index, (groups, answer) in enumerate(ANALYSIS_PATTERNS):
        if len(groups) < min_groups:
            continue
        score = _score(groups, text)
        if score is None:
            continue
        candidate = (score, index, groups, answer)
        if best is None or candidate[0] > best[0]:
            best = candidate
    if best is None:
        return None
    score, index, groups, answer = best
    return {"text": answer, "pattern": index,
            "signals": [group[0] for group in groups],
            "required_groups": score[0], "options_matched": score[1],
            "confidence": round(min(0.95, 0.6 + 0.1 * score[0] + 0.02 * score[1]), 2)}
