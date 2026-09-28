
from __future__ import annotations

import numpy as np
import pandas as pd


def safe_ratio(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return np.divide(a, b, out=np.zeros(len(a), float), where=b != 0)


def entropy_from_counts(c):
    p = c.div(c.sum(axis=1), axis=0).fillna(0)
    return -(p * np.log2(p.where(p > 0, 1))).sum(axis=1)


def add_count_features(f, e, col, prefix, shares=True):
    c = e.groupby(["cookie_id", col], observed=True).size().unstack(fill_value=0)
    c.columns = [f"{prefix}__{x}" for x in c.columns]
    c = c.reindex(f.index, fill_value=0)
    f[c.columns] = c
    if shares:
        total = f["n_events"]
        f[[x + "__share" for x in c.columns]] = c.div(
            total.replace(0, np.nan), axis=0
        ).fillna(0)
    return f


def add_entropy(f, e, col, name):
    c = e.groupby(["cookie_id", col], observed=True).size().unstack(fill_value=0)
    c = c.reindex(f.index, fill_value=0)
    f[name] = entropy_from_counts(c)


def build_features(events, meta, feature_columns=None):
    m = meta[
        ["cookie_id", "cookie_created_at", "window_start_ts", "window_end_ts"]
    ].copy()

    # Full duplicate rows are treated as ingestion duplicates.
    events = events.drop_duplicates(ignore_index=True)

    e = events.merge(m, on="cookie_id", how="inner", validate="many_to_one")
    e = e[
        (e.event_ts >= e.window_start_ts) & (e.event_ts < e.window_end_ts)
    ].copy()
    e = e.sort_values(["cookie_id", "event_ts", "eid"], kind="mergesort")

    e["platform_norm"] = e.platform.astype(str).str.lower()
    e["hour"] = e.event_ts.dt.hour.astype("int8")
    e["dow"] = e.event_ts.dt.dayofweek.astype("int8")
    e["is_night"] = e.hour.isin([0, 1, 2, 3, 4, 5]).astype("int8")
    e["is_weekend"] = (e.dow >= 5).astype("int8")
    e["age_hours"] = (
        e.window_start_ts - e.cookie_created_at
    ).dt.total_seconds() / 3600

    g = e.groupby("cookie_id", sort=False)
    f = g.size().to_frame("n_events")
    f["unique_event_types"] = g.event_name.nunique()
    f["unique_items"] = g.item_id.nunique()
    f["unique_categories"] = g.item_category.nunique()
    f["unique_locations"] = g.item_location.nunique()
    f["unique_seller_types"] = g.seller_type.nunique()
    f["unique_queries"] = g.search_query.nunique()
    f["unique_platforms"] = g.platform_norm.nunique()
    f["unique_user_agents"] = g.user_agent.nunique()
    f["active_span_sec"] = g.event_ts.agg(
        lambda x: (x.iloc[-1] - x.iloc[0]).total_seconds()
    )
    f["active_hours"] = g.event_ts.agg(lambda x: x.dt.floor("h").nunique())
    f["active_days"] = g.event_ts.agg(lambda x: x.dt.date.nunique())
    f["mean_age_hours"] = g.age_hours.mean()
    f["min_age_hours"] = g.age_hours.min()

    add_count_features(f, e, "event_name", "event")
    add_count_features(f, e, "platform_norm", "platform")
    add_count_features(f, e, "seller_type", "seller")
    add_count_features(f, e, "item_category", "category")

    for col, prefix in [
        ("event_name", "event"),
        ("platform_norm", "platform"),
        ("item_category", "category"),
        ("item_location", "location"),
        ("seller_type", "seller"),
    ]:
        add_entropy(f, e, col, prefix + "_entropy")

    sp = e.loc[e.search_page.notna(), ["cookie_id", "search_page"]]
    if len(sp):
        sg = sp.groupby("cookie_id").search_page
        f["search_page_n"] = sg.count()
        f["search_page_mean"] = sg.mean()
        f["search_page_std"] = sg.std().fillna(0)
        f["search_page_max"] = sg.max()
        f["search_page_median"] = sg.median()
        f["search_page_deep_share"] = sg.apply(lambda x: (x >= 5).mean())

    sq = e.loc[e.search_query.notna(), ["cookie_id", "search_query"]].copy()
    if len(sq):
        sq["q_len"] = sq.search_query.astype(str).str.len()
        sq["q_words"] = sq.search_query.astype(str).str.split().str.len()
        qg = sq.groupby("cookie_id")
        qn = qg.size().reindex(f.index, fill_value=0)
        f["query_chars_mean"] = qg.q_len.mean()
        f["query_words_mean"] = qg.q_words.mean()
        f["query_repeat_share"] = (
            1 - safe_ratio(f.unique_queries, qn)
        ).clip(0, 1)

    e["_gap"] = e.groupby("cookie_id").event_ts.diff().dt.total_seconds()
    dg = e.loc[e._gap.notna()].groupby("cookie_id")._gap
    if len(dg):
        f["gap_q10"] = dg.quantile(0.1)
        f["gap_q25"] = dg.quantile(0.25)
        f["gap_median"] = dg.quantile(0.5)
        f["gap_q75"] = dg.quantile(0.75)
        f["gap_q90"] = dg.quantile(0.9)
        f["gap_mean"] = dg.mean()
        f["gap_std"] = dg.std().fillna(0)
        f["gap_min"] = dg.min()
        f["gap_max"] = dg.max()
        f["gap_zero_share"] = dg.apply(lambda x: (x <= 0).mean())
        f["gap_lt_1s_share"] = dg.apply(lambda x: (x < 1).mean())
        f["gap_lt_5s_share"] = dg.apply(lambda x: (x < 5).mean())
        f["gap_lt_30s_share"] = dg.apply(lambda x: (x < 30).mean())

    f["events_per_active_hour"] = safe_ratio(
        f.n_events, f.active_span_sec / 3600 + 1 / 3600
    )
    f["events_per_active_day"] = safe_ratio(f.n_events, f.active_days)

    hc = e.groupby(["cookie_id", "hour"]).size().unstack(fill_value=0)
    hc = hc.reindex(f.index, fill_value=0)
    hc = hc.div(f.n_events, axis=0)
    hc.columns = [f"hour_share__{x}" for x in hc.columns]
    f = f.join(hc)
    f["night_share"] = g.is_night.mean()
    f["weekend_share"] = g.is_weekend.mean()
    add_entropy(f, e, "hour", "hour_entropy")

    p = e.loc[
        e.pointer_x.notna() & e.pointer_y.notna(),
        ["cookie_id", "pointer_x", "pointer_y"],
    ].copy()
    if len(p):
        pg = p.groupby("cookie_id")
        f["pointer_n"] = pg.size()
        f["pointer_x_std"] = pg.pointer_x.std().fillna(0)
        f["pointer_y_std"] = pg.pointer_y.std().fillna(0)
        p["dx"] = pg.pointer_x.diff()
        p["dy"] = pg.pointer_y.diff()
        p["move_dist"] = np.hypot(p.dx.fillna(0), p.dy.fillna(0))
        mg = p.groupby("cookie_id").move_dist
        f["pointer_move_mean"] = mg.mean()
        f["pointer_move_median"] = mg.median()
        f["pointer_move_std"] = mg.std().fillna(0)
        f["pointer_zero_move_share"] = mg.apply(lambda x: (x == 0).mean())

    ua = e.user_agent.fillna("").astype(str)
    flags = pd.DataFrame(index=e.index)
    pats = {
        "headless": "HeadlessChrome|PhantomJS|Headless",
        "chrome": "Chrome/",
        "firefox": "Firefox/",
        "safari": "Safari/",
        "yabrowser": "YaBrowser/",
        "edge": "Edg/",
        "android": "Android",
        "iphone": "iPhone",
        "linux": "Linux",
        "windows": "Windows",
        "macos": "Macintosh|Mac OS X",
    }
    for name, pattern in pats.items():
        flags[name] = ua.str.contains(
            pattern, case=False, regex=True
        ).astype(int)
    flags["cookie_id"] = e.cookie_id.to_numpy()
    ug = flags.groupby("cookie_id")
    for c in pats:
        f["ua_" + c + "_share"] = ug[c].mean()
    f["ua_nunique"] = g.user_agent.nunique()

    pairs = [
        ("event__item_view", "n_events", "item_view_share"),
        ("event__search_results_view", "n_events", "search_share"),
        ("event__photo_swipe", "event__item_view", "swipe_per_item"),
        ("event__favorite_add", "event__item_view", "favorite_per_item"),
        ("event__seller_page_view", "event__item_view", "seller_per_item"),
        ("event__contact_phone_show", "event__item_view", "phone_per_item"),
        ("event__contact_chat_open", "event__item_view", "chat_per_item"),
        ("event__contact_message_sent", "event__item_view", "message_per_item"),
        ("unique_items", "n_events", "items_per_event"),
        ("unique_categories", "n_events", "categories_per_event"),
        ("unique_queries", "event__search_results_view", "queries_per_search"),
    ]
    for a, b, name in pairs:
        if a in f and b in f:
            f[name] = safe_ratio(f[a], f[b])

    f = f.replace([np.inf, -np.inf], np.nan).fillna(0).reset_index()
    out = m[["cookie_id"]].merge(f, on="cookie_id", how="left").fillna(0)

    if feature_columns is not None:
        feature_columns = list(feature_columns)
        for c in feature_columns:
            if c not in out.columns:
                out[c] = 0.0
        out = out[["cookie_id"] + feature_columns]

    return out
