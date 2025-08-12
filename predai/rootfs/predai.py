#!/usr/bin/env python3
"""
PredAI (fork) — config‑driven multi‑horizon forecasting for Home Assistant.

Key features:
* Configurable sensor roles (energy counters, temperature, Mixergy immersion demand, etc.).
* Automatic power→energy integration for non‑cumulative power sensors (mean power -> kWh/interval).
* Covariate alias + scaling from YAML covariates: section.
* Interval, cumulative, daily_cum, and horizon (+2h/+8h/+12h or custom) forecast publishing.
* Async Home Assistant API via aiohttp.
* SQLite history cache.
* Config hot‑reload each cycle.

British spelling used in comments.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import json

import numpy as np
import pandas as pd
import yaml

# NeuralProphet
try:
    from neuralprophet import NeuralProphet, set_log_level as np_set_log_level
except Exception as e:  # pragma: no cover
    NeuralProphet = None
    _NP_IMPORT_ERROR = e
else:
    _NP_IMPORT_ERROR = None
    np_set_log_level("ERROR")

import aiohttp
import aiohttp.client_exceptions

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

TIMEOUT = 240
TIME_FORMAT_HA = "%Y-%m-%dT%H:%M:%S%z"
TIME_FORMAT_HA_DOT = "%Y-%m-%dT%H:%M:%S.%f%z"

DEFAULT_CONFIG_PATH = "/config/predai.yaml"
DEFAULT_DB_PATH = "/config/predai.db"

DEFAULT_PUBLISH_PREFIX = "predai_"
DEFAULT_INTERVAL_MIN = 30
DEFAULT_HORIZONS_MIN = [120, 480, 720]  # +2h, +8h, +12h

SAFE_TBL_RE = re.compile(r"^[A-Za-z0-9_]+$")

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

logger = logging.getLogger("predai")
if not logger.handlers:
    h = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    h.setFormatter(fmt)
    logger.addHandler(h)
logger.setLevel(logging.DEBUG)

# --------------------------------------------------------------------------- #
# Utility: timestamps
# --------------------------------------------------------------------------- #

def timestr_to_datetime(timestamp: str) -> Optional[datetime]:
    """Parse a Home‑Assistant timestamp and return a tz‑aware datetime
    rounded to the nearest minute (seconds & microseconds cleared).

    Accepts:
      • ISO‑8601 with colon in offset, e.g. '2025‑07‑28T10:47:12+01:00'
      • ISO‑8601 with fractional seconds
      • Legacy HA formats without the colon
    """
    if not timestamp:
        return None

    # 1.  Robust ISO parser (handles “+01:00”)
    try:
        dt = datetime.fromisoformat(timestamp)
        return dt.replace(second=0, microsecond=0)
    except ValueError:
        pass  # fall back to legacy formats

    # 2.  Legacy formats
    for fmt in (TIME_FORMAT_HA, TIME_FORMAT_HA_DOT):
        try:
            dt = datetime.strptime(timestamp, fmt)
            return dt.replace(second=0, microsecond=0)
        except ValueError:
            continue

    return None


def ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


# --------------------------------------------------------------------------- #
# Config dataclasses
# --------------------------------------------------------------------------- #

@dataclass
class RoleCfg:
    target_transform: str = "interval"      # interval|level|cumulative
    aggregation: str = "sum"                # sum|mean|last
    publish_state_class: str = "measurement"
    model_backend: str = "neuralprophet"
    n_lags: int = 8
    seasonality_reg: float = 5.0
    seasonality_mode: str = "additive"
    learning_rate: Optional[float] = None


@dataclass
class ResetDetectionCfg:
    enabled: bool = False
    low: float = 1.0
    high: float = 2.0
    hard_reset_value: Optional[float] = None


@dataclass
class SensorCfg:
    name: str
    role: str
    units: str = ""
    output_units: Optional[str] = None        # override publish units (e.g., W in, kWh out)
    days_hist: int = 7
    export_days: Optional[int] = None

    subtract: List[str] = field(default_factory=list)
    incrementing: bool = False
    reset_daily: bool = False
    interval: Optional[int] = None
    future_periods: Optional[int] = None
    reset_low: Optional[float] = None
    reset_high: Optional[float] = None

    source_is_cumulative: bool = False
    train_target: str = "interval"            # interval|level|cumulative
    aggregation: Optional[str] = None         # resample override
    log_transform: bool = False

    reset_detection: ResetDetectionCfg = field(default_factory=ResetDetectionCfg)

    publish_interval: bool = True
    publish_cumulative: bool = True
    publish_daily_cumulative: bool = True

    covariates_future: List[str] = field(default_factory=list)
    covariates_lagged: List[str] = field(default_factory=list)

    n_lags: Optional[int] = None
    seasonality_reg: Optional[float] = None
    seasonality_mode: Optional[str] = None
    learning_rate: Optional[float] = None
    country: Optional[str] = None

    database: bool = True
    max_age: int = 365
    max_increment: Optional[float] = None

    plot: bool = False
    cascade_outputs: Dict[str, bool] = field(default_factory=dict)
    publish_name: Optional[str] = None

    def effective_aggregation(self, role_cfg: RoleCfg) -> str:
        return self.aggregation or role_cfg.aggregation

    def effective_n_lags(self, role_cfg: RoleCfg) -> int:
        return self.n_lags if self.n_lags is not None else role_cfg.n_lags

    def effective_seasonality_reg(self, role_cfg: RoleCfg) -> float:
        return self.seasonality_reg if self.seasonality_reg is not None else role_cfg.seasonality_reg

    def effective_seasonality_mode(self, role_cfg: RoleCfg) -> str:
        return self.seasonality_mode or role_cfg.seasonality_mode

    def effective_learning_rate(self, role_cfg: RoleCfg) -> Optional[float]:
        return self.learning_rate if self.learning_rate is not None else role_cfg.learning_rate


@dataclass
class PredAIConfig:
    update_every: int = 30
    common_interval: int = DEFAULT_INTERVAL_MIN
    horizons: List[int] = field(default_factory=lambda: DEFAULT_HORIZONS_MIN)
    publish_prefix: str = DEFAULT_PUBLISH_PREFIX
    defaults: Dict[str, Any] = field(default_factory=dict)
    roles: Dict[str, RoleCfg] = field(default_factory=dict)
    sensors: List[SensorCfg] = field(default_factory=list)
    timezone_name: str = "Europe/London"
    cov_map: Dict[str, Any] = field(default_factory=dict)

    @property
    def tz(self):
        try:
            from zoneinfo import ZoneInfo
            return ZoneInfo(self.timezone_name)
        except Exception:
            logger.warning("Could not load timezone %s; falling back to UTC.", self.timezone_name)
            return timezone.utc


# --------------------------------------------------------------------------- #
# Config loading
# --------------------------------------------------------------------------- #

def _load_role(name: str, d: Dict[str, Any]) -> RoleCfg:
    return RoleCfg(
        target_transform=d.get("target_transform", "interval"),
        aggregation=d.get("aggregation", "sum"),
        publish_state_class=d.get("publish_state_class", "measurement"),
        model_backend=d.get("model", {}).get("backend", "neuralprophet"),
        n_lags=d.get("model", {}).get("n_lags", 8),
        seasonality_reg=d.get("model", {}).get("seasonality_reg", 5.0),
        seasonality_mode=d.get("model", {}).get("seasonality_mode", "additive"),
        learning_rate=d.get("model", {}).get("learning_rate"),
    )


def _load_sensor(dflt: Dict[str, Any], d: Dict[str, Any]) -> SensorCfg:
    merged = dict(dflt)
    merged.update(d)
    rd_map = merged.get("reset_detection", {})
    rd = ResetDetectionCfg(
        enabled=rd_map.get("enabled", False),
        low=rd_map.get("low", 1.0),
        high=rd_map.get("high", 2.0),
        hard_reset_value=rd_map.get("hard_reset_value"),
    )
    subtract_val = merged.get("subtract", [])
    if isinstance(subtract_val, str):
        subtract_list = [subtract_val]
    else:
        subtract_list = list(subtract_val) if subtract_val else []

    incrementing_val = merged.get("incrementing")
    if incrementing_val is not None:
        merged.setdefault("source_is_cumulative", bool(incrementing_val))

    if merged.get("reset_low") is not None:
        rd.low = float(merged["reset_low"])
    if merged.get("reset_high") is not None:
        rd.high = float(merged["reset_high"])
    return SensorCfg(
        name=merged["name"],
        role=merged.get("role", "incrementing_energy"),
        units=merged.get("units", ""),
        output_units=merged.get("output_units"),
        days_hist=merged.get("days", merged.get("days_hist", 7)),
        export_days=merged.get("export_days"),
        subtract=subtract_list,
        incrementing=bool(incrementing_val) if incrementing_val is not None else False,
        reset_daily=merged.get("reset_daily", False),
        interval=merged.get("interval"),
        future_periods=merged.get("future_periods"),
        reset_low=merged.get("reset_low"),
        reset_high=merged.get("reset_high"),
        source_is_cumulative=merged.get("source_is_cumulative", False),
        train_target=merged.get("train_target", "interval"),
        aggregation=merged.get("aggregation"),
        log_transform=merged.get("log_transform", False),
        reset_detection=rd,
        publish_interval=merged.get("publish_interval", True),
        publish_cumulative=merged.get("publish_cumulative", True),
        publish_daily_cumulative=merged.get("publish_daily_cumulative", True),
        covariates_future=merged.get("covariates_future", []) or [],
        covariates_lagged=merged.get("covariates_lagged", []) or [],
        n_lags=merged.get("n_lags"),
        seasonality_reg=merged.get("seasonality_reg"),
        seasonality_mode=merged.get("seasonality_mode"),
        learning_rate=merged.get("learning_rate"),
        country=merged.get("country"),
        database=merged.get("database", True),
        max_age=merged.get("max_age", 365),
        max_increment=merged.get("max_increment"),
        plot=merged.get("plot", False),
        cascade_outputs=merged.get("cascade_outputs", {}) or {},
        publish_name=merged.get("publish_name"),
    )


def load_config(path: str = DEFAULT_CONFIG_PATH) -> PredAIConfig:
    """Load YAML configuration for PredAI.

    A short info log is emitted with the path being loaded so users can
    verify which file is in use. If the file is missing, the function
    falls back to defaults and reports an error.
    """
    logger.info("Loading configuration from %s", path)
    if not os.path.exists(path):
        logger.error("Configuration file %s not found.", path)
        return PredAIConfig()
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}

    dflt = raw.get("defaults", {})
    publish_prefix = dflt.get("publish_prefix", DEFAULT_PUBLISH_PREFIX)

    roles_raw = raw.get("roles", {}) or {}
    roles = {k: _load_role(k, v) for k, v in roles_raw.items()}

    sensors_raw = raw.get("sensors", []) or []
    sensors: List[SensorCfg] = []
    for s in sensors_raw:
        s_clean = {k: v for k, v in s.items() if k != "roles"}  # ignore stray nested roles
        sensors.append(_load_sensor(dflt, s_clean))

    logger.info("Configuration: %s roles, %s sensors", len(roles), len(sensors))

    cfg = PredAIConfig(
        update_every=raw.get("update_every", 30),
        common_interval=raw.get("common_interval", DEFAULT_INTERVAL_MIN),
        horizons=raw.get("horizons", DEFAULT_HORIZONS_MIN),
        publish_prefix=publish_prefix,
        defaults=dflt,
        roles=roles,
        sensors=sensors,
        timezone_name=raw.get("timezone", "Europe/London"),
        cov_map=raw.get("covariates", {}) or {},
    )
    return cfg


# --------------------------------------------------------------------------- #
# Home Assistant Interface
# --------------------------------------------------------------------------- #

class HAInterface:
    def __init__(self, ha_url: Optional[str], ha_key: Optional[str], session: Optional[aiohttp.ClientSession] = None):
        self.ha_url = ha_url or "http://supervisor/core"
        self.ha_key = ha_key or os.environ.get("SUPERVISOR_TOKEN")
        if not self.ha_key:
            raise RuntimeError("No Home Assistant key found.")
        self._session = session
        mask = self.ha_key[:6] + "…" if len(self.ha_key) >= 6 else "***"
        logger.info("HA Interface initialised (token %s, url %s)", mask, self.ha_url)

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=TIMEOUT)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def api_call(self, method: str, endpoint: str, params: Optional[dict] = None, json_data: Optional[dict] = None) -> Any:
        url = self.ha_url.rstrip("/") + endpoint
        sess = await self._get_session()
        headers = {
            "Authorization": f"Bearer {self.ha_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        try:
            async with sess.request(method, url, headers=headers, params=params, json=json_data) as resp:
                text = await resp.text()
                if resp.status >= 400:
                    logger.warning("HA API %s %s -> %s: %s", method, endpoint, resp.status, text)
                try:
                    return json.loads(text) if text else None
                except json.JSONDecodeError:
                    logger.error("Non‑JSON response from %s: %s", url, text)
                    return None
        except aiohttp.client_exceptions.ClientError as e:
            logger.error("HA API error %s %s: %s", method, endpoint, e)
            return None

    async def get_history(self, sensor: str, start: datetime, end: datetime) -> Tuple[List[dict], Optional[datetime], Optional[datetime]]:
        params = {
            "filter_entity_id": sensor,
            "end_time": end.strftime(TIME_FORMAT_HA),
        }
        endpoint = "/api/history/period/" + start.strftime(TIME_FORMAT_HA)
        res = await self.api_call("GET", endpoint, params=params)
        if not res:
            logger.warning("No history for %s", sensor)
            return [], None, None
        arr = res[0] if isinstance(res, list) and res else []
        try:
            st = timestr_to_datetime(arr[0]["last_updated"]) if arr else None
            en = timestr_to_datetime(arr[-1]["last_updated"]) if arr else None
        except Exception:
            st = en = None
        return arr, st, en

    async def get_state(self, entity_id: str, default: Any = None, attribute: Optional[str] = None):
        item = await self.api_call("GET", f"/api/states/{entity_id}")
        if not item:
            return default
        if attribute:
            return item.get("attributes", {}).get(attribute, default)
        return item.get("state", default)

    async def set_state(self, entity_id: str, state: Any, attributes: Optional[dict] = None):
        data = {"state": str(state)}
        if attributes:
            data["attributes"] = attributes
        await self.api_call("POST", f"/api/states/{entity_id}", json_data=data)


# --------------------------------------------------------------------------- #
# SQLite history cache
# --------------------------------------------------------------------------- #

class HistoryDB:
    def __init__(self, path: str = DEFAULT_DB_PATH):
        self.path = path
        self.con = sqlite3.connect(self.path)
        self.cur = self.con.cursor()

    def safe_name(self, name: str) -> str:
        t = name.replace(".", "_")
        if not SAFE_TBL_RE.match(t):
            raise ValueError(f"Unsafe table name: {name}")
        return t

    def create_table(self, table: str):
        t = self.safe_name(table)
        self.cur.execute(f"CREATE TABLE IF NOT EXISTS {t} (timestamp TEXT PRIMARY KEY, value REAL)")
        self.con.commit()

    def cleanup_table(self, table: str, oldest_dt: datetime):
        t = self.safe_name(table)
        oldest_stamp = oldest_dt.strftime("%Y-%m-%d %H:%M:%S%z")
        self.cur.execute(f"DELETE FROM {t} WHERE timestamp < ?", (oldest_stamp,))
        self.con.commit()

    def get_history(self, table: str) -> pd.DataFrame:
        t = self.safe_name(table)
        # Ensure the table exists so the SELECT does not fail on first run
        self.create_table(t)
        self.cur.execute(f"SELECT * FROM {t} ORDER BY timestamp")
        rows = self.cur.fetchall()
        if not rows:
            return pd.DataFrame(columns=["ds", "y"])
        df = pd.DataFrame(rows, columns=["ds", "y"])
        df["ds"] = pd.to_datetime(df["ds"], utc=True, errors="coerce")
        return df.dropna(subset=["ds"])

    def store_history(self, table: str, history: pd.DataFrame, prev: pd.DataFrame) -> pd.DataFrame:
        t = self.safe_name(table)
        self.create_table(t)
        # Normalise previously stored timestamps to the same ISO format that we
        # use when inserting new rows. `str(dt) would produce a space between
        # date and time ("YYYY-MM-DD HH:MM:SS+00:00"), whereas `isoformat()
        # yields "YYYY-MM-DDTHH:MM:SS+00:00".  The mismatch allowed duplicates to
        # slip past the "timestamp_s not in prev_values" check and triggered
        # SQLite UNIQUE constraint errors.  Build the set using `isoformat so
        # comparisons are consistent.
        prev_values = set(dt.isoformat() for dt in prev["ds"] if pd.notna(dt))
        added = 0
        for _, row in history.iterrows():
            timestamp = pd.to_datetime(row["ds"], utc=True, errors="coerce")
            if pd.isna(timestamp):
                continue
            timestamp_s = timestamp.isoformat()
            value = float(row["y"])
            if timestamp_s not in prev_values:
                self.cur.execute(
                    f"INSERT OR IGNORE INTO {t} (timestamp, value) VALUES (?, ?)",
                    (timestamp_s, value),
                )
                if self.cur.rowcount:
                    prev_values.add(timestamp_s)
                    prev.loc[len(prev)] = {"ds": timestamp, "y": value}
                    added += 1
        self.con.commit()
        logger.info("DB: added %s rows to %s", added, t)
        return prev

    def close(self):
        self.con.close()


# --------------------------------------------------------------------------- #
# Transform utilities
# --------------------------------------------------------------------------- #

def normalise_history(raw: List[dict]) -> pd.DataFrame:
    logger.debug("Normalising history with %s raw rows", len(raw))
    if not raw:
        return pd.DataFrame(columns=["ds", "value"])
    df = pd.DataFrame(raw)
    df["ds"] = pd.to_datetime(df["last_updated"], utc=True, errors="coerce")
    df["value"] = pd.to_numeric(df["state"], errors="coerce")
    df = df.dropna(subset=["ds", "value"]).sort_values("ds")
    logger.debug("Normalised history result rows=%s", len(df))
    return df[["ds", "value"]]


def resample_sensor(df: pd.DataFrame, freq: str, how: str) -> pd.DataFrame:
    if df.empty:
        return df
    logger.debug(
        "Resampling %s rows to freq=%s how=%s", len(df), freq, how
    )
    df = df.set_index("ds").sort_index()
    if how == "sum":
        agg = df["value"].resample(freq).sum(min_count=1)
    elif how == "last":
        agg = df["value"].resample(freq).last()
    else:  # mean default
        agg = df["value"].resample(freq).mean()
    out = agg.to_frame("value").reset_index()
    out = out.dropna(subset=["value"])
    logger.debug("Resampled result rows=%s", len(out))
    return out


def cumulative_to_interval(df: pd.DataFrame, reset_cfg: ResetDetectionCfg, max_increment: Optional[float] = None) -> pd.DataFrame:
    if df.empty:
        df["y"] = []
        return df
    logger.debug("Cumulative->interval on %s rows", len(df))
    df = df.sort_values("ds").reset_index(drop=True)
    v = df["value"].to_numpy()
    delta = np.diff(v, prepend=v[0])
    delta[0] = np.nan
    neg_mask = delta < 0
    reset_count = 0
    if reset_cfg.enabled:
        if reset_cfg.hard_reset_value is not None:
            reset_mask = np.isclose(v, reset_cfg.hard_reset_value)
            reset_count = int(np.sum(reset_mask))
            for i in np.where(reset_mask)[0]:
                delta[i] = v[i]
    delta[neg_mask] = np.nan
    spike_count = 0
    if max_increment is not None:
        spike_mask = np.abs(delta) > max_increment
        spike_count = int(np.sum(spike_mask))
        delta[spike_mask] = np.nan
    delta = np.nan_to_num(delta, nan=0.0)
    delta = np.clip(delta, 0.0, None)
    df["y"] = delta
    logger.debug(
        "Cumulative->interval result rows=%s neg=%s resets=%s spikes=%s",
        len(df),
        int(neg_mask.sum()),
        reset_count,
        spike_count,
    )
    return df


def apply_log_transform(df: pd.DataFrame) -> pd.DataFrame:
    logger.debug("Applying log transform to %s rows", len(df))
    df = df.copy()
    df["y"] = np.log1p(df["y"].clip(lower=0))
    df.attrs["log_transform_applied"] = True
    logger.debug("Log transform complete")
    return df


def invert_log_transform(arr: np.ndarray, applied: bool) -> np.ndarray:
    if not applied:
        return arr
    return np.expm1(arr)


def subtract_set(base: pd.DataFrame, sub: pd.DataFrame, *, inc: bool = False) -> pd.DataFrame:
    """Subtract one dataset from another by timestamp."""
    if base.empty:
        return base
    merged = base.merge(sub, on="ds", how="left", suffixes=("", "_sub"))
    merged["y_sub"].fillna(0, inplace=True)
    merged["y"] = merged.apply(
        lambda row: max(row["y"] - row["y_sub"], 0) if inc else row["y"] - row["y_sub"],
        axis=1,
    )
    return merged[["ds", "y"]]


# --------------------------------------------------------------------------- #
# CovariateResolver
# --------------------------------------------------------------------------- #

class CovariateResolver:
    """
    Basic covariate resolution with alias & scale support.

    Example YAML:
      covariates:
        mixergy_charge:
          entity: sensor.current_charge
          scale: 0.01
        outdoor_temp_forecast: sensor.external_temperature
    """

    def __init__(self, iface: HAInterface, cov_map: Dict[str, Any]):
        self.iface = iface
        self.map = cov_map

    def _resolve(self, name: str) -> dict:
        val = self.map.get(name, name)
        if isinstance(val, str):
            return {"entity": val, "scale": 1.0}
        if isinstance(val, dict):
            return {
                "entity": val.get("entity", name),
                "scale": val.get("scale", 1.0),
                "attr": val.get("attr"),
                "forecast_attr": val.get("forecast_attr"),
                "units": val.get("units"),
            }
        return {"entity": str(val), "scale": 1.0}

    async def get_hist_series(self, cov_name: str, start: datetime, end: datetime, freq: str, how: str) -> pd.Series:
        meta = self._resolve(cov_name)
        entity_id = meta["entity"]
        logger.debug(
            "Covariate %s: fetching history for %s from %s to %s",
            cov_name,
            entity_id,
            start,
            end,
        )
        raw, _, _ = await self.iface.get_history(entity_id, start, end)
        df = normalise_history(raw)
        logger.debug(
            "Covariate %s: normalised history rows=%s", cov_name, len(df)
        )
        if df.empty:
            return pd.Series([], dtype=float)
        # never sum temps/% etc.; if how=='sum' use mean
        df = resample_sensor(df, freq, "mean" if how == "sum" else how)
        logger.debug(
            "Covariate %s: resampled rows=%s", cov_name, len(df)
        )
        df["value"] = df["value"] * meta.get("scale", 1.0)
        logger.debug(
            "Covariate %s: scaled by %s", cov_name, meta.get("scale", 1.0)
        )
        out = df.set_index("ds")["value"]
        logger.info(
            "Covariate %s: min=%s max=%s %s",
            cov_name,
            out.min(),
            out.max(),
            meta.get("units", ""),
        )
        logger.debug(
            "Covariate %s: obtained %s rows", cov_name, len(out)
        )
        logger.debug(
            "Covariate %s: head=%s", cov_name, out.head().to_dict()
        )
        return out

    async def get_future_series(self, cov_name: str, future_index: pd.DatetimeIndex, default: float = 0.0) -> pd.Series:
        meta = self._resolve(cov_name)
        entity_id = meta["entity"]
        val = await self.iface.get_state(entity_id)
        try:
            v = float(val) * meta.get("scale", 1.0)
        except (TypeError, ValueError):
            v = default
        logger.debug(
            "Covariate %s: future value %s from %s", cov_name, v, entity_id
        )
        logger.debug(
            "Covariate %s: future series len=%s", cov_name, len(future_index)
        )
        logger.info(
            "Covariate %s future: min=%s max=%s %s",
            cov_name,
            v,
            v,
            meta.get("units", ""),
        )
        return pd.Series(v, index=future_index)


# --------------------------------------------------------------------------- #
# Model Backend (NeuralProphet)
# --------------------------------------------------------------------------- #

class NPBackend:
    def __init__(self,
                 n_lags: int,
                 n_forecasts: int,
                 seasonality_reg: float,
                 seasonality_mode: str = "additive",
                 learning_rate: Optional[float] = None,
                 country: Optional[str] = None):
        if NeuralProphet is None:
            raise RuntimeError(f"NeuralProphet import failed: {_NP_IMPORT_ERROR}")
        kw = dict(
            n_lags=n_lags,
            n_forecasts=n_forecasts,
            seasonality_mode=seasonality_mode,
            seasonality_reg=seasonality_reg,
        )
        if learning_rate is not None:
            kw["learning_rate"] = learning_rate
        self.model = NeuralProphet(**kw, drop_missing=True)
        if country:
            self.model.add_country_holidays(country)
        self.fitted = False

    def add_future_regressor(self, name: str, mode: str = "additive"):
        self.model.add_future_regressor(name, mode=mode)

    def add_lagged_regressor(self, name: str, n_lags: Optional[int] = None):
        self.model.add_lagged_regressor(name, n_lags=n_lags)

    def fit(self, df: pd.DataFrame, freq: str):
        self.model.fit(df, freq=freq, progress=None)
        self.fitted = True

    def make_future(self, df: pd.DataFrame, periods: int) -> pd.DataFrame:
        return self.model.make_future_dataframe(df, n_historic_predictions=True, periods=periods)

    def predict(self, df_future: pd.DataFrame) -> pd.DataFrame:
        return self.model.predict(df_future)


# --------------------------------------------------------------------------- #
# Horizon helpers & publishing
# --------------------------------------------------------------------------- #

def horizon_steps(minutes_ahead: int, interval_min: int) -> int:
    return max(1, minutes_ahead // interval_min)


def horizon_agg(yhat_interval: Sequence[float], interval_min: int, minutes_ahead: int) -> float:
    steps = min(horizon_steps(minutes_ahead, interval_min), len(yhat_interval))
    return float(np.nansum(yhat_interval[:steps]))


def make_entity_name(prefix: str, base: str, suffix: Optional[str] = None) -> str:
    """Return a valid Home Assistant `entity_id for publishing state."""
    prefix = re.sub(r"^sensor[._]", "", prefix, flags=re.IGNORECASE)
    prefix = re.sub(r"[^a-z0-9_]+", "_", prefix.lower())
    base = re.sub(r"[^a-z0-9_]+", "_", base.lower())
    parts = [prefix + base]
    if suffix:
        parts.append(str(suffix))
    object_id = "_".join(parts)
    object_id = re.sub(r"_+", "_", object_id).strip("_")
    return f"sensor.{object_id}"


def dict_from_series(index: Sequence[datetime], values: Sequence[float], tz: timezone) -> Dict[str, float]:
    return {
        ensure_utc(ts).astimezone(tz).strftime(TIME_FORMAT_HA): round(float(v), 3)
        for ts, v in zip(index, values)
    }


def daily_cumulative_series(index: Sequence[datetime], values: Sequence[float], tz: timezone) -> Dict[str, float]:
    cum = 0.0
    out: Dict[str, float] = {}
    current_day = None
    for ts, v in zip(index, values):
        lts = ensure_utc(ts).astimezone(tz)
        if current_day != lts.date():
            cum = 0.0
            current_day = lts.date()
        cum += max(float(v), 0.0)
        out[lts.strftime(TIME_FORMAT_HA)] = round(cum, 3)
    return out

def energy_already_used_today(df_cum: pd.DataFrame, tz: timezone) -> float:
    """Return the cumulative kWh that have been consumed *today*."""
    if df_cum.empty:
        return 0.0
    today = datetime.now(tz).date()
    today_rows = df_cum[df_cum["ds"].dt.date == today]
    if today_rows.empty:
        return 0.0
    return float(today_rows["value"].iloc[-1])


async def publish_forecasts(sensor: SensorCfg,
                            role_cfg: RoleCfg,
                            iface: HAInterface,
                            cfg: PredAIConfig,
                            ds_future: Sequence[datetime],
                            yhat_interval: Sequence[float],
                            yhat_level: Optional[Sequence[float]] = None,
                            metrics: Optional[dict] = None,
                            sensor_hist_cum: Optional[pd.DataFrame] = None):
    logger.info("Publishing forecasts for %s", sensor.name)
    tz = cfg.tz
    interval_min = sensor.interval or cfg.common_interval
    hist_df   = sensor_hist_cum if sensor_hist_cum is not None else pd.DataFrame()
    used_today = energy_already_used_today(hist_df, tz)
    prefix    = cfg.publish_prefix
    base_name = sensor.publish_name or sensor.name

    # ── Clip any forecast buckets that lie in the past ─────────────────────
    now_local = datetime.now(tz)
    for i, ts in enumerate(ds_future):
        if ensure_utc(ts).astimezone(tz) < now_local:
            yhat_interval[i] = 0.0
    # ───────────────────────────────────────────────────────────────────────

    yhat_interval = np.array(yhat_interval, dtype=float)
    yhat_interval = np.nan_to_num(yhat_interval, nan=0.0, posinf=0.0, neginf=0.0)
    yhat_interval = np.clip(yhat_interval, 0, None)  # no negatives

    if (sensor.units or "").lower() == "wh" and (publish_units or "").lower().endswith("kwh"):
        yhat_interval = yhat_interval / 1000.0

    if yhat_level is not None:
        yhat_level = np.array(yhat_level, dtype=float)
        yhat_level = np.nan_to_num(yhat_level, nan=0.0, posinf=0.0, neginf=0.0)

    # ------------------------------------------------------------------
    # *counter* sensors must continue from the last real reading
    # ------------------------------------------------------------------
    baseline = 0.0
    if sensor.source_is_cumulative:
        try:
            baseline = float(await iface.get_state(sensor.name, default=0.0))
        except Exception:
            baseline = 0.0

        # If the meter resets at midnight, start the forecast curve at the
        # energy already used today so orange & blue meet at "now".
        if sensor.reset_daily:
            baseline = used_today

    cum_from_now = baseline + np.cumsum(yhat_interval)
    if sensor.reset_daily and sensor.source_is_cumulative:
        midnight = datetime.now(tz).replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        for i, ts in enumerate(ds_future):
            if ensure_utc(ts).astimezone(tz) >= midnight:
                reset_val = cum_from_now[i-1] if i > 0 else baseline
                cum_from_now[i:] -= reset_val
                break
    # - Daily cumulative forecast starting from the energy already used today so
    #   the curve meets the live meter reading at 'now'.
    # Re-build daily cumulative after clipping & baseline offset
    daily_cum = daily_cumulative_series(ds_future, yhat_interval, tz)
    today_str = now_local.strftime("%Y-%m-%d")
    for ts_iso in list(daily_cum.keys()):
        if ts_iso.startswith(today_str):
            daily_cum[ts_iso] += used_today

    ser_interval = dict_from_series(ds_future, yhat_interval, tz)
    ser_cum = dict_from_series(ds_future, cum_from_now, tz)

    model_ts_iso = datetime.now(timezone.utc).astimezone(tz).isoformat()
    meta = {
        "model_ts": model_ts_iso,
        "model_backend": role_cfg.model_backend,
        "training_rows": metrics.get("training_rows") if metrics else None,
        "mae_recent": metrics.get("mae_recent") if metrics else None,
    }

    publish_units = sensor.output_units or sensor.units
    state_class = ("total_increasing" if sensor.source_is_cumulative else role_cfg.publish_state_class)

    if sensor.publish_interval:
        ent_interval = make_entity_name(prefix, base_name, "interval")
        await iface.set_state(
            ent_interval,
            state=round(float(yhat_interval[0]) if len(yhat_interval) else 0.0, 3),
            attributes={
                "unit_of_measurement": publish_units,
                "state_class": "measurement",
                "forecast_series": ser_interval,
                **meta,
            },
        )

    if sensor.publish_cumulative:
        ent_cum = make_entity_name(prefix, base_name, "cum")
        await iface.set_state(
            ent_cum,
            state=round(float(cum_from_now[-1]) if len(cum_from_now) else 0.0, 3),
            attributes={
                "unit_of_measurement": publish_units,
                "state_class": state_class,
                "forecast_series": ser_cum,
                **meta,
            },
        )

    if sensor.publish_daily_cumulative:
        ent_daily = make_entity_name(prefix, base_name, "daily_cum")
        today_str = datetime.now(tz).strftime("%Y-%m-%d")
        todays = {k: v for k, v in daily_cum.items() if k.startswith(today_str)}
        state_val = list(todays.values())[-1] if todays else list(daily_cum.values())[-1]
        await iface.set_state(
            ent_daily,
            state=round(float(state_val), 3),
            attributes={
                "unit_of_measurement": publish_units,
                "state_class": state_class,
                "forecast_series": daily_cum,
                **meta,
            },
        )

        # --------------------------------------------------------------
        #  Preserve the previous forecast curve → “…_curve_yesterday”
        # --------------------------------------------------------------
        ent_curve      = make_entity_name(prefix, base_name, "pred_curve")
        ent_curve_prev = make_entity_name(prefix, base_name, "curve_yesterday")

        old_item = await iface.api_call("GET", f"/api/states/{ent_curve}")
        if old_item:
            await iface.set_state(
                ent_curve_prev,
                state=old_item.get("state", 0),
                attributes=old_item.get("attributes", {}),
            )

        await iface.set_state(
            ent_curve,
            state=round(list(daily_cum.values())[-1], 3),
            attributes={
                "unit_of_measurement": publish_units,
                "state_class": state_class,
                "forecast_series": daily_cum,
                **meta,
            },
        )

        # --------------------------------------------------------------
        #  One-shot "initial" curve -> "…_curve_initial"
        # --------------------------------------------------------------
        def _same_day(ts1: str, ts2: str) -> bool:
            try:
                d1 = timestr_to_datetime(ts1).astimezone(tz).date()
                d2 = timestr_to_datetime(ts2).astimezone(tz).date()
                return d1 == d2
            except Exception:
                return False

        ent_curve_init = make_entity_name(prefix, base_name, "pred_curve_initial")
        existing = await iface.api_call("GET", f"/api/states/{ent_curve_init}")
        need_update = True
        if existing and "attributes" in existing:
            old_ts = existing["attributes"].get("model_ts")
            need_update = not _same_day(old_ts, model_ts_iso)

        if need_update:
            await iface.set_state(
                ent_curve_init,
                state=round(list(daily_cum.values())[-1], 3),
                attributes={
                    "unit_of_measurement": publish_units,
                    "state_class": state_class,
                    "forecast_series": daily_cum,
                    "generated_from": make_entity_name(prefix, base_name, "interval"),
                    **meta,
                },
            )

        # --------------------------------------------------------------
        #  Rolling 7‑day buffer of initial curves  → “…_pred_curve_initial_7d”
        # --------------------------------------------------------------
        ent_curve_hist = make_entity_name(prefix, base_name, "pred_curve_initial_7d")
        today_key = now_local.strftime("%Y-%m-%d")

        # 1.  Load existing map (if any)
        prev_item = await iface.api_call("GET", f"/api/states/{ent_curve_hist}")
        hist_map: Dict[str, Any] = {}
        if prev_item and "attributes" in prev_item:
            hist_map = prev_item["attributes"].get("forecast_series_map", {}) or {}

        # 2.  Prune to the most‑recent six previous days
        cutoff = now_local.date() - timedelta(days=6)
        hist_map = {
            d: s for d, s in hist_map.items()
            if datetime.fromisoformat(d).date() >= cutoff
        }

        # 3.  Add today’s curve only if not already present
        if today_key not in hist_map:
            hist_map[today_key] = daily_cum

        # 4.  Publish / overwrite the rolling sensor
        await iface.set_state(
            ent_curve_hist,
            state=round(list(daily_cum.values())[-1], 3),
            attributes={
                "unit_of_measurement": publish_units,
                "state_class": state_class,
                "forecast_series_map": hist_map,
                **meta,
            },
        )

    # Horizon scalars
    for m in cfg.horizons:
        suffix = f"pred_{m//60}h"
        ent_h = make_entity_name(prefix, base_name, suffix)
        if sensor.train_target == "level":  # e.g., temperature
            arr = np.array(yhat_level if yhat_level is not None else yhat_interval)
            steps = min(horizon_steps(m, interval_min), len(arr))
            val = arr[steps - 1]
        else:
            val = horizon_agg(yhat_interval, interval_min, m)
        await iface.set_state(
            ent_h,
            state=round(float(val), 3),
            attributes={
                "unit_of_measurement": publish_units,
                "state_class": "measurement",
                "generated_from": make_entity_name(prefix, base_name, "interval"),
                **meta,
            },
        )

    logger.info("Finished publishing forecasts for %s", sensor.name)


# --------------------------------------------------------------------------- #
# Sensor job execution
# --------------------------------------------------------------------------- #

async def run_sensor_job(sensor: SensorCfg,
                         role_cfg: RoleCfg,
                         cfg: PredAIConfig,
                         iface: HAInterface,
                         cov_res: CovariateResolver,
                         db: Optional[HistoryDB]) -> None:
    interval_min = sensor.interval or cfg.common_interval
    freq = f"{interval_min}min"
    tz = cfg.tz

    # Round now to interval boundary
    now = datetime.now(timezone.utc).astimezone(tz).replace(second=0, microsecond=0)
    minute_floor = (now.minute // interval_min) * interval_min
    now = now.replace(minute=minute_floor)

    start_hist = now - timedelta(days=sensor.days_hist)
    end_hist = now

    logger.info("Sensor %s: fetching history %s → %s", sensor.name, start_hist, end_hist)
    raw_hist, st, en = await iface.get_history(sensor.name, start_hist, end_hist)
    df = normalise_history(raw_hist)
    logger.info(
        "Sensor %s: history rows=%s first=%s last=%s",
        sensor.name,
        len(df),
        df["ds"].min() if not df.empty else None,
        df["ds"].max() if not df.empty else None,
    )
    if not df.empty:
        logger.info(
            "Sensor %s: min=%s max=%s %s",
            sensor.name,
            df["value"].min(),
            df["value"].max(),
            sensor.units or "",
        )

    # DB merge
    if sensor.database and db:
        tname = sensor.name.replace(".", "_")
        prev = db.get_history(tname)
        if not df.empty:
            tmp = df.rename(columns={"value": "y"})[["ds", "y"]]
            prev = db.store_history(tname, tmp, prev)
        oldest = now - timedelta(days=sensor.max_age)
        db.cleanup_table(tname, oldest)
        if not prev.empty:
            prev = prev.sort_values("ds")
            prev = prev.rename(columns={"y": "value"})
            df = prev
        logger.info(
            "Sensor %s: dataset after DB merge rows=%s", sensor.name, len(df)
        )

    if df.empty:
        logger.warning("Sensor %s: no data; skipping.", sensor.name)
        return

    df_cum_raw = df.copy()

    # --------------------------------------------------
    # 1.  Convert cumulative counter → interval
    # --------------------------------------------------
    if sensor.source_is_cumulative:
        df = cumulative_to_interval(df, sensor.reset_detection, sensor.max_increment)
        # resampler needs a column called 'value', so replace it
        df["value"] = df["y"]
    
    # --------------------------------------------------
    # 2.  Resample the interval series (sum / mean / last)
    # --------------------------------------------------
    agg = sensor.effective_aggregation(role_cfg)      # keep "sum" for energy
    df = resample_sensor(df, freq, agg)
    logger.info(
        "Sensor %s: after resample %s rows from %s", sensor.name, len(df), freq
    )
    
    # --------------------------------------------------
    # 3.  Rename value → y (now exists for *all* sensors)
    # --------------------------------------------------
    df = df.rename(columns={"value": "y"})

    if sensor.subtract:
        for sub_name in sensor.subtract:
            logger.info("Sensor %s: subtracting %s", sensor.name, sub_name)
            raw_sub, _, _ = await iface.get_history(sub_name, start_hist, end_hist)
            sub_df = normalise_history(raw_sub)
            if sensor.source_is_cumulative or sensor.incrementing:
                sub_df = cumulative_to_interval(sub_df, sensor.reset_detection, sensor.max_increment)
                sub_df["value"] = sub_df["y"]
            sub_df = resample_sensor(sub_df, freq, agg)
            sub_df = sub_df.rename(columns={"value": "y"})
            sub_df["y"] = pd.to_numeric(sub_df["y"], errors="coerce").fillna(0.0)
            df = subtract_set(df, sub_df, inc=sensor.incrementing or sensor.source_is_cumulative)

    # Power->energy (heuristic)
    if (not sensor.source_is_cumulative) and sensor.train_target == "interval":
        units_lower = (sensor.units or "").lower()
        if "w" in units_lower:  # W or kW
            if df["y"].max() > 50:  # assume W
                df["y"] = df["y"] / 1000.0
            df["y"] = df["y"] * (interval_min / 60.0)  # kWh per bucket
            if not sensor.output_units:
                sensor.output_units = "kWh"

    # Clean
    df["y"] = pd.to_numeric(df["y"], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["y"])
    logger.debug(
        "Sensor %s: cleaned data rows=%s", sensor.name, len(df)
    )

    log_applied = False
    if sensor.log_transform:
        df = apply_log_transform(df)
        log_applied = True
        logger.debug("Sensor %s: applied log transform", sensor.name)

    if len(df) < role_cfg.n_lags + 5:
        logger.warning("Sensor %s: insufficient history (%s rows); skipping model.", sensor.name, len(df))
        return

    # Training frame
    train_df = df[["ds", "y"]].copy()
    logger.info(
        "Sensor %s: training frame %s rows", sensor.name, len(train_df)
    )
    if not train_df.empty:
        logger.info(
            "Sensor %s: training min=%s max=%s %s",
            sensor.name,
            train_df["y"].min(),
            train_df["y"].max(),
            sensor.output_units or sensor.units or "",
        )

    if role_cfg.model_backend == "neuralprophet":
        steps = sensor.future_periods if sensor.future_periods is not None else max(
            horizon_steps(m, interval_min) for m in cfg.horizons
        )
        backend = NPBackend(
            n_lags=sensor.effective_n_lags(role_cfg),
            n_forecasts=steps,
            seasonality_reg=sensor.effective_seasonality_reg(role_cfg),
            seasonality_mode=sensor.effective_seasonality_mode(role_cfg),
            learning_rate=sensor.effective_learning_rate(role_cfg),
            country=sensor.country,
        )

        # lagged covariates
        for cov in sensor.covariates_lagged:
            s = await cov_res.get_hist_series(cov, start_hist, end_hist, freq, agg)
            if not s.empty:
                train_df = train_df.merge(s.rename(cov), left_on="ds", right_index=True, how="left")
                logger.debug(
                    "Covariate %s lagged: merged %s rows", cov, len(s)
                )
                backend.add_lagged_regressor(cov, n_lags=sensor.effective_n_lags(role_cfg))
            else:
                logger.debug("Covariate %s lagged: no history.", cov)

        # future covariates
        for cov in sensor.covariates_future:
            s = await cov_res.get_hist_series(cov, start_hist, end_hist, freq, agg)
            if not s.empty:
                train_df = train_df.merge(s.rename(cov), left_on="ds", right_index=True, how="left")
                logger.debug(
                    "Covariate %s future: merged %s rows", cov, len(s)
                )
            else:
                train_df[cov] = np.nan
                logger.debug("Covariate %s future: no history, filled NaN", cov)
            backend.add_future_regressor(cov, mode="additive")

        # ensure no NaN values at the end of the training data
        cov_cols = sensor.covariates_lagged + sensor.covariates_future
        if cov_cols:
            train_df[cov_cols] = train_df[cov_cols].fillna(method="ffill")
            train_df[cov_cols] = train_df[cov_cols].fillna(0.0)

        # Fit
        backend.fit(train_df, freq=freq)
        logger.info("Sensor %s: model trained", sensor.name)

        # ------------------------------------------------------------------
        # Future frame + prediction  ✱ handles future‑regressor check
        # ------------------------------------------------------------------
        steps = sensor.future_periods if sensor.future_periods is not None else max(
            horizon_steps(m, interval_min) for m in cfg.horizons
        )

        last_ts = train_df["ds"].max()

        # 1.  Build placeholder rows for the forecast horizon so NP sees the
        #     future‑regressor columns *before* it validates the DataFrame.
        fut_idx = [last_ts + timedelta(minutes=interval_min * i)
                   for i in range(1, steps + 1)]
        extra_rows = pd.DataFrame({"ds": fut_idx})

        #    • future regressors → provisional 0.0
        for cov in sensor.covariates_future:
            extra_rows[cov] = 0.0

        #    • lagged regressors → repeat last observed value
        for cov in sensor.covariates_lagged:
            last_val = train_df[cov].iloc[-1] if cov in train_df.columns else 0.0
            extra_rows[cov] = last_val if pd.notna(last_val) else 0.0

        #    • target column must exist (value is ignored during prediction)
        extra_rows["y"] = train_df["y"].iloc[-1]

        # 2.  Concatenate history + placeholder future rows.
        df_make_future = pd.concat([train_df, extra_rows], ignore_index=True)

        # 3.  Tell NP *not* to append additional rows (periods=0) because we
        #     have already supplied them.
        df_future = backend.make_future(df_make_future, periods=0)
        df_future["ds"] = pd.to_datetime(df_future["ds"], utc=True)

        # 4.  Overwrite placeholder future‑regressor values with real ones.
        if sensor.covariates_future:
            fut_mask = df_future["ds"] > last_ts
            fut_idx = df_future.loc[fut_mask, "ds"]
            for cov in sensor.covariates_future:
                fut_series = await cov_res.get_future_series(cov, fut_idx, default=0.0)
                df_future.loc[fut_mask, cov] = fut_series.to_numpy()

        # 5.  Predict.
        fcst = backend.predict(df_future)
        fcst["ds"] = pd.to_datetime(fcst["ds"], utc=True)

        # 6.  Take the first *true‑future* row and collect yhat₁ … yhatₙ.
        # Extract the forecast vector from the *last* historic row
        row_mask = fcst["ds"] == last_ts
        if not row_mask.any():              # fallback: last row of frame
            row_mask = fcst.index == (len(fcst) - 1)

        last_row = fcst.loc[row_mask].iloc[0]

        yhat_cols = sorted(
            [c for c in last_row.index if c.startswith("yhat")],
            key=lambda s: int(s[4:]),
        )
        yhat_int = last_row[yhat_cols].to_numpy()

        if log_applied:
            yhat_int = invert_log_transform(yhat_int, True)

        ds_future = [last_ts + timedelta(minutes=interval_min * i)
                     for i in range(1, steps + 1)]

        metrics = {"training_rows": int(len(train_df)), "mae_recent": None}
        await publish_forecasts(sensor, role_cfg, iface, cfg, ds_future, yhat_int,
                                metrics=metrics, sensor_hist_cum=df_cum_raw)
        logger.info("Sensor %s: forecasting complete", sensor.name)

    else:
        logger.error("Unsupported backend %s for sensor %s", role_cfg.model_backend, sensor.name)


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #

async def predai_main():
    cfg = load_config(DEFAULT_CONFIG_PATH)

    # read raw for HA creds & initial cov_map
    try:
        with open(DEFAULT_CONFIG_PATH, "r") as f:
            raw = yaml.safe_load(f) or {}
    except Exception:
        raw = {}
    ha_url = raw.get("ha_url")
    ha_key = raw.get("ha_key") or os.environ.get("SUPERVISOR_TOKEN")

    iface = HAInterface(ha_url, ha_key)
    cov_res = CovariateResolver(iface, cfg.cov_map)
    db = HistoryDB(DEFAULT_DB_PATH)

    try:
        while True:
            logger.info("PredAI cycle start.")
            # hot-reload config
            cfg = load_config(DEFAULT_CONFIG_PATH)
            # refresh covariate map
            cov_res.map = cfg.cov_map

            for s in cfg.sensors:
                role_cfg = cfg.roles.get(s.role, RoleCfg())
                try:
                    await run_sensor_job(s, role_cfg, cfg, iface, cov_res, db)
                except Exception as e:
                    logger.exception("Sensor job failed for %s: %s", s.name, e)

            now_str = datetime.now(timezone.utc).isoformat()
            await iface.set_state(
                "sensor.predai_last_run",
                state=now_str,
                attributes={"unit_of_measurement": "time"},
            )

            logger.info("PredAI sleeping %s minutes.", cfg.update_every)
            # Sleep minute chunks; break early if heartbeat lost. Output a
            # progress log each minute so hosting platforms don't kill the
            # process for lack of output. Log once before the first sleep so
            # there is never a full minute with no output.
            logger.info(
                "PredAI sleep progress: 0/%s minutes",
                cfg.update_every,
            )
            for i in range(cfg.update_every):
                await asyncio.sleep(60)
                last_run = await iface.get_state("sensor.predai_last_run")
                if last_run is None:
                    logger.warning("PredAI heartbeat lost; restarting early.")
                    break
                logger.info(
                    "PredAI sleep progress: %s/%s minutes",
                    i + 1,
                    cfg.update_every,
                )

    finally:
        await iface.close()
        db.close()


def main():
    asyncio.run(predai_main())


if __name__ == "__main__":
    main()
