"""业务层：登记、读数、核平排队、复核与并发守卫（SQLite，仅标准库）。

并发与留存语义
--------------
- meta.source_version 单调递增：任何登记表/更新读数都在同一事务内 +1。
- 每次核平在 runs 表排队留存：原始读数快照 + 校正依据（权重/限值/锁定/结构/版本）。
- 源数据更新后，未复核（completed）且版本落后的结果即失效（stale），页面标注。
- 复核通过是单条原子 UPDATE：仅当状态为 completed 且 source_version 等于当前
  版本才生效；复核与更新并发时，版本已变则拒绝过时通过（StaleRunError）。
- 无可行解：记录冲突约束，不生成可复核报告（result 为空，不可通过）。
- 超时/迭代超限：记录 failed，不产出任何近似结果。
"""
from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timezone

from .solver import (IterationLimitExceeded, MeterNode, SolverError,
                     SolverTimeout, solve)


class ValidationError(Exception):
    """输入校验失败（可恢复）。"""


class NotFoundError(Exception):
    """对象不存在。"""


class StaleRunError(Exception):
    """源数据已更新，结果失效，拒绝过时通过。"""


class ReviewStateError(Exception):
    """当前状态不可复核（已复核/无可行解/失败）。"""


SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS meters (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL UNIQUE,
  parent_id INTEGER REFERENCES meters(id),
  weight REAL NOT NULL,
  lower REAL,
  upper REAL,
  locked REAL,
  factor REAL NOT NULL DEFAULT 1.0,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS readings (
  meter_id INTEGER PRIMARY KEY REFERENCES meters(id),
  value REAL NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  source_version INTEGER NOT NULL,
  status TEXT NOT NULL,              -- completed/infeasible/failed/approved/rejected
  message TEXT,
  objective REAL,
  leaf_total REAL,
  max_residual REAL,
  iterations INTEGER,
  snapshot_json TEXT NOT NULL,       -- 原始读数 + 校正依据（权重/限值/锁定/版本）
  result_json TEXT,                  -- 校正值/调整量/守恒残差（infeasible/failed 为空）
  conflicts_json TEXT,               -- 无可行解时的冲突约束清单
  created_at TEXT NOT NULL,
  finished_at TEXT,
  reviewed_at TEXT
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _float_or_none(v, label: str):
    if v is None or v == "":
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise ValidationError(f"{label}须为数值")
    if not math.isfinite(f):
        raise ValidationError(f"{label}须为有限数值")
    return f


class ReconService:
    def __init__(self, db_path: str):
        self.db_path = str(db_path)
        with self._conn() as conn:
            conn.executescript(SCHEMA)
            conn.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES('source_version', '0')")

    # ---------------------------------------------------------------- 基础
    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _version(self, conn: sqlite3.Connection) -> int:
        return int(conn.execute(
            "SELECT value FROM meta WHERE key='source_version'").fetchone()[0])

    def _bump(self, conn: sqlite3.Connection) -> None:
        conn.execute("UPDATE meta SET value=CAST(value AS INTEGER)+1 "
                     "WHERE key='source_version'")

    def current_version(self) -> int:
        with self._conn() as conn:
            return self._version(conn)

    # ---------------------------------------------------------------- 登记
    def add_meter(self, name, parent_id=None, weight=1.0, lower=None,
                  upper=None, locked=None, factor=1.0) -> int:
        name = (name or "").strip()
        if not name:
            raise ValidationError("表名称不能为空")
        try:
            weight = float(weight)
            factor = float(factor)
        except (TypeError, ValueError):
            raise ValidationError("权重/排放因子须为数值")
        if not math.isfinite(weight) or weight <= 0:
            raise ValidationError("权重必须为正数")
        if not math.isfinite(factor):
            raise ValidationError("排放因子须为有限数值")
        lo = _float_or_none(lower, "下限")
        hi = _float_or_none(upper, "上限")
        if lo is not None and hi is not None and lo > hi:
            raise ValidationError("下限不能大于上限")
        lk = _float_or_none(locked, "锁定值")
        if lk is not None:
            if lo is not None and lk < lo:
                raise ValidationError("锁定值低于下限")
            if hi is not None and lk > hi:
                raise ValidationError("锁定值高于上限")
        with self._conn() as conn:
            if parent_id is not None:
                if conn.execute("SELECT 1 FROM meters WHERE id=?",
                                (parent_id,)).fetchone() is None:
                    raise NotFoundError("父表不存在")
            try:
                cur = conn.execute(
                    "INSERT INTO meters(name,parent_id,weight,lower,upper,locked,"
                    "factor,created_at) VALUES(?,?,?,?,?,?,?,?)",
                    (name, parent_id, weight, lo, hi, lk, factor, _now()))
            except sqlite3.IntegrityError:
                raise ValidationError(f"登记表「{name}」失败：名称已存在")
            self._bump(conn)
            return cur.lastrowid

    def set_reading(self, meter_id: int, value) -> None:
        try:
            v = float(value)
        except (TypeError, ValueError):
            raise ValidationError("读数须为数值")
        if not math.isfinite(v):
            raise ValidationError("读数须为有限数值")
        with self._conn() as conn:
            if conn.execute("SELECT 1 FROM meters WHERE id=?",
                            (meter_id,)).fetchone() is None:
                raise NotFoundError("计量表不存在")
            conn.execute(
                "INSERT INTO readings(meter_id,value,updated_at) VALUES(?,?,?) "
                "ON CONFLICT(meter_id) DO UPDATE SET value=excluded.value, "
                "updated_at=excluded.updated_at", (meter_id, v, _now()))
            self._bump(conn)

    def list_meters(self) -> dict:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT m.*, r.value AS reading, r.updated_at AS reading_at "
                "FROM meters m LEFT JOIN readings r ON r.meter_id=m.id "
                "ORDER BY m.id").fetchall()
            version = self._version(conn)
        return {"version": version, "meters": [dict(r) for r in rows]}

    # ---------------------------------------------------------------- 核平
    def reconcile(self, timeout_s: float = 30.0) -> dict:
        data = self.list_meters()
        meters = data["meters"]
        if not meters:
            raise ValidationError("尚未登记计量表")
        missing = [m["name"] for m in meters if m["reading"] is None]
        if missing:
            raise ValidationError("以下计量表缺少读数: " + ", ".join(missing))
        nodes = [MeterNode(meter_id=m["id"], name=m["name"],
                           parent_id=m["parent_id"], reading=m["reading"],
                           weight=m["weight"], lower=m["lower"], upper=m["upper"],
                           lock=m["locked"], factor=m["factor"]) for m in meters]
        snapshot = {
            "source_version": data["version"],
            "captured_at": _now(),
            "meters": [{
                "id": m["id"], "name": m["name"], "parent_id": m["parent_id"],
                "reading": m["reading"], "weight": m["weight"],
                "lower": m["lower"], "upper": m["upper"], "locked": m["locked"],
                "factor": m["factor"],
            } for m in meters],
        }
        try:
            res = solve(nodes, timeout_s=timeout_s)
        except (SolverTimeout, IterationLimitExceeded) as e:
            rid = self._save_run(data["version"], "failed", str(e), snapshot,
                                 None, None, None)
            return self.get_run(rid)
        except SolverError as e:
            raise ValidationError(str(e))
        if res.status == "infeasible":
            conflicts = [{
                "meter": c.meter, "detail": c.detail,
                "items": [{"meter": it.meter, "constraint": it.constraint,
                           "value": it.value} for it in c.items],
            } for c in res.conflicts]
            rid = self._save_run(data["version"], "infeasible",
                                 "无可行解：存在冲突约束，不生成可复核报告",
                                 snapshot, None, conflicts, None)
            return self.get_run(rid)
        result = {"x": res.x, "adjustments": res.adjustments,
                  "residuals": res.residuals, "lam": res.lam,
                  "active_bounds": res.active_bounds}
        stats = {"objective": res.objective, "leaf_total": res.leaf_total,
                 "max_residual": res.max_residual, "iterations": res.iterations}
        rid = self._save_run(data["version"], "completed", None, snapshot,
                             result, None, stats)
        return self.get_run(rid)

    def _save_run(self, version, status, message, snapshot, result, conflicts,
                  stats) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO runs(source_version,status,message,objective,"
                "leaf_total,max_residual,iterations,snapshot_json,result_json,"
                "conflicts_json,created_at,finished_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (version, status, message,
                 stats["objective"] if stats else None,
                 stats["leaf_total"] if stats else None,
                 stats["max_residual"] if stats else None,
                 stats["iterations"] if stats else None,
                 json.dumps(snapshot, ensure_ascii=False),
                 json.dumps(result, ensure_ascii=False)
                 if result is not None else None,
                 json.dumps(conflicts, ensure_ascii=False)
                 if conflicts is not None else None,
                 _now(), _now()))
            return cur.lastrowid

    # ---------------------------------------------------------------- 队列
    def list_runs(self) -> list[dict]:
        with self._conn() as conn:
            version = self._version(conn)
            rows = conn.execute(
                "SELECT id,source_version,status,message,objective,leaf_total,"
                "max_residual,iterations,created_at,finished_at,reviewed_at "
                "FROM runs ORDER BY id DESC").fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["current_version"] = version
            d["stale"] = d["status"] == "completed" and d["source_version"] != version
            out.append(d)
        return out

    def get_run(self, run_id: int) -> dict:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM runs WHERE id=?",
                               (run_id,)).fetchone()
            if row is None:
                raise NotFoundError("核算单不存在")
            version = self._version(conn)
        d = dict(row)
        d["snapshot"] = json.loads(d.pop("snapshot_json"))
        result_json = d.pop("result_json")
        conflicts_json = d.pop("conflicts_json")
        d["result"] = json.loads(result_json) if result_json else None
        d["conflicts"] = json.loads(conflicts_json) if conflicts_json else None
        d["current_version"] = version
        d["stale"] = d["status"] == "completed" and d["source_version"] != version
        return d

    # ---------------------------------------------------------------- 复核
    def approve(self, run_id: int) -> None:
        """原子通过：仅当待复核且版本未变。复核与更新并发时拒绝过时通过。"""
        with self._conn() as conn:
            cur = conn.execute(
                "UPDATE runs SET status='approved', reviewed_at=? "
                "WHERE id=? AND status='completed' AND source_version="
                "(SELECT CAST(value AS INTEGER) FROM meta "
                " WHERE key='source_version')", (_now(), run_id))
            if cur.rowcount == 1:
                return
            row = conn.execute("SELECT status FROM runs WHERE id=?",
                               (run_id,)).fetchone()
            if row is None:
                raise NotFoundError("核算单不存在")
            if row["status"] != "completed":
                raise ReviewStateError(
                    f"核算单状态为 {row['status']}，不可复核通过")
            raise StaleRunError("源数据已更新，该结果已失效，拒绝过时通过")

    def reject(self, run_id: int) -> None:
        with self._conn() as conn:
            cur = conn.execute(
                "UPDATE runs SET status='rejected', reviewed_at=? "
                "WHERE id=? AND status='completed'", (_now(), run_id))
            if cur.rowcount == 1:
                return
            row = conn.execute("SELECT status FROM runs WHERE id=?",
                               (run_id,)).fetchone()
            if row is None:
                raise NotFoundError("核算单不存在")
            raise ReviewStateError(f"核算单状态为 {row['status']}，不可驳回")
