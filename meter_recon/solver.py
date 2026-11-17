"""树形计量核平核算求解器（仅依赖标准库）。

模型
----
n 块计量表构成一棵树。表 i 有读数 y_i、正权重 w_i、下限 lo_i、上限 hi_i，
可选锁定值 lock_i。求校正值 x：

    min   Σ_i w_i · (x_i − y_i)²
    s.t.  x_p = Σ_{c∈children(p)} x_c     每个父表等于其子表之和（守恒）
          lo_i ≤ x_i ≤ hi_i               限值
          x_i = lock_i                    锁定表

排放只按叶表（无子表的表）的校正值 × 因子合计，避免总表/分表直接相加重复计。

算法
----
主动集法。等式约束 QP 的 KKT 系统是树状稀疏的：以叶→根消元、根→叶回代，
每次 O(n) 精确求解；违限约束批量加入主动集、对偶不可行的约束逐个释放。
无可行解时自底向上做可行区间传播，给出冲突约束清单（不生成可复核报告）。

资源承诺
--------
内存为若干长度 n 的浮点/整数数组，随表数线性增长（万表可行）。
超时或迭代超限抛出异常，绝不返回近似解冒充完成。
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field


class SolverError(Exception):
    """输入数据非法（权重非正、树不连通、缺根等）。"""


class SolverTimeout(Exception):
    """求解超时：中止且不产出任何近似结果。"""


class IterationLimitExceeded(Exception):
    """主动集迭代超限：中止且不产出任何近似结果。"""


class InconsistentActiveSet(Exception):
    """主动集导致等式子问题不一致：携带需修复的退化行节点下标。"""

    def __init__(self, node: int):
        super().__init__(f"主动集在节点 {node} 处不一致")
        self.node = node


@dataclass
class MeterNode:
    """一块计量表的登记信息与读数。"""
    meter_id: int
    name: str
    parent_id: int | None
    reading: float
    weight: float
    lower: float = -math.inf
    upper: float = math.inf
    lock: float | None = None
    factor: float = 1.0


@dataclass
class ConflictItem:
    """一条参与冲突的约束。"""
    meter: str
    constraint: str          # 锁定 / 下限 / 上限 / 可行区间
    value: str


@dataclass
class Conflict:
    """一处不可行冲突：位置、说明、涉及约束清单。"""
    meter: str
    detail: str
    items: list[ConflictItem] = field(default_factory=list)


@dataclass
class SolveResult:
    status: str                                  # 'optimal' | 'infeasible'
    x: dict[int, float] | None = None            # meter_id -> 校正值
    lam: dict[int, float] | None = None          # 父表 meter_id -> 拉格朗日乘子
    objective: float | None = None               # 加权调整平方和
    adjustments: dict[int, float] | None = None  # meter_id -> 调整量 x−y
    residuals: dict[int, float] | None = None    # 父表 meter_id -> 守恒残差
    max_residual: float | None = None
    leaf_total: float | None = None              # 只按叶表计的排放合计
    active_bounds: list[tuple[int, str]] | None = None  # 生效限值 (meter_id, 'lower'|'upper')
    iterations: int = 0
    conflicts: list[Conflict] = field(default_factory=list)


def _fmt(v) -> str:
    if v is None:
        return "—"
    if v == math.inf:
        return "+∞"
    if v == -math.inf:
        return "−∞"
    return f"{v:.6g}"


class _Tree:
    """求解器内部数组表示，全部长度 O(n)。"""
    __slots__ = ("n", "meter_ids", "names", "parent", "children", "pre", "post",
                 "y", "w", "s", "elo", "ehi", "lock", "factor", "scale")


def _prepare(nodes: list[MeterNode]) -> _Tree:
    n = len(nodes)
    if n == 0:
        raise SolverError("尚未登记任何计量表")
    t = _Tree()
    t.n = n
    t.meter_ids = [nd.meter_id for nd in nodes]
    t.names = [nd.name for nd in nodes]
    index = {}
    for i, mid in enumerate(t.meter_ids):
        if mid in index:
            raise SolverError(f"表ID重复: {mid}")
        index[mid] = i
    parent = [-1] * n
    children: list[list[int]] = [[] for _ in range(n)]
    root = -1
    for i, nd in enumerate(nodes):
        if nd.parent_id is None:
            if root != -1:
                raise SolverError("存在多个根表，计量关系须为单棵树")
            root = i
        else:
            if nd.parent_id not in index:
                raise SolverError(f"表「{nd.name}」的父表未登记")
            p = index[nd.parent_id]
            parent[i] = p
            children[p].append(i)
    if root == -1:
        raise SolverError("缺少根表，计量关系须为单棵树")
    order: list[int] = []
    stack = [root]
    while stack:                       # 迭代先序：父先子后，无递归深度风险
        i = stack.pop()
        order.append(i)
        stack.extend(children[i])
    if len(order) != n:
        raise SolverError("计量树不连通或存在环")
    t.parent = parent
    t.children = children
    t.pre = order                      # 父先子后（回代用）
    t.post = order[::-1]               # 子先父后（消元/区间传播用）
    y = [0.0] * n
    w = [0.0] * n
    elo = [0.0] * n
    ehi = [0.0] * n
    lock: list[float | None] = [None] * n
    factor = [0.0] * n
    scale = 1.0
    for i, nd in enumerate(nodes):
        if not math.isfinite(nd.reading):
            raise SolverError(f"表「{nd.name}」读数非法")
        if not (math.isfinite(nd.weight) and nd.weight > 0.0):
            raise SolverError(f"表「{nd.name}」权重必须为正数")
        lo = -math.inf if nd.lower is None else float(nd.lower)
        hi = math.inf if nd.upper is None else float(nd.upper)
        if nd.lock is not None and not math.isfinite(nd.lock):
            raise SolverError(f"表「{nd.name}」锁定值非法")
        if not math.isfinite(nd.factor):
            raise SolverError(f"表「{nd.name}」排放因子非法")
        y[i] = float(nd.reading)
        w[i] = float(nd.weight)
        elo[i] = lo
        ehi[i] = hi
        lock[i] = None if nd.lock is None else float(nd.lock)
        factor[i] = float(nd.factor)
        scale = max(scale, abs(y[i]), abs(w[i]))
        if math.isfinite(lo):
            scale = max(scale, abs(lo))
        if math.isfinite(hi):
            scale = max(scale, abs(hi))
        if lock[i] is not None:
            scale = max(scale, abs(lock[i]))
    t.y, t.w, t.elo, t.ehi, t.lock, t.factor = y, w, elo, ehi, lock, factor
    t.s = [0.5 / wi for wi in w]
    t.scale = scale
    return t


def _check_feasible(t: _Tree) -> list[Conflict]:
    """自底向上可行区间传播；返回冲突清单（空表示可行）。

    子表合计可行区间 [ΣLo, ΣHi] 与父表自身区间 [lo, hi] 不相交即不可行，
    对树守恒 + 区间限值该判定是精确的。冲突处列出涉及约束便于定位。
    """
    n = t.n
    tol = 1e-9 * t.scale
    Lo = [0.0] * n
    Hi = [0.0] * n
    bad = [False] * n
    conflicts: list[Conflict] = []
    for i in t.post:
        li, hi = t.elo[i], t.ehi[i]
        lv = t.lock[i]
        if lv is not None:
            if lv < li - tol or lv > hi + tol:
                bad[i] = True
                conflicts.append(Conflict(
                    meter=t.names[i],
                    detail=f"表「{t.names[i]}」锁定值 {_fmt(lv)} 超出限值 [{_fmt(li)}, {_fmt(hi)}]",
                    items=[ConflictItem(t.names[i], "锁定", _fmt(lv)),
                           ConflictItem(t.names[i], "下限", _fmt(li)),
                           ConflictItem(t.names[i], "上限", _fmt(hi))]))
                Lo[i] = Hi[i] = lv
                continue
            li = hi = lv
        if not t.children[i]:
            if li > hi + tol:
                bad[i] = True
                conflicts.append(Conflict(
                    meter=t.names[i],
                    detail=f"表「{t.names[i]}」下限 {_fmt(li)} 大于上限 {_fmt(hi)}",
                    items=[ConflictItem(t.names[i], "下限", _fmt(li)),
                           ConflictItem(t.names[i], "上限", _fmt(hi))]))
            Lo[i], Hi[i] = li, hi
            continue
        slo = 0.0
        shi = 0.0
        child_bad = False
        for c in t.children[i]:
            slo += Lo[c]
            shi += Hi[c]
            if bad[c]:
                child_bad = True
        if child_bad:                # 子树已有冲突，不再重复上报祖先
            bad[i] = True
            Lo[i] = li if li > slo else slo
            Hi[i] = hi if hi < shi else shi
            continue
        lo_i = li if li > slo else slo
        hi_i = hi if hi < shi else shi
        if lo_i > hi_i + tol:
            bad[i] = True
            items = [ConflictItem(t.names[i], "下限", _fmt(li)),
                     ConflictItem(t.names[i], "上限", _fmt(hi))]
            if lv is not None:
                items.append(ConflictItem(t.names[i], "锁定", _fmt(lv)))
            for c in t.children[i]:
                items.append(ConflictItem(t.names[c], "可行区间",
                                          f"[{_fmt(Lo[c])}, {_fmt(Hi[c])}]"))
                if t.lock[c] is not None:
                    items.append(ConflictItem(t.names[c], "锁定", _fmt(t.lock[c])))
                else:
                    if math.isfinite(t.elo[c]):
                        items.append(ConflictItem(t.names[c], "下限", _fmt(t.elo[c])))
                    if math.isfinite(t.ehi[c]):
                        items.append(ConflictItem(t.names[c], "上限", _fmt(t.ehi[c])))
            conflicts.append(Conflict(
                meter=t.names[i],
                detail=(f"父表「{t.names[i]}」自身区间 [{_fmt(li)}, {_fmt(hi)}] "
                        f"与子表合计区间 [{_fmt(slo)}, {_fmt(shi)}] 不相交"),
                items=items))
            Lo[i], Hi[i] = lo_i, hi_i
            continue
        if lo_i > hi_i:                # 容差内：收拢为空前的中点，保持区间有效
            m = 0.5 * (lo_i + hi_i)
            lo_i = hi_i = m
        Lo[i], Hi[i] = lo_i, hi_i
    return conflicts


def _solve_eq(t: _Tree, fixed: list[bool], fval: list[float]):
    """在当前固定集下精确求解等式约束 QP（树 KKT，O(n)）。

    未知量 λ_p（每个父表一个守恒约束）。自由表：
        x_i = y_i + s_i·(λ_parent(i) − λ_i·[i 是父表])，s_i = 1/(2w_i)
    代入守恒约束得树状对称正定三对角型系统，叶→根消元、根→叶回代。
    """
    n = t.n
    s = t.s
    y = t.y
    children = t.children
    parent = t.parent
    diag = [0.0] * n
    rhs = [0.0] * n
    cp = [0.0] * n                    # 与父表 λ 的耦合系数
    for p in range(n):
        ch = children[p]
        if not ch:
            continue
        sf = 0.0
        sy = 0.0
        ss = 0.0
        for c in ch:
            if fixed[c]:
                sf += fval[c]
            else:
                sy += y[c]
                ss += s[c]
        if fixed[p]:
            diag[p] = ss
            rhs[p] = fval[p] - sf - sy
        else:
            diag[p] = s[p] + ss
            rhs[p] = y[p] - sf - sy
            cp[p] = -s[p] if parent[p] != -1 else 0.0
    smax = 1.0
    for si in s:
        if si > smax:
            smax = si
    eps = 1e-12 * smax
    tol_c = 1e-7 * t.scale
    for p in t.post:                  # 消元：子先父后
        if not children[p]:
            continue
        if diag[p] <= eps:            # 退化行（子树全固定/线性相关）
            if abs(rhs[p]) > tol_c:   # 主动集造成不一致：交由外层释放限值修复
                raise InconsistentActiveSet(p)
            diag[p] = 1.0             # 一致性已由可行性预检保证：λ 置零即可
            rhs[p] = 0.0
            cp[p] = 0.0
            continue
        par = parent[p]
        if par == -1:
            continue
        f = cp[p] / diag[p]
        diag[par] -= cp[p] * f
        rhs[par] -= rhs[p] * f
    lam = [0.0] * n
    for p in t.pre:                   # 回代：父先子后
        if not children[p]:
            continue
        par = parent[p]
        lam[p] = (rhs[p] - (cp[p] * lam[par] if par != -1 else 0.0)) / diag[p]
    x = [0.0] * n
    for i in range(n):
        if fixed[i]:
            x[i] = fval[i]
        else:
            v = y[i]
            par = parent[i]
            if par != -1:
                v += s[i] * lam[par]
            if children[i]:
                v -= s[i] * lam[i]
            x[i] = v
    return x, lam


def _release_candidate(t: _Tree, fixed: list[bool], p: int) -> int | None:
    """在不一致退化行的刚性锥内找一个可释放的主动限值（锁定永不释放）。

    锥 = 从 p 向下、穿过自由内部节点到达的所有固定节点（固定节点自身子树
    自洽，不下钻）。释放其中任一非锁定固定节点即可恢复一个自由度。
    """
    if fixed[p] and t.lock[p] is None:
        return p
    stack = list(t.children[p])
    while stack:
        u = stack.pop()
        if fixed[u]:
            if t.lock[u] is None:
                return u
            continue
        if t.children[u]:
            stack.extend(t.children[u])
    return None


def solve(nodes: list[MeterNode], timeout_s: float | None = None,
          max_iterations: int | None = None) -> SolveResult:
    """求解树形计量核平问题。

    timeout_s 为 None 表示不限时；超时抛 SolverTimeout，迭代超限抛
    IterationLimitExceeded，两者都不产出任何结果（拒绝近似冒充完成）。
    无可行解返回 status='infeasible' 及冲突约束清单。
    """
    t0 = time.monotonic()
    deadline = None if timeout_s is None else t0 + max(0.0, float(timeout_s))
    t = _prepare(nodes)
    n = t.n
    conflicts = _check_feasible(t)
    if conflicts:
        return SolveResult(status="infeasible", conflicts=conflicts)

    tol_p = 1e-9 * t.scale
    gscale = 1.0
    for i in range(n):
        gscale = max(gscale, 2.0 * t.w[i] * t.scale)
    tol_d = 1e-7 * gscale
    max_it = max_iterations if max_iterations is not None else 8 * n + 64

    fixed = [t.lock[i] is not None for i in range(n)]
    fval = [t.lock[i] if t.lock[i] is not None else 0.0 for i in range(n)]
    at_upper = [False] * n
    it = 0
    while True:
        if deadline is not None and time.monotonic() >= deadline:
            raise SolverTimeout(f"求解超时（超过 {timeout_s}s 时限），已中止，未产出任何结果")
        it += 1
        if it > max_it:
            raise IterationLimitExceeded(
                f"主动集迭代超过上限 {max_it}，已中止，未产出任何结果")
        try:
            x, lam = _solve_eq(t, fixed, fval)
        except InconsistentActiveSet as e:
            cand = _release_candidate(t, fixed, e.node)
            if cand is None:          # 锥内全是锁定：预检之外的数值退化，显式失败
                raise SolverError("锁定约束相互矛盾，无法核平")
            fixed[cand] = False       # 释放造成不一致的主动限值后重解
            at_upper[cand] = False
            continue
        violated = False
        for i in range(n):            # 原可行：违限约束批量加入主动集
            if fixed[i]:
                continue
            if x[i] < t.elo[i] - tol_p:
                fixed[i] = True
                at_upper[i] = False
                fval[i] = t.elo[i]
                violated = True
            elif x[i] > t.ehi[i] + tol_p:
                fixed[i] = True
                at_upper[i] = True
                fval[i] = t.ehi[i]
                violated = True
        if violated:
            continue
        worst = 0.0                   # 对偶可行：释放最违的主动限值（锁定永不释放）
        wi = -1
        for i in range(n):
            if not fixed[i] or t.lock[i] is not None:
                continue
            r = 2.0 * t.w[i] * (x[i] - t.y[i])
            if t.children[i]:
                r += lam[i]
            p = t.parent[i]
            if p != -1:
                r -= lam[p]
            if at_upper[i]:
                if r > worst and r > tol_d:
                    worst = r
                    wi = i
            else:
                if -r > worst and -r > tol_d:
                    worst = -r
                    wi = i
        if wi == -1:
            break
        fixed[wi] = False
        at_upper[wi] = False

    residuals: dict[int, float] = {}
    max_res = 0.0
    for p in range(n):
        if t.children[p]:
            acc = 0.0
            for c in t.children[p]:
                acc += x[c]
            r = x[p] - acc
            residuals[t.meter_ids[p]] = r
            if abs(r) > max_res:
                max_res = abs(r)
    objective = 0.0
    leaf_total = 0.0
    xd: dict[int, float] = {}
    adjustments: dict[int, float] = {}
    for i in range(n):
        mid = t.meter_ids[i]
        xd[mid] = x[i]
        adjustments[mid] = x[i] - t.y[i]
        d = x[i] - t.y[i]
        objective += t.w[i] * d * d
        if not t.children[i]:         # 只按叶表计排放
            leaf_total += x[i] * t.factor[i]
    lam_d = {t.meter_ids[p]: lam[p] for p in range(n) if t.children[p]}
    active = [(t.meter_ids[i], "upper" if at_upper[i] else "lower")
              for i in range(n) if fixed[i] and t.lock[i] is None]
    return SolveResult(status="optimal", x=xd, lam=lam_d, objective=objective,
                       adjustments=adjustments, residuals=residuals,
                       max_residual=max_res, leaf_total=leaf_total,
                       active_bounds=active, iterations=it)
