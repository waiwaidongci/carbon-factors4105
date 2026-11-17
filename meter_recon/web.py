"""录入 / 核对 Web 界面与 JSON API（仅标准库 http.server）。

页面
----
- GET  /            录入页：台账、登记表、录读数、发起核平
- GET  /runs        核平队列：状态、版本、失效标注
- GET  /runs/{id}   复核页：核对调整量与守恒残差、冲突约束、通过/驳回
- POST /meters /readings /reconcile /runs/{id}/approve /runs/{id}/reject

JSON API（测试与集成用）
------------------------
- GET  /api/meters            POST /api/meters
- POST /api/readings          POST /api/reconcile
- GET  /api/runs              GET  /api/runs/{id}
- POST /api/runs/{id}/approve POST /api/runs/{id}/reject
"""
from __future__ import annotations

import json
import math
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .service import (NotFoundError, ReconService, ReviewStateError,
                      StaleRunError, ValidationError)

STATUS_LABEL = {
    "completed": "待复核",
    "approved": "已通过",
    "rejected": "已驳回",
    "infeasible": "无可行解",
    "failed": "失败",
}

STYLE = """
body{font-family:"Segoe UI",system-ui,sans-serif;margin:0;background:#f6f7f4;color:#24302a}
header{background:#2e5d43;color:#fff;padding:14px 24px}
header h1{margin:0;font-size:18px}
header a{color:#cfe6d8;margin-right:14px;font-size:13px}
main{max-width:1080px;margin:20px auto;padding:0 16px}
.card{background:#fff;border:1px solid #dde3dc;border-radius:8px;padding:16px 18px;margin-bottom:16px}
.card h2{margin:0 0 12px;font-size:15px;color:#2e5d43}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{border:1px solid #e2e7e1;padding:6px 8px;text-align:right}
th{background:#eef3ee;text-align:center}
td.l,th.l{text-align:left}
input,select{padding:5px 6px;border:1px solid #c6cfc6;border-radius:4px;font-size:13px}
button{background:#2e5d43;color:#fff;border:0;border-radius:4px;padding:7px 16px;font-size:13px;cursor:pointer}
button.ghost{background:#8a9a8e}
button:disabled{background:#c6cfc6;cursor:not-allowed}
.badge{display:inline-block;padding:2px 10px;border-radius:10px;font-size:12px;background:#e6ece6}
.badge.ok{background:#d9ecd9;color:#245c33}.badge.warn{background:#f7e8c9;color:#8a5a00}
.badge.err{background:#f6d5d5;color:#8c1f1f}.badge.info{background:#dbe7f6;color:#1f4e8c}
.msg{background:#e3f0e3;border:1px solid #b9d8b9;padding:8px 12px;border-radius:6px;margin-bottom:12px}
.errbox{background:#f9e3e3;border:1px solid #e0a8a8;padding:10px 14px;border-radius:6px;margin-bottom:12px}
.hint{color:#5c6b60;font-size:12px}
.pos{color:#8c1f1f}.neg{color:#1f6e8c}
pre{background:#f2f4f1;padding:10px;border-radius:6px;overflow:auto;font-size:12px}
form.inline{display:inline}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:8px;align-items:end}
.grid label{font-size:12px;color:#5c6b60;display:block;margin-bottom:3px}
"""


def _f(v, digits=6):
    if v is None:
        return "—"
    if isinstance(v, float):
        if math.isinf(v):
            return "+∞" if v > 0 else "−∞"
        if v != 0 and (abs(v) < 1e-4 or abs(v) >= 1e7):
            return f"{v:.4e}"
        return f"{v:.{digits}g}"
    return str(v)


def _signed(v):
    if v is None:
        return "—"
    cls = "pos" if v > 0 else ("neg" if v < 0 else "")
    sign = "+" if v > 0 else ""
    return f'<span class="{cls}">{sign}{_f(v)}</span>'


def _status_badge(run):
    st = run["status"]
    if run.get("stale"):
        return '<span class="badge warn">已失效</span>'
    cls = {"completed": "info", "approved": "ok", "rejected": "warn",
           "infeasible": "err", "failed": "err"}.get(st, "")
    return f'<span class="badge {cls}">{STATUS_LABEL.get(st, st)}</span>'


def _layout(title, body):
    return f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>{escape(title)} · 树形计量核平</title><style>{STYLE}</style></head><body>
<header><h1>树形计量核平核算</h1>
<a href="/">录入台账</a><a href="/runs">核平队列</a></header>
<main>{body}</main></body></html>"""


def page_entry(svc: ReconService, msg: str = "") -> str:
    data = svc.list_meters()
    meters = data["meters"]
    rows = []
    for m in meters:
        rows.append(
            f"<tr><td>{m['id']}</td><td class='l'>{escape(m['name'])}</td>"
            f"<td>{m['parent_id'] or '—'}</td><td>{_f(m['reading'])}</td>"
            f"<td>{_f(m['weight'])}</td><td>{_f(m['lower'])}</td>"
            f"<td>{_f(m['upper'])}</td><td>{_f(m['locked'])}</td>"
            f"<td>{_f(m['factor'])}</td></tr>")
    table = ("<table><tr><th>ID</th><th class='l'>名称</th><th>父表</th><th>读数</th>"
             "<th>权重</th><th>下限</th><th>上限</th><th>锁定</th><th>排放因子</th></tr>"
             + "".join(rows) + "</table>") if rows else "<p class='hint'>尚未登记计量表。</p>"
    options = "".join(f"<option value='{m['id']}'>{escape(m['name'])}</option>"
                      for m in meters)
    banner = f"<div class='msg'>{escape(msg)}</div>" if msg else ""
    body = f"""
{banner}
<div class="card"><h2>计量台账 <span class="badge">源数据版本 v{data['version']}</span></h2>{table}</div>
<div class="card"><h2>登记计量表</h2>
<form method="post" action="/meters"><div class="grid">
<div><label>名称</label><input name="name" required></div>
<div><label>父表（根表留空）</label><select name="parent_id"><option value="">—</option>{options}</select></div>
<div><label>权重（正数）</label><input name="weight" value="1" required></div>
<div><label>下限</label><input name="lower"></div>
<div><label>上限</label><input name="upper"></div>
<div><label>锁定值</label><input name="locked"></div>
<div><label>排放因子</label><input name="factor" value="1"></div>
<div><button type="submit">登记</button></div>
</div></form></div>
<div class="card"><h2>录入 / 更新读数</h2>
<form method="post" action="/readings"><div class="grid">
<div><label>计量表</label><select name="meter_id">{options}</select></div>
<div><label>读数</label><input name="value" required></div>
<div><button type="submit">保存读数</button></div>
</div></form>
<p class="hint">保存读数或登记表都会推进源数据版本，未复核的旧结果随即失效。</p></div>
<div class="card"><h2>发起核平</h2>
<form method="post" action="/reconcile"><div class="grid">
<div><label>超时（秒）</label><input name="timeout_s" value="30"></div>
<div><button type="submit">开始核平并入队</button></div>
</div></form>
<p class="hint">最小化全体加权调整平方和，使每个父表等于其子表之和且满足限值；排放只按叶表计。</p></div>
"""
    return _layout("录入", body)


def page_runs(svc: ReconService) -> str:
    runs = svc.list_runs()
    rows = []
    for r in runs:
        rows.append(
            f"<tr><td><a href='/runs/{r['id']}'>#{r['id']}</a></td>"
            f"<td class='l'>{escape(r['created_at'])}</td>"
            f"<td>v{r['source_version']} / v{r['current_version']}</td>"
            f"<td>{_status_badge(r)}</td><td>{_f(r['objective'])}</td>"
            f"<td>{_f(r['leaf_total'])}</td><td>{_f(r['max_residual'])}</td>"
            f"<td>{r['iterations'] or '—'}</td>"
            f"<td class='l'>{escape(r['message'] or '')}</td></tr>")
    table = ("<table><tr><th>单号</th><th class='l'>创建时间</th><th>版本(快照/当前)</th>"
             "<th>状态</th><th>目标值</th><th>叶表排放合计</th><th>最大守恒残差</th>"
             "<th>迭代</th><th class='l'>说明</th></tr>" + "".join(rows) + "</table>"
             ) if rows else "<p class='hint'>暂无核平记录。</p>"
    return _layout("核平队列", f"<div class='card'><h2>核平队列（留存原始读数与校正依据）</h2>{table}</div>")


def page_run(svc: ReconService, run_id: int, query: dict) -> str:
    run = svc.get_run(run_id)
    msg = ""
    if query.get("msg"):
        msg = f"<div class='msg'>{escape(query['msg'][0])}</div>"
    banner = f"<h2>核算单 #{run['id']} {_status_badge(run)}</h2>"
    banner += (f"<p class='hint'>快照版本 v{run['source_version']} · 当前版本 "
               f"v{run['current_version']} · 创建于 {escape(run['created_at'])}"
               + (f" · 复核于 {escape(run['reviewed_at'])}" if run.get("reviewed_at") else "")
               + "</p>")
    if run["stale"]:
        banner += ("<div class='errbox'>源数据已更新，该结果已失效，"
                   "不能复核通过；请重新发起核平。</div>")
    if run["status"] == "failed":
        return _layout("核算单", f"{msg}<div class='card'>{banner}"
                       f"<div class='errbox'>{escape(run['message'] or '')}"
                       f"<br>未产出任何结果，不以近似冒充完成。</div></div>")
    if run["status"] == "infeasible":
        blocks = []
        for c in run["conflicts"] or []:
            items = "".join(
                f"<tr><td class='l'>{escape(it['meter'])}</td>"
                f"<td>{escape(it['constraint'])}</td><td>{escape(it['value'])}</td></tr>"
                for it in c["items"])
            blocks.append(
                f"<div class='errbox'><b>{escape(c['detail'])}</b>"
                f"<table style='margin-top:8px'><tr><th class='l'>表</th>"
                f"<th>约束</th><th>值</th></tr>{items}</table></div>")
        return _layout("核算单", f"{msg}<div class='card'>{banner}"
                       "<div class='errbox'><b>无可行解，不生成可复核报告。</b>"
                       "请调整下列冲突约束后重新核平：</div>"
                       + "".join(blocks) + "</div>")
    # completed / approved / rejected：调整核对表
    snap = {str(m["id"]): m for m in run["snapshot"]["meters"]}
    result = run["result"]
    x = result["x"]
    adj = result["adjustments"]
    residuals = result["residuals"]
    children = {}
    for m in run["snapshot"]["meters"]:
        if m["parent_id"] is not None:
            children.setdefault(str(m["parent_id"]), 0)
            children[str(m["parent_id"])] += 1
    rows = []
    all_sum = 0.0
    for mid, m in snap.items():
        is_parent = children.get(mid, 0) > 0
        xv = x[mid]
        all_sum += xv * m["factor"]
        res = f"{_f(residuals[mid], 4)}" if is_parent else ""
        rows.append(
            f"<tr><td class='l'>{escape(m['name'])}</td>"
            f"<td>{'父表' if is_parent else '叶表'}</td>"
            f"<td>{_f(m['reading'])}</td><td><b>{_f(xv)}</b></td>"
            f"<td>{_signed(adj[mid])}</td><td>{_f(m['weight'])}</td>"
            f"<td>{_f(m['lower'])}</td><td>{_f(m['upper'])}</td>"
            f"<td>{_f(m['locked'])}</td><td>{res}</td></tr>")
    table = ("<table><tr><th class='l'>表</th><th>类型</th><th>原始读数</th>"
             "<th>校正值</th><th>调整量</th><th>权重</th><th>下限</th><th>上限</th>"
             "<th>锁定</th><th>守恒残差</th></tr>" + "".join(rows) + "</table>")
    summary = (
        f"<p>加权调整平方和（目标值）：<b>{_f(run['objective'])}</b>　"
        f"最大守恒残差：<b>{_f(run['max_residual'], 4)}</b>　"
        f"迭代：{run['iterations']}</p>"
        f"<p>排放合计（<b>只按叶表计</b>）：<b>{_f(run['leaf_total'])}</b>　"
        f"<span class='hint'>对照：全部表直接相加 {_f(all_sum)}（重复计，仅参考）</span></p>")
    actions = ""
    if run["status"] == "completed" and not run["stale"]:
        actions = ("<form class='inline' method='post' "
                   f"action='/runs/{run_id}/approve'><button type='submit'>复核通过</button></form> "
                   f"<form class='inline' method='post' action='/runs/{run_id}/reject'>"
                   "<button type='submit' class='ghost'>驳回</button></form>")
    elif run["status"] == "completed" and run["stale"]:
        actions = "<button disabled>复核通过</button> <span class='hint'>已失效不可通过</span>"
    snap_json = escape(json.dumps(run["snapshot"], ensure_ascii=False, indent=1))
    body = (f"{msg}<div class='card'>{banner}{summary}{table}<p>{actions}</p></div>"
            f"<div class='card'><h2>原始读数与校正依据（快照留存）</h2>"
            f"<details><summary class='hint'>展开快照 JSON</summary><pre>{snap_json}</pre></details></div>")
    return _layout("核算单", body)


def make_handler(svc: ReconService):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        # ---------------------------------------------------------- 工具
        def _send(self, code, body, ctype="text/html; charset=utf-8"):
            data = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _json(self, code, obj):
            self._send(code, json.dumps(obj, ensure_ascii=False),
                       "application/json; charset=utf-8")

        def _redirect(self, location):
            self.send_response(303)
            self.send_header("Location", location)
            self.end_headers()

        def _error(self, code, message):
            if self.path.startswith("/api/"):
                self._json(code, {"error": message})
            else:
                self._send(code, _layout("错误",
                                         f"<div class='errbox'>{escape(message)}</div>"
                                         "<p><a href='javascript:history.back()'>返回</a></p>"))

        def _body(self):
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            ctype = self.headers.get("Content-Type", "")
            if "application/json" in ctype:
                return json.loads(raw.decode("utf-8") or "{}")
            return {k: v[0] for k, v in parse_qs(raw.decode("utf-8")).items()}

        # ---------------------------------------------------------- 路由
        def do_GET(self):
            parsed = urlparse(self.path)
            path = parsed.path
            query = parse_qs(parsed.query)
            try:
                if path == "/":
                    self._send(200, page_entry(svc, (query.get("msg") or [""])[0]))
                elif path == "/runs":
                    self._send(200, page_runs(svc))
                elif path.startswith("/runs/"):
                    self._send(200, page_run(svc, int(path.split("/")[2]), query))
                elif path == "/api/meters":
                    self._json(200, svc.list_meters())
                elif path == "/api/runs":
                    self._json(200, {"runs": svc.list_runs()})
                elif path.startswith("/api/runs/"):
                    self._json(200, svc.get_run(int(path.split("/")[3])))
                else:
                    self._error(404, "页面不存在")
            except NotFoundError as e:
                self._error(404, str(e))
            except (ValueError, IndexError):
                self._error(400, "请求格式错误")

        def do_POST(self):
            path = urlparse(self.path).path
            try:
                form = self._body()
                if path == "/meters":
                    svc.add_meter(
                        name=form.get("name"),
                        parent_id=int(form["parent_id"]) if form.get("parent_id") else None,
                        weight=form.get("weight", "1"), lower=form.get("lower") or None,
                        upper=form.get("upper") or None,
                        locked=form.get("locked") or None,
                        factor=form.get("factor", "1"))
                    self._redirect("/?msg=" + "登记成功")
                elif path == "/readings":
                    svc.set_reading(int(form["meter_id"]), form.get("value"))
                    self._redirect("/?msg=读数已保存")
                elif path == "/reconcile":
                    run = svc.reconcile(timeout_s=float(form.get("timeout_s") or 30))
                    self._redirect(f"/runs/{run['id']}")
                elif path.startswith("/runs/") and path.endswith("/approve"):
                    svc.approve(int(path.split("/")[2]))
                    self._redirect(f"/runs/{path.split('/')[2]}?msg=已复核通过")
                elif path.startswith("/runs/") and path.endswith("/reject"):
                    svc.reject(int(path.split("/")[2]))
                    self._redirect(f"/runs/{path.split('/')[2]}?msg=已驳回")
                elif path == "/api/meters":
                    mid = svc.add_meter(
                        name=form.get("name"), parent_id=form.get("parent_id"),
                        weight=form.get("weight", 1), lower=form.get("lower"),
                        upper=form.get("upper"), locked=form.get("locked"),
                        factor=form.get("factor", 1))
                    self._json(201, {"id": mid})
                elif path == "/api/readings":
                    svc.set_reading(int(form["meter_id"]), form.get("value"))
                    self._json(200, {"ok": True})
                elif path == "/api/reconcile":
                    run = svc.reconcile(timeout_s=float(form.get("timeout_s", 30)))
                    self._json(201, run)
                elif path.startswith("/api/runs/") and path.endswith("/approve"):
                    svc.approve(int(path.split("/")[3]))
                    self._json(200, {"ok": True})
                elif path.startswith("/api/runs/") and path.endswith("/reject"):
                    svc.reject(int(path.split("/")[3]))
                    self._json(200, {"ok": True})
                else:
                    self._error(404, "接口不存在")
            except ValidationError as e:
                self._error(400, str(e))
            except NotFoundError as e:
                self._error(404, str(e))
            except StaleRunError as e:
                self._error(409, str(e))
            except ReviewStateError as e:
                self._error(409, str(e))
            except (ValueError, KeyError, IndexError):
                self._error(400, "请求格式错误")

    return Handler


def create_server(svc: ReconService, host: str = "127.0.0.1", port: int = 8000):
    return ThreadingHTTPServer((host, port), make_handler(svc))
