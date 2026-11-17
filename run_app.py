#!/usr/bin/env python3
"""启动树形计量核平核算服务。

用法：
    python run_app.py [--db meter_recon.db] [--host 127.0.0.1] [--port 8000] [--demo]

--demo 在空库中播种一套总表/分表演示数据（读数故意不平，便于核对调整）。
"""
from __future__ import annotations

import argparse

from meter_recon.service import ReconService
from meter_recon.web import create_server


def seed_demo(svc: ReconService) -> None:
    """总表-分表-支表三层树，读数不平，权重不同（演示非均摊最优解）。"""
    if svc.list_meters()["meters"]:
        return
    root = svc.add_meter("厂区总表", weight=1.0, factor=0.0)
    a = svc.add_meter("一号分表", parent_id=root, weight=2.0, factor=0.0)
    b = svc.add_meter("二号分表", parent_id=root, weight=2.0, factor=0.0)
    a1 = svc.add_meter("材料车间", parent_id=a, weight=1.0, factor=0.57)
    a2 = svc.add_meter("动力车间", parent_id=a, weight=5.0, factor=0.57)
    b1 = svc.add_meter("办公楼", parent_id=b, weight=1.0, factor=0.57)
    b2 = svc.add_meter("宿舍区", parent_id=b, weight=1.0, factor=0.57,
                       lower=50, upper=400)
    svc.set_reading(root, 1000.0)
    svc.set_reading(a, 615.0)
    svc.set_reading(b, 402.0)
    svc.set_reading(a1, 305.0)
    svc.set_reading(a2, 318.0)
    svc.set_reading(b1, 210.0)
    svc.set_reading(b2, 196.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="树形计量核平核算服务")
    parser.add_argument("--db", default="meter_recon.db", help="SQLite 数据库路径")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--demo", action="store_true", help="空库时播种演示数据")
    args = parser.parse_args()
    svc = ReconService(args.db)
    if args.demo:
        seed_demo(svc)
    server = create_server(svc, args.host, args.port)
    host, port = server.server_address[:2]
    print(f"树形计量核平核算服务已启动: http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
