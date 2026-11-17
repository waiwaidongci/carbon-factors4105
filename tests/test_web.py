"""Web 端到端测试：录入、核平、核对页面、复核、失效拒绝。"""
import json
import os
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

from meter_recon.service import ReconService
from meter_recon.web import create_server


class WebTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fd, cls.db = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        os.unlink(cls.db)
        cls.svc = ReconService(cls.db)
        cls.server = create_server(cls.svc, "127.0.0.1", 0)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        # 台账：总表100 / 分表A 40(w=100) / 分表B 40(w=1)，不平衡 20
        cls.root = cls.api("POST", "/api/meters",
                           {"name": "总表", "weight": 1.0, "factor": 0.0})[1]["id"]
        cls.a = cls.api("POST", "/api/meters",
                        {"name": "分表A", "parent_id": cls.root,
                         "weight": 100.0, "factor": 0.5})[1]["id"]
        cls.b = cls.api("POST", "/api/meters",
                        {"name": "分表B", "parent_id": cls.root,
                         "weight": 1.0, "factor": 0.5})[1]["id"]
        for mid, v in ((cls.root, 100.0), (cls.a, 40.0), (cls.b, 40.0)):
            cls.api("POST", "/api/readings", {"meter_id": mid, "value": v})

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)
        if os.path.exists(cls.db):
            os.unlink(cls.db)

    @classmethod
    def api(cls, method, path, payload=None):
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(
            f"http://127.0.0.1:{cls.port}{path}", data=data,
            headers={"Content-Type": "application/json"}, method=method)
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status, json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8")
            try:
                return e.code, json.loads(body)
            except json.JSONDecodeError:
                return e.code, body

    @classmethod
    def page(cls, path):
        with urllib.request.urlopen(f"http://127.0.0.1:{cls.port}{path}",
                                    timeout=10) as resp:
            return resp.status, resp.read().decode("utf-8")

    def test_01_entry_page(self):
        status, body = self.page("/")
        self.assertEqual(status, 200)
        self.assertIn("登记计量表", body)
        self.assertIn("发起核平", body)
        self.assertIn("总表", body)
        self.assertIn("源数据版本", body)

    def test_02_reconcile_and_review_page(self):
        status, run = self.api("POST", "/api/reconcile", {"timeout_s": 30})
        self.assertEqual(status, 201)
        self.assertEqual(run["status"], "completed")
        rid = run["id"]
        # 核对页面：调整量、守恒残差、只按叶表计、快照留存
        status, body = self.page(f"/runs/{rid}")
        self.assertEqual(status, 200)
        self.assertIn("守恒残差", body)
        self.assertIn("调整量", body)
        self.assertIn("只按叶表计", body)
        self.assertIn("原始读数与校正依据", body)
        # 非均摊最优解：分表B（权重1）调整量远大于分表A（权重100）
        x = run["result"]["x"]
        self.assertGreater(abs(x[str(self.b)] - 40.0),
                           50 * abs(x[str(self.a)] - 40.0))
        self.assertLess(run["max_residual"], 1e-8)
        # 复核通过；重复通过被拒
        status, _ = self.api("POST", f"/api/runs/{rid}/approve")
        self.assertEqual(status, 200)
        status, body = self.page(f"/runs/{rid}")
        self.assertIn("已通过", body)
        status, _ = self.api("POST", f"/api/runs/{rid}/approve")
        self.assertEqual(status, 409)

    def test_03_stale_approve_rejected_409(self):
        status, run = self.api("POST", "/api/reconcile", {})
        self.assertEqual(run["status"], "completed")
        rid = run["id"]
        # 源数据更新 → 未复核结果失效 → 过时通过被拒
        status, _ = self.api("POST", "/api/readings",
                             {"meter_id": self.b, "value": 44.0})
        self.assertEqual(status, 200)
        status, r = self.api("POST", f"/api/runs/{rid}/approve")
        self.assertEqual(status, 409)
        self.assertIn("失效", r["error"])
        status, body = self.page(f"/runs/{rid}")
        self.assertIn("已失效", body)
        status, body = self.page("/runs")
        self.assertIn("已失效", body)

    def test_04_timeout_no_fake_result(self):
        status, run = self.api("POST", "/api/reconcile", {"timeout_s": 0})
        self.assertEqual(status, 201)
        self.assertEqual(run["status"], "failed")
        self.assertIn("超时", run["message"])
        self.assertIsNone(run["result"])  # 不以近似冒充完成
        status, body = self.page(f"/runs/{run['id']}")
        self.assertIn("不以近似冒充完成", body)

    def test_05_lock_conflict_page_no_report(self):
        # 总表下挂锁定冲突子树：父锁 500 ≠ 子锁 10
        status, r = self.api("POST", "/api/meters",
                             {"name": "锁定分表", "parent_id": self.root,
                              "locked": 500.0})
        mid = r["id"]
        status, r = self.api("POST", "/api/meters",
                             {"name": "锁定叶表", "parent_id": mid,
                              "locked": 10.0})
        leaf = r["id"]
        self.api("POST", "/api/readings", {"meter_id": mid, "value": 500.0})
        self.api("POST", "/api/readings", {"meter_id": leaf, "value": 10.0})
        status, run = self.api("POST", "/api/reconcile", {})
        self.assertEqual(run["status"], "infeasible")
        self.assertIsNone(run["result"])  # 不生成可复核报告
        self.assertTrue(run["conflicts"])
        status, body = self.page(f"/runs/{run['id']}")
        self.assertEqual(status, 200)
        self.assertIn("无可行解", body)
        self.assertIn("锁定", body)
        self.assertNotIn("复核通过</button>", body)
        status, r = self.api("POST", f"/api/runs/{run['id']}/approve")
        self.assertEqual(status, 409)

    def test_06_validation_errors(self):
        status, r = self.api("POST", "/api/meters", {"name": "零权重", "weight": 0})
        self.assertEqual(status, 400)
        self.assertIn("正数", r["error"])
        status, r = self.api("POST", "/api/readings",
                             {"meter_id": 99999, "value": 1})
        self.assertEqual(status, 404)


if __name__ == "__main__":
    unittest.main()
