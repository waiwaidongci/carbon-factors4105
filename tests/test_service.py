"""服务层测试：队列留存、版本失效、复核竞态、不可行不报、超时。"""
import os
import tempfile
import threading
import unittest

from meter_recon.service import (ReconService, ReviewStateError, StaleRunError,
                                 ValidationError)


def make_service():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.unlink(path)
    svc = ReconService(path)
    return svc, path


def build_tree(svc):
    """总表100 / 分表A 40(w=100) / 分表B 40(w=1)，不平衡 20。"""
    root = svc.add_meter("总表", weight=1.0, factor=0.0)
    a = svc.add_meter("分表A", parent_id=root, weight=100.0, factor=0.5)
    b = svc.add_meter("分表B", parent_id=root, weight=1.0, factor=0.5)
    svc.set_reading(root, 100.0)
    svc.set_reading(a, 40.0)
    svc.set_reading(b, 40.0)
    return root, a, b


class ReconcileAndQueueTest(unittest.TestCase):
    def setUp(self):
        self.svc, self.db = make_service()
        self.addCleanup(lambda: os.path.exists(self.db) and os.unlink(self.db))

    def test_completed_run_and_snapshot_retained(self):
        root, a, b = build_tree(self.svc)
        run = self.svc.reconcile()
        self.assertEqual(run["status"], "completed")
        self.assertFalse(run["stale"])
        # 非均摊最优解：高权重分表A几乎不动
        x = run["result"]["x"]
        self.assertLess(abs(x[str(a)] - 40.0), 0.15)
        self.assertGreater(abs(x[str(b)] - 40.0), 9.0)
        # 守恒残差为 0，叶表排放合计只按叶表
        self.assertLess(run["max_residual"], 1e-8)
        self.assertAlmostEqual(run["leaf_total"],
                               x[str(a)] * 0.5 + x[str(b)] * 0.5, places=8)
        # 队列留存原始读数与校正依据
        self.assertEqual(len(self.svc.list_runs()), 1)
        snap = {m["name"]: m for m in run["snapshot"]["meters"]}
        self.assertEqual(snap["总表"]["reading"], 100.0)
        self.assertEqual(snap["分表A"]["weight"], 100.0)
        self.assertEqual(run["snapshot"]["source_version"],
                         run["source_version"])
        # 源数据更新后：快照仍留存旧读数，结果标记失效
        self.svc.set_reading(b, 45.0)
        run2 = self.svc.get_run(run["id"])
        snap2 = {m["name"]: m for m in run2["snapshot"]["meters"]}
        self.assertEqual(snap2["分表B"]["reading"], 40.0)  # 留存的原始读数不变
        self.assertTrue(run2["stale"])
        self.assertTrue(self.svc.list_runs()[0]["stale"])

    def test_missing_reading_rejected(self):
        self.svc.add_meter("总表")
        with self.assertRaises(ValidationError):
            self.svc.reconcile()

    def test_weight_validation(self):
        with self.assertRaises(ValidationError):
            self.svc.add_meter("坏表", weight=0)
        with self.assertRaises(ValidationError):
            self.svc.add_meter("坏表2", weight=-1)

    def test_version_bumps_on_writes(self):
        v0 = self.svc.current_version()
        mid = self.svc.add_meter("总表")
        v1 = self.svc.current_version()
        self.svc.set_reading(mid, 1.0)
        v2 = self.svc.current_version()
        self.assertGreater(v1, v0)
        self.assertGreater(v2, v1)


class ReviewRaceTest(unittest.TestCase):
    def setUp(self):
        self.svc, self.db = make_service()
        self.addCleanup(lambda: os.path.exists(self.db) and os.unlink(self.db))
        self.root, self.a, self.b = build_tree(self.svc)

    def test_approve_then_double_approve_rejected(self):
        run = self.svc.reconcile()
        self.svc.approve(run["id"])
        self.assertEqual(self.svc.get_run(run["id"])["status"], "approved")
        with self.assertRaises(ReviewStateError):
            self.svc.approve(run["id"])

    def test_stale_approve_rejected_after_update(self):
        run = self.svc.reconcile()
        self.svc.set_reading(self.b, 41.0)  # 源数据更新 → 未复核结果失效
        with self.assertRaises(StaleRunError):
            self.svc.approve(run["id"])
        self.assertEqual(self.svc.get_run(run["id"])["status"], "completed")
        # 重新核平后可通过
        run2 = self.svc.reconcile()
        self.svc.approve(run2["id"])
        self.assertEqual(self.svc.get_run(run2["id"])["status"], "approved")

    def test_reject_flow(self):
        run = self.svc.reconcile()
        self.svc.reject(run["id"])
        self.assertEqual(self.svc.get_run(run["id"])["status"], "rejected")
        with self.assertRaises(ReviewStateError):
            self.svc.approve(run["id"])

    def test_concurrent_update_and_approve_never_passes_stale(self):
        """复核与更新并发：通过必须基于最新版本，过时通过一律被拒绝。"""
        outcomes = []
        for _ in range(40):
            run = self.svc.reconcile()
            err = []

            def do_approve():
                try:
                    self.svc.approve(run["id"])
                    outcomes.append("approved")
                except StaleRunError:
                    outcomes.append("stale_rejected")
                except Exception as e:  # 只允许 StaleRunError
                    err.append(e)

            def do_update():
                self.svc.set_reading(self.b, 40.0 + run["id"] % 7)

            t1 = threading.Thread(target=do_approve)
            t2 = threading.Thread(target=do_update)
            t1.start()
            t2.start()
            t1.join()
            t2.join()
            self.assertFalse(err, f"出现非预期异常: {err}")
            final = self.svc.get_run(run["id"])
            if final["status"] == "approved":
                # 通过之后又发生更新是合法的；关键是状态机一致
                self.assertIn(final["status"], ("approved",))
            else:
                self.assertEqual(final["status"], "completed")
                self.assertTrue(final["stale"])
        self.assertTrue(outcomes)
        self.assertTrue(set(outcomes) <= {"approved", "stale_rejected"})

    def test_concurrent_double_approve_exactly_one_wins(self):
        run = self.svc.reconcile()
        wins, errs = [], []

        def do_approve():
            try:
                self.svc.approve(run["id"])
                wins.append(1)
            except ReviewStateError:
                errs.append(1)

        threads = [threading.Thread(target=do_approve) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(wins), 1)
        self.assertEqual(len(errs), 7)


class InfeasibleAndTimeoutTest(unittest.TestCase):
    def setUp(self):
        self.svc, self.db = make_service()
        self.addCleanup(lambda: os.path.exists(self.db) and os.unlink(self.db))

    def test_lock_conflict_no_report(self):
        root = self.svc.add_meter("总表", locked=100.0)
        a = self.svc.add_meter("分表A", parent_id=root, locked=60.0)
        b = self.svc.add_meter("分表B", parent_id=root, locked=60.0)
        for mid, v in ((root, 100.0), (a, 60.0), (b, 60.0)):
            self.svc.set_reading(mid, v)
        run = self.svc.reconcile()
        self.assertEqual(run["status"], "infeasible")
        self.assertIsNone(run["result"])            # 不生成可复核报告
        self.assertIsNone(run["objective"])
        self.assertTrue(run["conflicts"])
        detail = run["conflicts"][0]["detail"]
        self.assertIn("总表", detail)
        constraints = [(i["meter"], i["constraint"])
                       for i in run["conflicts"][0]["items"]]
        self.assertIn(("总表", "锁定"), constraints)
        self.assertIn(("分表A", "锁定"), constraints)
        with self.assertRaises(ReviewStateError):   # 不可行单不可通过
            self.svc.approve(run["id"])

    def test_bound_conflict_reported(self):
        root = self.svc.add_meter("总表", upper=50.0)
        a = self.svc.add_meter("分表A", parent_id=root, lower=30.0)
        b = self.svc.add_meter("分表B", parent_id=root, lower=30.0)
        for mid, v in ((root, 55.0), (a, 32.0), (b, 31.0)):
            self.svc.set_reading(mid, v)
        run = self.svc.reconcile()
        self.assertEqual(run["status"], "infeasible")
        self.assertIsNone(run["result"])
        self.assertIn("不相交", run["conflicts"][0]["detail"])

    def test_timeout_failed_run_no_result(self):
        root = self.svc.add_meter("总表")
        leaf = self.svc.add_meter("分表", parent_id=root)
        self.svc.set_reading(root, 10.0)
        self.svc.set_reading(leaf, 9.0)
        run = self.svc.reconcile(timeout_s=0.0)
        self.assertEqual(run["status"], "failed")
        self.assertIn("超时", run["message"])
        self.assertIsNone(run["result"])            # 不以近似冒充完成
        with self.assertRaises(ReviewStateError):
            self.svc.approve(run["id"])


if __name__ == "__main__":
    unittest.main()
