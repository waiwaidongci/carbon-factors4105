"""求解器数学正确性测试：非均摊最优解、KKT、守恒、限值、锁定、冲突、超时。"""
import math
import random
import unittest

from meter_recon.solver import (MeterNode, SolverError, SolverTimeout, solve)


def star(y_root=100.0, w_root=1.0, ya=40.0, wa=100.0, yb=40.0, wb=1.0, **kw):
    """根 + 两叶的三表星形树，可用解析解对照。"""
    return [
        MeterNode(1, "总表", None, y_root, w_root,
                  lower=kw.get("lo_r"), upper=kw.get("hi_r"), lock=kw.get("lock_r")),
        MeterNode(2, "分表A", 1, ya, wa,
                  lower=kw.get("lo_a"), upper=kw.get("hi_a"), lock=kw.get("lock_a")),
        MeterNode(3, "分表B", 1, yb, wb,
                  lower=kw.get("lo_b"), upper=kw.get("hi_b"), lock=kw.get("lock_b")),
    ]


def assert_kkt(tc, nodes, res, tol=1e-6):
    """凸 QP 的 KKT 充要条件：满足即全局最优（非均摊等任何解都须满足）。"""
    tc.assertEqual(res.status, "optimal")
    by_id = {nd.meter_id: nd for nd in nodes}
    children = {}
    for nd in nodes:
        if nd.parent_id is not None:
            children.setdefault(nd.parent_id, []).append(nd.meter_id)
    x, lam = res.x, res.lam
    scale = max(1.0, max(abs(v) for v in x.values()))
    locked = {nd.meter_id for nd in nodes if nd.lock is not None}
    active = {mid: side for mid, side in res.active_bounds}
    for nd in nodes:
        i = nd.meter_id
        # 守恒：每个父表等于子表之和
        if i in children:
            resid = x[i] - sum(x[c] for c in children[i])
            tc.assertAlmostEqual(resid, 0.0, delta=1e-8 * scale,
                                 msg=f"{nd.name} 守恒残差")
        # 限值
        lo = -math.inf if nd.lower is None else nd.lower
        hi = math.inf if nd.upper is None else nd.upper
        tc.assertGreaterEqual(x[i], lo - 1e-7 * scale, f"{nd.name} 越下限")
        tc.assertLessEqual(x[i], hi + 1e-7 * scale, f"{nd.name} 越上限")
        # 锁定
        if nd.lock is not None:
            tc.assertAlmostEqual(x[i], nd.lock, delta=1e-9 * scale)
        # 平稳性 / 互补松弛
        r = 2.0 * nd.weight * (x[i] - nd.reading)
        if i in children:
            r += lam[i]
        if nd.parent_id is not None:
            r -= lam[nd.parent_id]
        if i in locked:
            continue
        if i in active:
            if active[i] == "upper":
                tc.assertLessEqual(r, tol * max(1.0, abs(r)),
                                   f"{nd.name} 上限乘子符号")
            else:
                tc.assertGreaterEqual(r, -tol * max(1.0, abs(r)),
                                      f"{nd.name} 下限乘子符号")
        else:
            tc.assertAlmostEqual(r, 0.0, delta=tol * max(1.0, abs(r)),
                                 msg=f"{nd.name} 平稳性")


class NonUniformOptimalTest(unittest.TestCase):
    """非均摊最优解：调整量与权重成反比，绝不平均分摊。"""

    def test_weighted_allocation_matches_closed_form(self):
        nodes = star()  # 总表100, 分表40+40, 权重 1/100/1
        res = solve(nodes)
        d = -20.0 / 2.01  # 解析解：根残差
        self.assertAlmostEqual(res.x[1], 100.0 + d, places=7)
        self.assertAlmostEqual(res.x[2], 40.0 - d / 100.0, places=7)
        self.assertAlmostEqual(res.x[3], 40.0 - d, places=7)
        # 非均摊：高权重的分表A几乎不动，低权重的分表B承担绝大部分
        self.assertLess(abs(res.x[2] - 40.0), 0.15)
        self.assertGreater(abs(res.x[3] - 40.0), 9.0)
        uniform = 20.0 / 3.0
        self.assertGreater(abs(abs(res.x[3] - 40.0) - uniform), 2.0)
        assert_kkt(self, nodes, res)

    def test_objective_is_weighted_sum(self):
        nodes = star()
        res = solve(nodes)
        expect = sum(nd.weight * (res.x[nd.meter_id] - nd.reading) ** 2
                     for nd in nodes)
        self.assertAlmostEqual(res.objective, expect, places=8)

    def test_leaf_only_emissions(self):
        nodes = star()
        res = solve(nodes)
        # 只按叶表计：总表不计入
        self.assertAlmostEqual(res.leaf_total, res.x[2] + res.x[3], places=8)
        self.assertNotAlmostEqual(res.leaf_total,
                                  res.x[1] + res.x[2] + res.x[3], places=4)


class BoundsAndLockTest(unittest.TestCase):
    def test_upper_bound_forces_reallocation(self):
        nodes = star(hi_b=45.0)
        res = solve(nodes)
        self.assertAlmostEqual(res.x[3], 45.0, places=9)  # 顶到上限
        # 解析解：x_a = 8110/202, x_r = x_a + 45
        self.assertAlmostEqual(res.x[2], 8110.0 / 202.0, places=7)
        self.assertAlmostEqual(res.x[1], 8110.0 / 202.0 + 45.0, places=7)
        self.assertIn((3, "upper"), res.active_bounds)
        assert_kkt(self, nodes, res)

    def test_lock_respected(self):
        nodes = star(lock_b=42.0)
        res = solve(nodes)
        self.assertEqual(res.x[3], 42.0)
        self.assertAlmostEqual(res.x[2], 8116.0 / 202.0, places=7)
        assert_kkt(self, nodes, res)

    def test_lock_and_bounds_combined(self):
        # 锁定 x_b=42；无约束时 x_a=8116/202≈40.178，上限 40.1 生效
        nodes = star(lock_b=42.0, hi_a=40.1)
        res = solve(nodes)
        self.assertEqual(res.x[3], 42.0)
        self.assertAlmostEqual(res.x[2], 40.1, places=9)
        self.assertAlmostEqual(res.x[1], 82.1, places=7)
        self.assertIn((2, "upper"), res.active_bounds)
        assert_kkt(self, nodes, res)


class InfeasibleTest(unittest.TestCase):
    def test_lock_conflict_parent_vs_children(self):
        # 父表锁定 100，子表锁定 60+60=120：守恒不可能
        nodes = star(lock_r=100.0, lock_a=60.0, lock_b=60.0)
        res = solve(nodes)
        self.assertEqual(res.status, "infeasible")
        self.assertIsNone(res.x)
        self.assertTrue(res.conflicts)
        text = res.conflicts[0].detail + "".join(
            i.constraint + i.value for i in res.conflicts[0].items)
        self.assertIn("锁定", text)
        self.assertIn("120", text)

    def test_bound_conflict_parent_upper_vs_children_lower(self):
        # 父表上限 50 < 子表下限合计 30+30
        nodes = star(hi_r=50.0, lo_a=30.0, lo_b=30.0)
        res = solve(nodes)
        self.assertEqual(res.status, "infeasible")
        self.assertTrue(res.conflicts)
        items = {(i.meter, i.constraint) for i in res.conflicts[0].items}
        self.assertIn(("总表", "上限"), items)
        self.assertIn(("分表A", "下限"), items)
        self.assertIn(("分表B", "下限"), items)

    def test_lock_out_of_bounds(self):
        nodes = [MeterNode(1, "总表", None, 10.0, 1.0, lock=99.0),
                 MeterNode(2, "分表", 1, 10.0, 1.0, upper=50.0, lock=60.0)]
        res = solve(nodes)
        self.assertEqual(res.status, "infeasible")
        self.assertIn("锁定", res.conflicts[0].detail)

    def test_lower_greater_than_upper(self):
        nodes = [MeterNode(1, "总表", None, 10.0, 1.0),
                 MeterNode(2, "分表", 1, 10.0, 1.0, lower=20.0, upper=10.0)]
        res = solve(nodes)
        self.assertEqual(res.status, "infeasible")


class ValidationTest(unittest.TestCase):
    def test_weight_must_be_positive(self):
        with self.assertRaises(SolverError):
            solve([MeterNode(1, "a", None, 1.0, 0.0)])
        with self.assertRaises(SolverError):
            solve([MeterNode(1, "a", None, 1.0, -2.0)])

    def test_tree_must_be_single_connected(self):
        with self.assertRaises(SolverError):  # 两个根
            solve([MeterNode(1, "a", None, 1.0, 1.0),
                   MeterNode(2, "b", None, 1.0, 1.0)])
        with self.assertRaises(SolverError):  # 父表不存在
            solve([MeterNode(1, "a", None, 1.0, 1.0),
                   MeterNode(2, "b", 99, 1.0, 1.0)])

    def test_single_node(self):
        res = solve([MeterNode(1, "a", None, 10.0, 1.0)])
        self.assertEqual(res.status, "optimal")
        self.assertAlmostEqual(res.x[1], 10.0)
        self.assertAlmostEqual(res.leaf_total, 10.0)
        res2 = solve([MeterNode(1, "a", None, 10.0, 1.0, upper=5.0)])
        self.assertAlmostEqual(res2.x[1], 5.0)


class TimeoutTest(unittest.TestCase):
    def test_timeout_raises_and_yields_nothing(self):
        nodes = star()
        with self.assertRaises(SolverTimeout):
            solve(nodes, timeout_s=0.0)

    def test_iteration_limit(self):
        from meter_recon.solver import IterationLimitExceeded
        with self.assertRaises(IterationLimitExceeded):
            solve(star(), max_iterations=0)


class ActiveSetRepairTest(unittest.TestCase):
    """主动集把父表固定到下限会与锁定子表矛盾，必须释放修复而非报错。"""

    def test_inconsistent_active_set_is_repaired(self):
        # 无约束解 x_root=-51.15<0、x_b=-98.85<0，批量加限后
        # root 固定 0 与锁定子表 47.7 矛盾；正确解：释放 root，x_root=47.7
        nodes = [
            MeterNode(1, "总表", None, -50.0, 1.0, lower=0.0),
            MeterNode(2, "锁定子表", 1, 47.7, 1.0, lock=47.7),
            MeterNode(3, "分表", 1, -100.0, 1.0, lower=0.0),
        ]
        res = solve(nodes)
        self.assertEqual(res.status, "optimal")
        self.assertAlmostEqual(res.x[1], 47.7, places=7)
        self.assertEqual(res.x[2], 47.7)
        self.assertAlmostEqual(res.x[3], 0.0, places=9)
        self.assertIn((3, "lower"), res.active_bounds)
        self.assertNotIn(1, {m for m, _ in res.active_bounds})
        assert_kkt(self, nodes, res)


class RandomTreeKktTest(unittest.TestCase):
    def test_random_trees_satisfy_kkt_and_conservation(self):
        rng = random.Random(42)
        for trial in range(20):
            n = rng.randint(2, 120)
            nodes = [MeterNode(1, "m1", None, rng.uniform(50, 150),
                               rng.uniform(0.5, 5.0))]
            for i in range(2, n + 1):
                parent = rng.randrange(1, i)
                kw = {}
                if rng.random() < 0.15:
                    v = rng.uniform(40, 160)
                    kw["lock"] = v
                    kw["reading"] = v + rng.uniform(-5, 5)
                else:
                    kw["reading"] = rng.uniform(40, 160)
                if rng.random() < 0.2 and "lock" not in kw:
                    kw["lower"] = 0.0
                nodes.append(MeterNode(i, f"m{i}", parent, kw.pop("reading"),
                                       rng.uniform(0.5, 5.0), **kw))
            res = solve(nodes)
            if res.status == "infeasible":
                continue  # 随机锁定可能冲突，另行专测
            assert_kkt(self, nodes, res)
            self.assertLess(res.max_residual, 1e-6)


if __name__ == "__main__":
    unittest.main()
