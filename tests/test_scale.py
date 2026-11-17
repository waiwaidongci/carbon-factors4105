"""规模测试：万表时内存随表数线性增长、时间可控、结果精确。"""
import random
import time
import tracemalloc
import unittest

from meter_recon.solver import MeterNode, solve


def random_tree(n, seed=7):
    rng = random.Random(seed)
    nodes = [MeterNode(1, "m1", None, rng.uniform(900, 1100),
                       rng.uniform(0.5, 2.0))]
    for i in range(2, n + 1):
        nodes.append(MeterNode(i, f"m{i}", rng.randrange(1, i),
                               rng.uniform(40, 160), rng.uniform(0.5, 2.0)))
    return nodes


class ScaleTest(unittest.TestCase):
    def test_10k_meters_linear_memory_and_exact_result(self):
        peaks = {}
        times = {}
        for n in (5000, 10000):
            nodes = random_tree(n)
            tracemalloc.start()
            t0 = time.monotonic()
            res = solve(nodes, timeout_s=60.0)
            times[n] = time.monotonic() - t0
            peaks[n] = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
            self.assertEqual(res.status, "optimal")
            # 精确解：守恒残差近零，不是近似冒充
            self.assertLess(res.max_residual, 1e-5)
        # 内存随表数线性增长：表数翻倍，峰值内存不超过 2.6 倍
        ratio = peaks[10000] / peaks[5000]
        self.assertLess(ratio, 2.6, f"内存增长非线性: {ratio:.2f}x")
        # 万表绝对内存与耗时上限（纯 Python 实现也应远低于此）
        self.assertLess(peaks[10000], 80 * 1024 * 1024)
        self.assertLess(times[10000], 30.0)
        print(f"\n5k: {peaks[5000]/1e6:.1f}MB {times[5000]:.2f}s | "
              f"10k: {peaks[10000]/1e6:.1f}MB {times[10000]:.2f}s | "
              f"内存比 {ratio:.2f}x")

    def test_10k_with_bounds_and_locks(self):
        rng = random.Random(11)
        parents = [None] + [rng.randrange(1, i) for i in range(2, 10001)]
        has_child = set(p for p in parents if p is not None)
        nodes = [MeterNode(1, "m1", None, 10000.0, 1.0)]
        for i in range(2, 10001):
            kw = {}
            if i not in has_child and rng.random() < 0.05:
                kw["lock"] = rng.uniform(40, 160)   # 只锁叶表，避免锁定间守恒冲突
            elif rng.random() < 0.3:
                kw["lower"] = 0.0
            nodes.append(MeterNode(i, f"m{i}", parents[i - 1],
                                   rng.uniform(40, 160),
                                   rng.uniform(0.5, 2.0), **kw))
        t0 = time.monotonic()
        res = solve(nodes, timeout_s=60.0)
        self.assertEqual(res.status, "optimal")
        self.assertLess(res.max_residual, 1e-5)
        self.assertLess(time.monotonic() - t0, 30.0)
        # 锁定表纹丝不动
        locks = {nd.meter_id: nd.lock for nd in nodes if nd.lock is not None}
        for mid, lv in locks.items():
            self.assertEqual(res.x[mid], lv)


if __name__ == "__main__":
    unittest.main()
