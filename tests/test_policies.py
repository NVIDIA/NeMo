import unittest

from nemo.utils.lr_policies import SquareAnnealing, CosineAnnealing, \
    WarmupAnnealing


class TestPolicies(unittest.TestCase):
    def test_square(self):
        policy = SquareAnnealing(100)
        lr1, lr2, lr3 = (policy(1e-3, x, 0) for x in (0, 10, 20))
        self.assertTrue(lr1 >= lr2)
        self.assertTrue(lr2 >= lr3)
        self.assertTrue(lr1 - lr2 >= lr2 - lr3)

    def test_working(self):
        total_steps = 1000
        lr_policy_cls = [SquareAnnealing, CosineAnnealing, WarmupAnnealing]
        lr_policies = [p(total_steps=total_steps) for p in lr_policy_cls]

        for step in range(1000):
            for p in lr_policies:
                assert p(1e-3, step, 0) > 0

    def test_warmup(self):
        policy = SquareAnnealing(100, warmup_ratio=0.5)
        lr1, lr2, lr3 = (policy(1e-3, x, 0) for x in (0, 50, 100))
        self.assertTrue(lr1 < lr2)
        self.assertTrue(lr2 > lr3)
