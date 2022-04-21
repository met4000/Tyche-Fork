import unittest
from tyche.language import *


class TestSubIndividual(Individual):
    y: Probability
    z: Probability

    def __init__(self, y: Probability, z: Probability):
        super().__init__()
        self.y = y
        self.z = z


class TestIndividual(Individual):
    x: Probability
    sub1: TestSubIndividual
    sub2: TestSubIndividual

    def __init__(self, x: Probability):
        super().__init__()
        self.x = x
        self.sub1 = TestSubIndividual(0.5, 0.5)
        self.sub2 = TestSubIndividual(0.2, 0.8)

    @Role
    def sub(self):
        dist = RoleProbabilityDistribution()
        dist.add(self.sub1, 0.4)
        dist.add(self.sub2, 0.6)
        return dist


class TestLanguage(unittest.TestCase):
    def setUp(self):
        self.x = Atom('my_X')
        self.y = Atom('my_Y')
        self.z = Atom('my_Z')
        self.r = 'my_R'

    def tearDown(self):
        pass

    def test_equals(self):
        """
        Tests equality checking of formulas.
        """

        # Constants
        self.assertEqual(always, always)
        self.assertNotEqual(always, never)
        self.assertNotEqual(always, self.y)

        # Atoms
        self.assertEqual(self.x, self.x)
        self.assertNotEqual(self.x, self.y)

        # Conditionals
        cond1 = Conditional(self.x, self.y, self.z)
        cond2 = Conditional(self.x, always, never)
        cond3 = Conditional(self.x, self.y, self.z)
        self.assertEqual(cond1, cond3)
        self.assertNotEqual(cond1, cond2)
        self.assertNotEqual(cond1, never)

        # Marginal Expectations
        marg1 = Expectation(self.r, self.x)
        marg2 = Expectation(self.r, self.y)
        marg3 = Expectation(self.r, self.x)
        self.assertEqual(marg1, marg3)
        self.assertNotEqual(marg1, marg2)
        self.assertNotEqual(marg1, cond1)

    def test_str(self):
        """
        Tests the conversion of formulas to strings.
        """
        self.assertEqual("\u22A4", str(always))
        self.assertEqual("\u22A5", str(never))

        a = Atom("a")
        b = Atom("b")
        abc = Atom("abc")

        self.assertEqual("a", str(a))
        self.assertEqual("b", str(b))
        self.assertEqual("abc", str(abc))
        self.assertEqual("(a ? b : abc)", str(b.when(a).otherwise(abc)))
        self.assertEqual("\u00ACabc", str(abc.complement()))
        self.assertEqual("(a \u2227 b)", str(a & b))
        self.assertEqual("(b \u2228 a)", str(b | a))
        self.assertEqual("((a \u2228 abc) \u2227 (b \u2227 \u00ACa))", str((a | abc) & (b & a.complement())))
        self.assertEqual("(\U0001D53C_X. abc)", str(Expectation("X", abc)))

    def test_eval(self):
        """
        Tests the evaluation of the probability of formulas.
        """
        individual = TestIndividual(0.5)

        x = Atom("x")
        y = Atom("y")
        z = Atom("z")

        self.assertAlmostEqual(0.5, x.eval(individual))
        self.assertAlmostEqual(0.4 * 0.5 + 0.6 * 0.2, Expectation("sub", y).eval(individual))
        self.assertAlmostEqual(0.4 * 0.5 + 0.6 * 0.8, Expectation("sub", z).eval(individual))
        self.assertAlmostEqual(
            0.4 * (0.5 * 0.5) + 0.6 * (0.2 * 0.8),
            Expectation("sub", y & z).eval(individual)
        )
