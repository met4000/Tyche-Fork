import unittest
from tyche.language import *


class TestLanguage(unittest.TestCase):
    def setUp(self):
        self.x = Concept('my_X')
        self.y = Concept('my_Y')
        self.z = Concept('my_Z')
        self.r = Role('my_R')

    def tearDown(self):
        pass

    def test_concept_symbols(self):
        """
        Tests concept symbol validity checking.
        """
        # Valid names, should not error

        self.assertIsInstance(Concept("a"), Concept)

        self.assertIsInstance(Concept("abc"), Concept)
        self.assertIsInstance(Concept("aBc"), Concept)

        self.assertIsInstance(Concept("a2C"), Concept)

        self.assertIsInstance(Concept("ab_cd"), Concept)
        self.assertIsInstance(Concept("a_2c__D"), Concept)


        # Invalid names, should error

        # `None` symbol
        with self.assertRaisesRegex(ValueError, "symbols cannot be None"):
            Concept(None)

        # empty string symbol
        with self.assertRaisesRegex(ValueError, "symbols cannot be empty strings"):
            self.assertNotIsInstance(Concept(""), Concept)

        # non-lowercase starting character
        starting_char_error_type = ValueError
        starting_char_error_regex = "symbols must start with a lowercase letter"
        with self.assertRaisesRegex(starting_char_error_type, starting_char_error_regex):
            self.assertNotIsInstance(Concept("_a1"), Concept)
        with self.assertRaisesRegex(starting_char_error_type, starting_char_error_regex):
            self.assertNotIsInstance(Concept("1a"), Concept)
        with self.assertRaisesRegex(starting_char_error_type, starting_char_error_regex):
            self.assertNotIsInstance(Concept("A1"), Concept)
        
        # invalid characters
        invalid_char_error_type = ValueError
        invalid_char_error_regex = "symbols can only contain alpha-numeric or underscore characters"
        with self.assertRaisesRegex(invalid_char_error_type, invalid_char_error_regex):
            self.assertNotIsInstance(Concept("abc!"), Concept)
        with self.assertRaisesRegex(invalid_char_error_type, invalid_char_error_regex):
            self.assertNotIsInstance(Concept("ab3_\u22A4"), Concept)

        # multiple problems
        with self.assertRaises((starting_char_error_type, invalid_char_error_type)):
            self.assertNotIsInstance(Concept("_!"), Concept)

    def test_equals(self):
        """
        Tests equality checking of formulas.
        """
        # Constants
        self.assertEqual(ALWAYS, ALWAYS)
        self.assertNotEqual(ALWAYS, NEVER)
        self.assertNotEqual(ALWAYS, self.y)

        # Atoms
        self.assertEqual(self.x, self.x)
        self.assertNotEqual(self.x, self.y)

        # Conditionals
        cond1 = Conditional(self.x, self.y, self.z)
        cond2 = Conditional(self.x, ALWAYS, NEVER)
        cond3 = Conditional(self.x, self.y, self.z)
        self.assertEqual(cond1, cond3)
        self.assertNotEqual(cond1, cond2)
        self.assertNotEqual(cond1, NEVER)

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
        self.assertEqual("\u22A4", str(ALWAYS))
        self.assertEqual("\u22A5", str(NEVER))

        a = Concept("a")
        b = Concept("b")
        abc = Concept("abc")

        self.assertEqual("a", str(a))
        self.assertEqual("b", str(b))
        self.assertEqual("abc", str(abc))
        self.assertEqual("(a ? b : abc)", str(b.when(a).otherwise(abc)))
        self.assertEqual("\u00ACabc", str(abc.complement()))
        self.assertEqual("(a \u2227 b)", str(a & b))
        self.assertEqual("(b \u2228 a)", str(b | a))
        self.assertEqual("((a \u2228 abc) \u2227 b \u2227 \u00ACa)", str((a | abc) & (b & a.complement())))
        self.assertEqual("[x](abc)", str(Expectation("x", abc)))
        self.assertEqual("Exists[x]", str(Exists("x")))

    def test_eval_constants(self):
        """
        Tests the evaluation of constant formulas.
        """
        context = EmptyContext()
        flip = Constant("flip", 0.5)

        self.assertEqual(1, context.eval(ALWAYS))
        self.assertEqual(0.5, context.eval(flip))
        self.assertEqual(0, context.eval(NEVER))

        self.assertEqual(1, context.eval(ALWAYS | NEVER))
        self.assertEqual(0, context.eval(ALWAYS & NEVER))
        self.assertEqual(1, context.eval(ALWAYS | flip))
        self.assertAlmostEqual(0.5, context.eval(NEVER | flip))
        self.assertAlmostEqual(0.5 * 0.5, context.eval(flip & flip))
        self.assertAlmostEqual(0.5 * 0.5, context.eval((flip & flip) | (NEVER & flip)))
        self.assertAlmostEqual(
            1 - ((1 - 0.5 * 0.5) * (1 - 1 * 0.5)),
            context.eval((flip & flip) | (ALWAYS & flip))
        )
