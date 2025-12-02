import unittest
from vllm_hybrid_parallel_engine import extract_result

class TestExtractResult(unittest.TestCase):
    def test_simple_number(self):
        text = "<FINAL> 42"
        self.assertEqual(extract_result(text), 42.0)

    def test_decimal_number(self):
        text = "<FINAL> 42.5"
        self.assertEqual(extract_result(text), 42.5)

    def test_negative_number(self):
        text = "<FINAL> -42.5"
        self.assertEqual(extract_result(text), -42.5)

    def test_boxed_number(self):
        text = "<FINAL> \n\n\n\n $$\\boxed{7000}$$"
        self.assertEqual(extract_result(text), 7000.0)

    def test_boxed_decimal(self):
        text = "<FINAL> $$\\boxed{123.45}$$"
        self.assertEqual(extract_result(text), 123.45)

    def test_no_number(self):
        text = "<FINAL> No number here"
        self.assertIsNone(extract_result(text))

    def test_multiple_numbers(self):
        text = "<FINAL> 42 and then 100"
        self.assertEqual(extract_result(text), 42.0)  # Should extract the first number

    def test_irrelevant_text(self):
        text = "Some irrelevant text"
        self.assertIsNone(extract_result(text))

    def test_complex_expression(self):
        text = "<FINAL> Result is $$\\boxed{-123.45}$$ with explanation"
        self.assertEqual(extract_result(text), -123.45)

if __name__ == "__main__":
    unittest.main()