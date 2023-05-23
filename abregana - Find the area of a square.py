import unittest

def calculate_square_area(side_length):
    return side_length ** 2

class SquareAreaTest(unittest.TestCase):
    def test_calculate_square_area(self):
        side_length = 5
        expected_area = 25
        self.assertEqual(calculate_square_area(side_length), expected_area)

if __name__ == '__main__':
    unittest.main()
