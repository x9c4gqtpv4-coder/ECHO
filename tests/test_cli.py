import unittest
from unittest.mock import patch

from batch_color.cli import main


class CliTests(unittest.TestCase):
    @patch("batch_color.cli._doctor", return_value=0)
    def test_doctor_command(self, doctor) -> None:
        self.assertEqual(main(["doctor"]), 0)
        doctor.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
