import unittest
import os
import json
from JsonHandler import JsonHandler


class TestJsonHandler(unittest.TestCase):
    def setUp(self):
        self.jsonHandler = JsonHandler('test.json')

    def tearDown(self):
        if os.path.exists('test.json'):
            os.remove('test.json')

    def test_read_json(self):
        with open('test.json', 'w') as f:
            json.dump({"key": "value"}, f)

        data = self.jsonHandler.read_json()
        self.assertEqual(data, {"key": "value"})

    def test_write_json(self):
        self.jsonHandler.write_json({"key": "value"})

        with open('test.json', 'r') as f:
            data = json.load(f)

        self.assertEqual(data, {"key": "value"})

    def test_update_json(self):
        with open('test.json', 'w') as f:
            json.dump({"key": "value"}, f)

        self.jsonHandler.update_json({"new_key": "new_value"})

        with open('test.json', 'r') as f:
            data = json.load(f)

        self.assertEqual(data, {"key": "value", "new_key": "new_value"})


if __name__ == '__main__':
    unittest.main()
