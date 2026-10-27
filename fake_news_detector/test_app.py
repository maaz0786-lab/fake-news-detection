import unittest

from app import app


class FakeNewsAppTests(unittest.TestCase):
    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_home_page_loads(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Fake News Detector", response.data)

    def test_prediction_is_shown_for_submitted_text(self):
        response = self.client.post(
            "/",
            data={"article_text": "A shocking secret miracle trick guarantees instant wealth"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Model prediction:", response.data)
        self.assertTrue(b"FAKE" in response.data or b"REAL" in response.data)

    def test_blank_text_does_not_show_prediction(self):
        response = self.client.post("/", data={"article_text": "   "})
        self.assertEqual(response.status_code, 200)
        self.assertNotIn(b"Model prediction:", response.data)


if __name__ == "__main__":
    unittest.main()
