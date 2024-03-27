import unittest
import json
from flask_api import app
import pandas as pd


class FlaskAppTest(unittest.TestCase):

    def setUp(self):
        app.testing = True
        self.app = app.test_client()

    def test_home_page(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data.decode('utf-8'), 'Welcome to the credit scoring API')

    def test_predict_endpoint(self):
        df = pd.read_csv('./data/data.csv')
        test_data_json = df.drop(columns=['SK_ID_CURR']).to_json(orient='records')
        headers = {'Content-Type': 'application/json'}
        response = self.app.post('/predict', data=json.dumps({'test_data': test_data_json}), headers=headers)
        data = json.loads(response.data.decode('utf-8'))

        self.assertEqual(response.status_code, 200)
        self.assertTrue('prediction' in data)

if __name__ == '__main__':
    unittest.main()