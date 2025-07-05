from flask import Flask
from flask_restful import Api, Resource

app = Flask(__name__)
api = Api(app)

names = { "tim" : {
                    "age": 20,
                    "gender": "male"},
         "jane" : {
                    "age": 22,
                    "gender" : "male"}}

class HelloWorld(Resource):
    def get(self, name):
        return names[name]
    
api.add_resource(HelloWorld, '/HelloWorld/<string:name>')

if __name__ == '__main__':
    app.run(debug=True, port=5000)  # Flask will now run on http://
    