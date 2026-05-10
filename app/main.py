from flask import Flask, request, render_template

app = Flask(__name__)

# 2. Define a route and a view function
@app.route('/')
def hello_world():
    return 'Hello, World!'

# 3. Run the development server
if __name__ == '__main__':
    app.run(debug=True)