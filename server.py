from flask import Flask, render_template, request, redirect, url_for, redirect
import json
app = Flask(__name__, template_folder='templates', static_folder='static')
content = json.load(open('configs/config.json'))
@app.route('/')
def index():
    return "Hello, World!"


@app.route('/config', methods=['POST', 'GET'])
def config():
    if request.method == 'POST':
        content['EYE_LEFT'] = request.form['EYE_LEFT']
        content['EYE_RIGHT'] = request.form['EYE_RIGHT']
        content['EYE_UP'] = request.form['EYE_UP']
        content['EYE_DOWN'] = request.form['EYE_DOWN']
        with open("configs/config.json", "w") as fp:
            json.dump(content,fp) 
        return redirect('/')
    return render_template('config.html', content=content)

if __name__ == '__main__':
    app.run()

