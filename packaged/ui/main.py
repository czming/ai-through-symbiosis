from flask import Flask, request, render_template_string
from flask_socketio import SocketIO, emit, send


_service = Flask(__name__)
service = SocketIO(_service)

@service.on('connect')
def on_connect(auth):
	emit('connected', {'auth': auth})

@service.on('disconnect')
def on_disconnect():
	print('discsonnected')

@_service.get("/")
def update_home_page():
	return render_template_string('''
<body>
<h1> HSV Train </h1>
<img src='/static/hist-real.png' id='real'/>
<br>
<h1> HSV Train Iterative</h1>
<img src='/static/hist-iter.png' id='iter'/>
<script>
let counter = 0;
setInterval(
	() => {
		['real', 'iter'].forEach(
			(id) => {
				document.getElementById(id).src = `/static/hist-${id}.png?${counter}`;
			}
		)
	}
, 1000)
</script>
</body>
''')

if __name__ == '__main__':
	service.run(
		host=os.environ.get("HOST", "0.0.0.0"),
		port=os.environ.get("PORT", 5000),
	)
