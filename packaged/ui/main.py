from flask import Flask, request, render_template_string
from turbo_flask import Turbo

service = Flask(__name__)
turbo = Turbo(service)

temp_logs = []

def logs_list(can_stream=False):
	global temp_logs
	return render_template_string('''
	<head>{{turbo()}}</head><body>
	<h1>Streaming: {{can_stream}} </h1>
	<ul>
	{% for log in logs %}
		<li> {{log}} </li>
	{% endfor %}
	</ul>
	</body>
''', can_stream=can_stream, logs=temp_logs)

@service.get("/")
def root():
	if turbo.can_stream():
		return turbo.stream([
			turbo.update(logs_list(can_stream=True), target='logs')
		])
	return logs_list()

@service.get("/worker/update/<_id>")
def worker_update(_id):
	global temp_logs
	if 'log' in request.args:
		temp_logs += [f'{_id}: {request.args["log"]}']
		if turbo.can_stream():
			with service.app_context():
				turbo.push(
					turbo.update(logs_list(can_stream=True), target='logs')
				)
	print("Worker", _id, "has updated")
	return "", 200


if __name__ == '__main__':
	service.run(
		host=os.environ.get("HOST", "0.0.0.0"),
		port=os.environ.get("PORT", 5000),
	)
