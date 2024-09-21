import flask
import abc
import flask_cors


class Service(abc.ABC):
	def __init__(self, name, function, inputs_mapping, methods=["POST"]):
		self.__name = name
		self.__function = function
		self.__inputs_mapping = {i: inputs_mapping.get(i, []) for i in ["args", "json", 'files', 'form']}
		self.__methods = methods
	
	def create_service(self, prefix="", app=None, init_cors=False, headers=dict()):
		if app is None:
			app = flask.Flask(self.__name)
		if init_cors == True:
			flask_cors.CORS(app)
		# TODO: add in other HTTP based data input schemes
		def handle():
			def __handle_route(): 
				response = flask.make_response(
					self.__function(
						**{i: flask.request.args[i] for i in self.__inputs_mapping['args']},
						**{i: flask.request.json[i] for i in self.__inputs_mapping['json']},
						**{i: flask.request.files[i] for i in self.__inputs_mapping['files']},
						**{i: flask.request.form[i] for i in self.__inputs_mapping['form']},
						)
					)
				for i in headers:
					response.headers[i] = headers[i]
				return response

			__handle_route.__name__ = self.__name
			return __handle_route
		app.route(f"{prefix}/{self.__name}", methods=self.__methods)(handle())
		return app
