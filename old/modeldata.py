import pandas as pd

class NotModifiableError(Exception):
	'''Exception raised for when a descriptive model variable
	(e.g. model id, product) is called to be modified
	by the user.
	'''

	def __init__(self, key):
		self.key = key
		message_str = "Modification or deletion of built-in identifier '" + str(key) + "' not allowed."
		self.message = message_str
		super().__init__(self.message)

class ModelRunSeries:
	'''A container to hold and access model data.

    Allows indexing to access model identifiers and custom parameters,
    and modification of custom parameters. Allows iteration
    to loop over file names. 
    '''

	def __init__(self, model, product, init_time, forecast_length, forecast_interval, file_paths, **kwargs):
		'''Create a model run series container.

		Inputs: Model name, product name (e.g. GFS 0.25 deg), model run start
		time, length of forecast (in hours), hours between each forecast output
		(in hours), paths to model files.

		Internally Generated: Runs, a list of SingleRunData objects 

		'''
		self.model = model
		self.product = product
		self.init_time = init_time
		self.forecast_length = forecast_length
		self.forecast_interval = forecast_interval
		self.file_paths = file_paths
		self.runs = self._index_model_files()

		self._builtins = ['model', 
						  'product', 
						  'init_time', 
						  'forecast_length',
						  'forecast_interval', 
						  'file_paths',
						  ]

		self._custom_parameters = kwargs

	def _index_model_files(self):
		runs = []
		forecast_hour = 0

		for file in self.file_paths:
			valid_time =  self.init_time + pd.Timedelta(hours=forecast_hour)
			run_data = SingleRunData(self.model, self.product, self.init_time, 
									 valid_time, forecast_hour, file)
			runs.append(run_data)
			forecast_hour = forecast_hour + self.forecast_interval

		return runs

	def __iter__(self):
		self._iter_loc = 0

		return self

	def __next__(self):
		iter_loc = self._iter_loc

		if iter_loc > len(self.runs):
			raise StopIteration

		selected_run = self.runs[iter_loc]

		self._iter_loc = iter_loc + 1
		return selected_run

	def __getitem__(self, key):

		if key in self._builtins:
			return getattr(self, key, None)
		else:
			if key in self._custom_parameters:
				return self._custom_parameters[key]
			else:
				return None

	def __setitem__(self, key, value):

		if key in self._builtins:
			raise NotModifiableError(key)
		else:
			self._custom_parameters[key] = value

class SingleRunData:
	'''A container to hold and access data from a single model run.

	Automatically processed and placed into a ModelRunSeries object,
	this allows indexing and access of the data associated with a single model.
    '''

	def __init__(self, model, product, init_time, valid_time, forecast_hour, file_path, **kwargs):
		self.model = model
		self.product = product
		self.init_time = init_time
		self.valid_time = valid_time
		self.forecast_hour = forecast_hour
		self.file_path = file_path

		self._builtins = ['model', 
						  'product', 
						  'init_time',
						  'valid_time', 
						  'forecast_hour', 
						  'file_path',
						  ]

		self._custom_parameters = kwargs

	def __getitem__(self, key):

		if key in self._builtins:
			return getattr(self, key, None)
		else:
			if key in self._custom_parameters:
				return self._custom_parameters[key]
			else:
				return None

	def __setitem__(self, key, value):

		if key in self._builtins:
			raise NotModifiableError(key)
		else:
			self._custom_parameters[key] = value
