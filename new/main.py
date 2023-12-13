import tomllib
from processor import ContrailProcessor

PATH_TO_CONFIG = "config/config.toml"

if __name__ == "__main__":
	__spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

	with open(PATH_TO_CONFIG, 'rb') as config_file:
		config_opts = tomllib.load(config_file)

	# print(config_opts)
	# print(type(config_opts))

	proc = ContrailProcessor(config_opts)

	proc.populate_keys()

	proc.aws_download_multithread()

	#print(proc.data_files)

	proc.plot_multiprocess()
