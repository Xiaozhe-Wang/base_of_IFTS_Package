import os
import torch
import yaml
import logging
import numpy as np
from functools import partial
from typing import Any

class Base_Para:
    # Constants are defined here
    SPEED_OF_LIGHT = 299792458              # speed of light (m/s)   
    PLANKS_CONSTANT = 6.626068e-34          # Plank's Constant
    EPSILON = 1e-12                         # Small epsilon value for numerical stability
    PI = np.pi                              # Pi, the mathematical constant (π)
    E = np.e                                # Euler's number (e) is a mathematical constant approximately equal to 2.71828

    def __init__(self):
        """
        Initializes the Base_Para object with an empty print buffer.
        """
        self.print_buff = []
    
    
    @staticmethod
    def _is_yaml_file(filename: Any):
        """
        Checks if the given filename is a YAML file.

        Args:
            filename (Any): The name of the file to check.

        Returns:
            bool: True if the file is a YAML file, False otherwise.
        """
        return filename.endswith(('.yml', '.yaml'))

    @staticmethod
    def read_configs(fd: str, field: str = None):
        """
        Reads YAML configurations from a given file path or a path constructed from input parameters.

        Args:
            fd (str): The file path as a string, or a dictionary with 'config_path' key.
            field (str, optional): An additional field to append to the 'config_path' if 'fd' is a dictionary.

        Returns:
            dict: The contents of the YAML file as a dictionary.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            yaml.YAMLError: If there is an error in parsing the YAML file.
        """
        file_path = fd if isinstance(fd, str) else os.path.join(fd['config_path'], field)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Unable to find the specified configuration file: {file_path}")
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing the YAML file: {e}")

    @staticmethod
    def save_configs(path, configs):
        # Static method to save configurations to a YAML file
        with open(path, 'w') as f:
            yaml.dump(configs, f, sort_keys=False)

    @staticmethod
    def _unknown_module(module: Any = None):
        """
        Placeholder method for unknown modules. Logs an error if an unknown module is encountered.

        Args:
            configs (dict): Configurations for the module (not used in this method).
        """
        if module is None:
            raise NotImplementedError("Attempted to call an unknown module.")
        else:
            raise NotImplementedError(f"Attempted to call an unknown module: {module}")

    def check_module(self,  path: str, para_name: str, lib: Any, configs: dict = None):
        """
        Initializes modules found in a given directory to be True in the configuration.

        This method scans a specified directory for module files, reads their configurations,
        and updates the instance's state based on these configurations.

        Args:
            path (str): The path of the directory containing the modules.
            para_name (str): The parameter name associated with the modules.
            lib (Any): The library or module in which the found modules are to be initialized.
            configs (dict, optional): A dictionary of configurations.

        Notes:
            It prints a warning message if a module fails to open due to empty configurations.
        """
        for filename in os.listdir(path):
            if self._is_yaml_file(filename):
                module_name, _ = os.path.splitext(filename)
                config_path = os.path.join(path, filename)
                method_configs = self.read_configs(config_path)
                if method_configs:
                    # Attempt to get the attribute from lib, if it fails, call _unknown_module
                    try:
                        getattr(lib, f'{module_name}')
                    except AttributeError:
                        self._unknown_module(module_name)
                    # Check and update the configuration for the module
                    self._check(module_name, 1, configs=configs)
                else:
                    # Append warning to the print buffer if the module configs are empty
                    self.print_buff.append(f'Warning: {para_name} module {module_name} failed to open, due to empty configs')

    def set_module(self, path: str) -> None:
        """
        Sets modules based on the YAML files in the given directory.
        
        Args:
            path (str): The path where YAML files are located.
            configs (dict, optional): A dictionary of configurations.
        
        Raises:
            AttributeError: If the module name is not found in the class.
            OSError: If the module is not found in the library.
        """
        lib_name = 'IFTS_Package'  # Setting by library name
        lib_path = os.path.join(lib_name, 'library')
        # Check if path is a file
        if os.path.isfile(path):
            # If path is a specific file, directly process it
            filename = os.path.basename(path)
            if self._is_yaml_file(filename):  # Verify YAML file
                yml_name = os.path.splitext(filename)[0]
                full_yml_name = filename
                file_path = path  # Directly use the provided file path
                
                # Read and configure from YAML
                method_configs = self.read_configs(file_path)
                method_configs['yml_name'] = yml_name
                
                if method_configs:
                    module_name = method_configs.get('Module_Name')
                    py_name = module_name
                    full_py_name = f'{py_name}.py'
                    
                    # Attempt to get the method from the class
                    try:
                        method_to_call = getattr(self, f'__{module_name}__')
                    except AttributeError:
                        raise AttributeError(f"Module {module_name} is not in para.")
                    
                    # Determine the path to the library
                    path_parts = [item for item in os.path.dirname(path).split(os.sep) if item.strip()]
                    case = path_parts[-1]  # Extract the last path segment
                    count = 0
                    path_list = []  # List for storing valid paths
                    library_path = os.path.join(os.getcwd(), lib_path)
                    
                    # Walk through the library to find matching paths
                    for lib_dirpath, _, lib_filenames in os.walk(library_path):
                        if case in os.path.basename(lib_dirpath) and full_py_name in lib_filenames:
                            count += 1
                            path_list.append(os.path.join(lib_dirpath, case))
                            if method_configs['init_flag'] == '1':
                                method_to_call(method_configs)  # Call the method with configs
                            self.print_buff.append(f'{full_yml_name} init module {full_py_name} succeeded.')
                    
                    # Raise an error if no matching path is found
                    if count == 0:
                        raise OSError(f"{filename} is not in the library.")
        else:
            # Iterate through the directories and files
            for dirpath, _, filenames in os.walk(path):
                for filename in filenames:
                    if self._is_yaml_file(filename):  # Verify YAML file
                        yml_name = os.path.splitext(filename)[0]
                        full_yml_name = filename
                        file_path = os.path.join(dirpath, filename)
                        
                        # Read and configure from YAML
                        method_configs = self.read_configs(file_path)
                        method_configs['yml_name'] = yml_name
                        
                        if method_configs:
                            module_name = method_configs.get('Module_Name')
                            py_name = module_name
                            full_py_name = f'{py_name}.py'
                            
                            # Attempt to get the method from the class
                            try:
                                method_to_call = getattr(self, f'__{module_name}__')
                            except AttributeError:
                                raise AttributeError(f"Module {module_name} is not in para.")
                            
                            # Determine the path to the library
                            path_parts = [item for item in dirpath.split(os.sep) if item.strip()]
                            case = path_parts[-1]  # Extract the last path segment
                            count = 0
                            path_list = []  # List for storing valid paths
                            library_path = os.path.join(os.getcwd(), lib_path)
                            
                            # Walk through the library to find matching paths
                            for lib_dirpath, _, lib_filenames in os.walk(library_path):
                                if case in os.path.basename(lib_dirpath) and full_py_name in lib_filenames:
                                    count += 1
                                    path_list.append(os.path.join(lib_dirpath, case))
                                    if method_configs['init_flag']=='1':
                                        method_to_call(method_configs)  # Call the method with configs
                                    self.print_buff.append(f'{full_yml_name} init module {full_py_name} succeeded.')
                            
                            # Raise an error if no matching path is found
                            if count == 0:
                                raise OSError(f"{filename} is not in the library.")
    
    def update_module(self, para_name, configs_dict):
        """
        Update modules based on the given configurations.

        Args:
            para_name (str): Parameter name used for logging.
            configs_dict (dict): A dictionary where keys are module names and values are their configurations.
        """
        for module_name, method_configs in configs_dict.items():
            # Retrieve the corresponding method for the module and call it with its configurations
            method_to_call = getattr(self, f'__{module_name}__', self._unknown_module)
            method_to_call(method_configs)
            self.print_buff.append(f'{para_name} update module: {module_name}')    

    def _check_data_type(self, value, data_type, para_name='some parameter'):
        """
        Checks if the data type of a value matches the expected data type.

        Args:
            value: The value to check.
            data_type: The expected data type.
            para_name: The name of value to check.

        Returns:
            value: The value, cast to the expected data type if necessary.
        """
        if not isinstance(value, data_type):
            self.print_buff.append(f'Warning: Expected data type {data_type} for {para_name}, but received {type(value)}')
            value = data_type(value)
        return value
            
    def _check(self, param_name, default_value='empty_value', configs=None, notes=None, data_type=None):
        """
        Checks if a parameter exists in the object's dictionary. If not, it sets it to a default value or 
        the value from configs if provided.

        Args:
            param_name (str): The name of the parameter to check.
            default_value: The default value to assign if the parameter is missing.
            configs (dict, optional): A dictionary of configurations to check against.
            notes (str, optional): Additional notes or description for the parameter.
            data_type (type, optional): The expected data type of the parameter.

        Returns:
            The value set for the parameter.

        Raises:
            RuntimeError: If the final value of the parameter is 'empty_value', indicating a required value was not provided.
        """
        if not hasattr(self, param_name):
            if configs is not None:
                if param_name in configs:
                    self.__dict__[param_name] = configs[param_name]  #使用字典的方式对其中实例（self）属性的值进行修改
                else:
                    self.__dict__[param_name] = default_value
                    configs[param_name] = default_value
            else:
                self.__dict__[param_name] = default_value
        else:
            if configs is not None:
                configs[param_name] = self.__dict__[param_name]
        
        if isinstance(self.__dict__[param_name],str) and self.__dict__[param_name] == 'empty_value':
            raise RuntimeError(param_name + ' is an empty value')     
        elif data_type is not None:
            self.__dict__[param_name] = self._check_data_type(self.__dict__[param_name], data_type, para_name=param_name)

        if notes is not None:
            self.__dict__[param_name+'_help'] = param_name + ': '+notes
            
        return self.__dict__[param_name]

    def _config_check(self, param_name, default_value='empty_value', configs={}, notes=None, data_type=None):
        """
        Checks if a parameter is in the configs. If not, it adds it with a default value. 
        Notes that the added parameter would not be added in the class, which is the difference with _check.

        Args:
            param_name (str): The name of the parameter to check.
            default_value: The default value to assign if the parameter is missing.
            configs (dict, optional): A dictionary of configurations.
            notes (str, optional): Optional notes regarding the parameter.
            data_type (type, optional): The expected data type of the parameter.

        Returns:
            The value from the configs corresponding to param_name.

        Raises:
            RuntimeError: If default_value is 'empty_value' and param_name is not in configs.
        """
        if param_name not in configs:
            if isinstance(default_value,str) and default_value == 'empty_value':
                raise RuntimeError(param_name + ' is an empty value')
            configs[param_name] = default_value
        if data_type is not None:
            configs[param_name] = self._check_data_type(configs[param_name], data_type, para_name=param_name)    
        if notes is not None:
            self.__dict__[param_name+'_help'] = param_name + ': '+notes
        return configs[param_name]

    def _check_data_mode(self, config, data_mode='numpy'):
        """
        Checks and sets the data mode for the configuration.

        Args:
            config (dict): The configuration dictionary to update.
            data_mode (str): The default data mode to use if 'hybrid' is set.

        Raises:
            AttributeError: If the data mode is not supported.
        """
        self._check('data_mode')
        if self.data_mode in ['numpy', 'torch', 'hybrid']:
            config['data_mode'] = self.data_mode if self.data_mode != 'hybrid' else data_mode
        elif self.data_mode == 'config_assign':
            self._config_check('data_mode', configs=config)
        else:
            raise AttributeError('Unsupported data mode. Should be numpy, torch, hybrid, or config_assign')

    def get_logger(self, name, verbosity=3):
        """
        Creates and returns a logger with a specified verbosity level.

        Args:
            name (str): The name of the logger.
            verbosity (int): The verbosity level for the logger.

        Returns:
            logging.Logger: A logger configured with the specified verbosity level.

        Raises:
            AssertionError: If the verbosity level is not recognized.
        """
        log_levels = {
            0: logging.ERROR,
            1: logging.WARNING,
            2: logging.INFO,
            3: logging.DEBUG
        }
        assert verbosity in log_levels, f'verbosity option {verbosity} is invalid. Valid options are {log_levels.keys()}.'
        logger = logging.getLogger(name)
        logger.setLevel(log_levels[verbosity])
        return logger

    def print_to_log(self, logger, print_flag=True):
        """
        Prints buffered messages to the log.

        Args:
            logger (logging.Logger): The logger to which messages will be printed.
            print_flag (bool): If True, prints messages to the log.

        Raises:
            RuntimeError: If an unrecognized logger class is encountered.
        """
        if print_flag:
            for contents in self.print_buff:
                logger_class, print_info = contents.split('__')
                getattr(logger, logger_class, logger.error)(print_info)
        self.logger = logger
            
    def save_data_func_npol(self, x, name, **kwargs):
        """
        Saves data with two components (e.g., x and y) to a compressed file. Supports numpy arrays and torch tensors.

        Args:
            x (Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]): The data tuple to save.
            name (str): The filename to use for saving the data.

        Keyword Args:
            Additional keyword arguments (not used in this function).
        """
        data_x, data_y = (x[0].cpu().numpy(), x[1].cpu().numpy()) if isinstance(x[0], torch.Tensor) else (x[0], x[1])
        np.savez_compressed(os.path.join(self.path_dict['rx_save_data'], f'{name}.npz'), data_x=data_x, data_y=data_y)

    def get(self, param_name, default_value='empty_value'):
        return self.__dict__.get(param_name, default_value)

    def init_obj(self, config, module, *args, **kwargs):
        """
        Instantiates an object from a module based on configuration.

        Args:
            name (str): The name of the configuration parameter.
            module (ModuleType): The module where the class is located.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            object: An instance of the class specified in the configuration.

        Raises:
            AssertionError: If kwargs in configuration are overwritten.
        """
        # config = self._get_config(name)
        module_name = config['Module_Name']
        module_module = config['Module_Model']
        self._assert_no_overwrite_kwargs(config[module_module], kwargs)
        config[module_module].update(kwargs)
        return getattr(getattr(module,module_name), module_module)(*args, **config[module_module])

    def init_ftn(self, name, module, *args, **kwargs):
        """
        Initializes a function with preset arguments using functools.partial.

        Args:
            name (str): The name of the configuration parameter.
            module (ModuleType): The module where the function is located.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            function: A function with some arguments fixed.

        Raises:
            AssertionError: If kwargs in configuration are overwritten.
        """
        config = self._get_config(name)
        module_name = config['mode']
        module_args = config['args']
        self._assert_no_overwrite_kwargs(module_args, kwargs)
        module_args.update(kwargs)
        return partial(getattr(module, module_name), *args, **module_args)

    def _get_config(self, name):
        """
        Retrieves the configuration for a given name.

        Args:
            name (str): The name of the configuration parameter.

        Returns:
            dict: Configuration dictionary.
        """
        return dict(self.__dict__[name])

    def _assert_no_overwrite_kwargs(self, module_args, kwargs):
        """
        Asserts that kwargs do not overwrite existing module args.

        Args:
            module_args (dict): Existing module arguments.
            kwargs (dict): New keyword arguments to be added.

        Raises:
            AssertionError: If an attempt is made to overwrite existing kwargs.
        """
        if any(k in module_args for k in kwargs):
            raise AssertionError('Overwriting kwargs given in config file is not allowed')



class function_property:
    """
    A descriptor class to allow functions to be assigned to attributes.
    """
    def __init__(self, f):
        self.f = f

    def __get__(self, obj, owner):
        return self.f(obj) if obj else self