import json
import numpy as np
import torch
import triton_python_backend_utils as pb_utils

class TritonPythonModel: 
    
    def initialize(self, args):
        
        self.model_config = model_config = json.loads(args['model_config'])
        
    def execute(self, requests):
        responses = [] 
        for request in requests: 
            inp = pb_utils.get_input_tensor_by_name(request, "prompt")
            