import sys 
paths = ['.', './0-Data']
for path in paths:
    if path not in sys.path:
        sys.path.append(path)

import Data_Generator as data
from Comparison import parameters

data.run(parameters)
# data.save_job_ids_params(parameters)