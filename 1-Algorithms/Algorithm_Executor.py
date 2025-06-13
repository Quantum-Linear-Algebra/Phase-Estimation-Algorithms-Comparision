import sys 
paths = ['.', './0-Data', './1-Algorithms']
for path in paths:
    if path not in sys.path:
        sys.path.append(path)

import Algorithm_Manager as algo
from Comparison import parameters

algo.run(parameters)