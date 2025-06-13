import sys 
paths = ['.', './2-Graphing']
for path in paths:
    if path not in sys.path:
        sys.path.append(path)

import Graph_Generator as graph_gen
from Comparison import parameters

graph_gen.run(parameters)