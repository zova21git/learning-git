<<<<<<< HEAD

import glob
import pandas as pd
import numpy as np

files = glob.glob("file path")
files.head(10)
files.head(1)
print(files)

s = 0

for fn in files:
    pf = pd.read_csv(fn)
    s += sum(pf['Chicago'].to_list())

print(s)

=======
import glob
import pandas as pd
import numpy as np

files = glob.glob("file path")

s = 0

for fn in files:
    pf = pd.read_csv(fn)
    s += sum(pf['Chicago'].to_list())

print(s)
>>>>>>> 88efeba (added a code to first.py)
