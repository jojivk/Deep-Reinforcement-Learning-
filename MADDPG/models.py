import os
import re
import glob

#files = "./models/checkpoint_*.pth*"
files = "./models/checkpoint_actor_local.pth_0_1.15*"

file = glob.glob(files)[0]
print(file)
#for filename in glob.glob(files):
#    print(filename)
