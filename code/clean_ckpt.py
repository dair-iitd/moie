import sys
import os
import regex as re

directory = sys.argv[1]

for file in os.listdir(directory):
    if 'ckpt-' in file:
        result = re.search('ckpt-(.*)\.', file)
        ckpt_no = result.group(1)
        if int(ckpt_no) % 2000 != 0:
            os.remove(directory+'/'+file)

