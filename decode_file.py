import ast
import sys

file_name = sys.argv[1]
out = []
for line in open(file_name):
  line = line.strip('\n')
  line = ast.literal_eval(line).decode('utf-8')
  out.append(line)
open(file_name,'w').write('\n'.join(out))