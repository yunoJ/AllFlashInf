#TODO: only one head is represented
import re
from collections import OrderedDict

depth = 96

def modify_tf_block():
  layerDict = OrderedDict()
  layerDict['gpt3'] = []

  result = []
  with open("./gpt3_block.m", 'r') as fi:
    for line in fi:
      # layerDict['gpt3'].append(line)
      result.append(line)

  with open("./gpt3.m", 'w') as fo:
    for line in result:
      fo.write(line)
        
  flag = True
  result = []

  with open("./gpt3_block.m", 'r') as fi:
    for line in fi:
      if "Network" in line:
        flag = False
        continue
      if (flag == True):
        continue
      else:
        result.append(line)
        
  with open("./gpt3.m", 'a') as fo:
    for i in range(depth):
      for line in result[:-2]:
        if (flag == True):
          if ("Constant" in line):
            flag = False;
            fo.write(line)
        fo.write(line)
    fo.write("}")

if __name__ == "__main__":
  modify_tf_block()
