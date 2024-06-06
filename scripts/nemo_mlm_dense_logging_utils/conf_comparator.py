import torch
import sys, json


def dfs(a, b, diffs):
   if a is None or b is None:
      diffs.append(('', a, b))
      return
   if type(a) is dict:
      for key, val in a.items():
         if type(val) is dict:
            if not key in b:
               diffs.append((key, val, a.get(key, None)))
            else:
               dfs(val, b.get(key, None), diffs)
         else:
            if val != b.get(key, None):
               diffs.append((key, val, b.get(key, None)))
   if type(b) is dict:
      for key, val in b.items():
         if key in a: continue
         if type(val) is dict:
            if not key in a:
               diffs.append((key, val, a.get(key, None)))
            else:
               dfs(val, a.get(key, None), diffs)
         else:
            if val != a.get(key, None):
               diffs.append((key, val, a.get(key, None)))

   if type(a) is not dict:
      if a != b:
         diffs.append(('', a,b))

a = json.load(open(sys.argv[1]))
b = json.load(open(sys.argv[2]))

diffs = []
dfs(a, b, diffs)
for diff in diffs:
   print(diff)
exit(len(diffs) > 0)
