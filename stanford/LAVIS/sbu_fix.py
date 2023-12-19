import json
import os

f = open(".cache/lavis/sbu_captions/sbu.json", "r")
s = f.readline()[2:-2].split("}, {")
o = []
e = os.listdir(".cache/lavis/sbu_captions/images/")

i = 0
for d in s:
    d = "{" + d + "}"
    n = json.loads(d)
    p = n["image"]
    if str(p) in e:
        o.append(n)
    if i % 1000 == 0:
        print(f"{i}/{len(s)}")
    i += 1

fo = open(".cache/lavis/sbu_captions/sbu_new.json", "w")
string = "["
for item in o:
    string += json.dumps(item) + ", "
string = string[:-2]
string += "]"
fo.write(string)
os.replace(".cache/lavis/sbu_captions/sbu_new.json", ".cache/lavis/sbu_captions/sbu.json")
print(len(o))