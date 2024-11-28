f = open("laba1.txt")

lines = []

lines = f.readlines()
lines2 = []

for l in lines:
    lines2.append(l.replace('\t', ','))

f2 = open("laba1.csv", "w+")

for l in lines2:
    f2.write(l)

f2.close()