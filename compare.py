import csv

file1 = open('./submit_convert_1.csv')
csvreader1 = csv.reader(file1)
header = next(csvreader1)
rows1 = []
for row in csvreader1:
    rows1.append(row)

#print(len(rows1))

file2 = open('./submit_meanEnsemble_convert.csv')
csvreader2 = csv.reader(file2)
header = next(csvreader2)
rows2 = []
for row in csvreader2:
    rows2.append(row)

#print(len(rows2))


file3 = open('./submit_convert_2.csv')
csvreader3 = csv.reader(file3)
header = next(csvreader3)
rows3 = []
for row in csvreader3:
    rows3.append(row)

#print(len(rows3))

ct = 0
for i in range(len(rows1)):
    if rows1[i] == rows3[i] and rows1[i] == rows2[i]:
        ct+=1
        #print(rows1[i])
print(ct)
