import os

i = 1

while i <= 10000:
    k = 1
    while k < 100:
        os.system("python decode-better.py -s " +  str(i) + " -k " + str(k) + " | python compute-model-score > decodeBetterNoDist" + str(k) + "_" + str(i))
        k *= 10
    i *= 10
