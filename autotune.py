import os
import subprocess
import re
import time

def update(b1, b2):
    lines = []
    with open('dgemm-blocked.c') as infile:
        for line in infile:
            if "#define BLOCK_2_SIZE" in line:
                line = "#define BLOCK_2_SIZE " + str(b2) + "\n"
            if "#define BLOCK_1_SIZE" in line:
                line = "#define BLOCK_1_SIZE " + str(b1) + "\n"

            lines.append(line)


    with open('dgemm-blocked.c', 'w') as outfile:
        for line in lines:
            outfile.write(line)

def make():
    os.system("make")

def run():
    proc = subprocess.Popen("qsub batch-job.sh", shell=True, stdout=subprocess.PIPE).stdout.read()
    procnum = re.findall(r"\d{5}", proc)    
    pid = procnum[0]
    print(pid)
    return pid

def extract(pid, param):
    fname = "DGEMM.o"+str(pid) 
    print(fname)
    while not os.path.isfile(fname): 
        print("waiting...")
        time.sleep(5) 

    out = subprocess.Popen("qstat", shell=True, stdout=subprocess.PIPE).stdout.read()
    while out not in ["\n", '', ' ']:
        print("still waiting...")
        time.sleep(5)
        out = subprocess.Popen("qstat", shell=True, stdout=subprocess.PIPE).stdout.read()

    with open(fname) as f:
        content = f.readlines()

    content = [line.strip() for line in content]
    regex_pattern = "(Size: )([0-9]*)(\t)(Gflop\/s: )([+-]?([0-9]*[.])?[0-9]+)"
    compiled_pattern = re.compile(regex_pattern)
    readings = []
    for line in content:
        if(compiled_pattern.match(line)):
            line = line.replace("Size: ","")
            line = line.replace("\tGflop/s: ",",")
            readings = readings + [line]
    # print(readings)
    matrix_size, gflops = zip(*(s.split(",") for s in readings))

    matrix_size = list(matrix_size)
    gflops = list(gflops)
    gflops = list(map(float,gflops))
    matrix_size = list(map(int,matrix_size))
    gflop_dict = dict(zip(matrix_size, gflops))
    avg = 0
    count = 1
    for key, val in gflop_dict.items():
       count = count+1
       avg += val
    speed1024 = gflop_dict[1024]
    avg = avg/count
    return (param, avg, speed1024)

b1 = 36
speed = []
for b2 in range(300, 610, 50):
    update(b1, b2)
    make()
    name = run()
    tup = extract(name, b2)
    print(tup)
    speed.append(tup)
#    break

print("**********************  FINAL RESULTS : ")
print(speed)

'''
speed.append(extract('35546', 100))
print (speed)
'''


