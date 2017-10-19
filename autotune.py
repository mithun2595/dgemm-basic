import os
import subprocess
import re

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
    procnum = re.findall(r"\D(\d{5})\D", proc)    
    pid = procnum[0]
    return pid

def extract(pid, param):
    
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
            readings = readings + [line];
    matrix_size, gflops = zip(*(s.split(",") for s in readings))
    matrix_size = list(matrix_size)
    gflops = list(gflops)
    gflops = list(map(float,gflops))
    matrix_size = list(map(int,matrix_size))
    gflop_dict = dict(zip(matrix_size, gflops))
    avg = 0
    count = 0
    for key, val in gflop_dict.items():
       count = count+1
       avg += val
    speed1024 = gflop_dict['1024']
    avg = avg/count
    return (param, avg, speed1024)

b1 = 36
speed = []

for b2 in range(100, 410, 10):
    update(b1, b2)
    make()
    name = run()
    speed.append(extract(name))
    break

print(speed)
