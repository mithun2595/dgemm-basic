import matplotlib
import matplotlib.pylab as plt
import collections
import sys

matplotlib.rcParams.update({'font.size': 5})
for filename in sys.argv[1:]:
	with open(filename) as f:
	    content = f.readlines()
	content = content[109:160]
	content = [line.strip() for line in content]
	readings = []
	for line in content:
	    line = line.replace("Size: ","")
	    line = line.replace("\tGflop/s: ",",")
	    readings = readings + [line];
	matrix_size, gflops = zip(*(s.split(",") for s in readings))
	matrix_size = list(matrix_size)
	gflops = list(gflops)
	gflops = list(map(float,gflops))
	matrix_size = list(map(int,matrix_size))
	gflop_dict = dict(zip(matrix_size, gflops))
	ordered_gflop = collections.OrderedDict(sorted(gflop_dict.items()))
	plt.xticks(range(len(ordered_gflop)), ordered_gflop.keys())
	plt.plot(ordered_gflop.keys(), ordered_gflop.values(), label=str(sum(gflops)/len(gflops)))
plt.legend(loc='upper left')
plt.show()

