import random
import math

input_file = "Diff2Mean.vals"
conf_interval = 0.90

def bootstrap(x):
    samp_x = []
    for i in range(len(x)):
        samp_x.append(random.choice(x))
    return samp_x

def meandiff(grpA, grpB):
    return sum(grpB) / float(len(grpB)) - sum(grpA) / float(len(grpA))

samples = []
a = 0
b = 1

infile = open(input_file)
for line in infile:
    if line.startswith('>'):
        samples.append([])
    elif not line.isspace():
        samples[len(samples) - 1] += list(map(float, line.split()))
infile.close()

observed_mean_diff = meandiff(samples[a], samples[b])

num_resamples = 10000
out = []

for i in range(num_resamples):
    bootstrap_samples = []
    for sample in samples:
        bootstrap_samples.append(bootstrap(sample))
    out.append(meandiff(bootstrap_samples[a], bootstrap_samples[b]))

out.sort()

tails = (1 - conf_interval) / 2
lower_bound = int(math.ceil(num_resamples * tails))
upper_bound = int(math.floor(num_resamples * (1 - tails)))

print("Observed difference between the means: %.4f" % observed_mean_diff)
print("We have %.0f%% confidence that the true difference between the means" % (conf_interval * 100), end=" ")
print("is between: %.4f and %.4f" % (out[lower_bound], out[upper_bound]))
