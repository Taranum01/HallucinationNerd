import random

input_file = "Diff2Mean.vals"

def shuffle(grps):
    num_grps = len(grps)
    pool = []
    for i in range(num_grps):
        pool.extend(grps[i])
    random.shuffle(pool)
    new_grps = []
    start_index = 0
    for i in range(num_grps):
        end_index = start_index + len(grps[i])
        new_grps.append(pool[start_index:end_index])
        start_index = end_index
    return new_grps

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

count = 0
num_shuffles = 10000

for i in range(num_shuffles):
    new_samples = shuffle(samples)
    mean_diff = meandiff(new_samples[a], new_samples[b])
    if observed_mean_diff < 0 and mean_diff <= observed_mean_diff:
        count = count + 1
    elif observed_mean_diff >= 0 and mean_diff >= observed_mean_diff:
        count = count + 1

print("Observed difference of two means: %.4f" % observed_mean_diff)
print("%d out of %d experiments had a difference of two means" % (count, num_shuffles), end=" ")
if observed_mean_diff < 0:
    print("less than or equal to", end=" ")
else:
    print("greater than or equal to", end=" ")
print("%.4f." % observed_mean_diff)
print("The p-value is: %.6f" % (count / float(num_shuffles)))
