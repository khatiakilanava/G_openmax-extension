import GOpenMaxClustering
import GenerateBMVC_fakes
import csv


full_run = True

Opennessid = [1, 2, 3, 4, 5]

def save_results(results):
    f = open("results.csv", 'wt')
    writer = csv.writer(f)
    writer.writerow(('F1',))
    writer.writerow(('Folding_id = 0', 'Opennessid', 'Class_fold', 'Tail_size 500', 'Folds=5'))
    for i in range(len(results)):
        row = []
        row.append(results[i::2])
        writer.writerow(tuple(row))

    writer.writerow(('Threshold',))
    writer.writerow(('Folding_id = 0', 'Opennessid', 'Class_fold', 'Tail_size 500', 'Folds=5'))

    for i in range(len(results)):
        row = []
        row.append(results[1::2])
        writer.writerow(tuple(row))
    f.close()


results = ()
full_results = ()


for fold in range(5 if full_run else 1):

    for i in range(5):
        res = GOpenMaxClustering.main(0, i, i, 500)
        results = results + res
    full_results = full_results + results

save_results(full_results)