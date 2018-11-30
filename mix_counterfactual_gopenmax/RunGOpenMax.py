import TrainCloseSetClassifier
import GOpenMax
import GenerateBMVC_fakes
import TrainClassifier_with_fakes
import TrainBMVC_GAN
import csv

full_run = True

fold = 0

tail_size = 500

def SaveResults(results):
    f = open("GOpenMaxResults.csv", 'wt')
    writer = csv.writer(f)
    writer.writerow(('Openness 0', 'Openness 1', 'Openness 2', 'Openness 3', 'Openness 4'))
    maxlength = 0
    for openness in range(5):
        list = results[openness]
        maxlength = max(maxlength, len(list))

    for r in range(maxlength):
        row = []
        for openness in range(5):
            list = results[openness]
            row.append(list[r] if len(list) > r else '')
        writer.writerow(tuple(row))
    f.close()

results = {}

for openness in range(5):
    results[openness] = []

for class_fold in range(5):
    for i in range(5 if full_run else 1):
        TrainCloseSetClassifier.main(i, class_fold)

    for i in range(5 if full_run else 1):
        TrainBMVC_GAN.main(i, class_fold)

        GenerateBMVC_fakes.main(i)

        TrainClassifier_with_fakes.main(i, class_fold)

        for openness in range(1, 5):
            F1, best_th = GOpenMax.main(i, openness, tail_size, class_fold)
            results[openness].append(F1)
            SaveResults(results)
