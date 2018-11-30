import GOpenMax
import TrainCloseSetClassifier
import TrainBMVC_GAN
import GenerateBMVC_fakes
import TrainClassifier_with_fakes
import csv


full_run = True


def save_results(results):
    f = open("results_GOpenmax.csv", 'wt')
    writer = csv.writer(f)
    writer.writerow(('F1',))
    writer.writerow(('Opennessid 0', 'Opennessid 1', 'Opennessid 2', 'Opennessid 3', 'Opennessid 4'))
    maxlength = 0
    for openessid in range(5):
        list = results[openessid]
    maxlength = max(maxlength, len(list))

    for r in range(maxlength):
        row = []
        for openessid in range(5):
            if r < len(results[openessid]):
                f1, th = results[openessid][r]
                row.append(f1)
        writer.writerow(tuple(row))

    writer.writerow(('Threshold',))
    writer.writerow(('Opennessid 0', 'Opennessid 1', 'Opennessid 2', 'Opennessid 3', 'Opennessid 4'))

    for r in range(maxlength):
        row = []
        for openessid in range(5):
            if r < len(results[openessid]):
                f1, th = results[openessid][r]
                row.append(th)
        writer.writerow(tuple(row))

    f.close()


results = {}

for openessid in range(5):
    results[openessid] = []

for fold in range(5 if full_run else 1):
    for class_fold in range(5):

        # Train Closed set clasifier
        TrainCloseSetClassifier.main(fold, class_fold)

        # Train BMVC GAN
        TrainBMVC_GAN.main(fold, class_fold)

        # Generate fakes
        GenerateBMVC_fakes.main(fold)

        TrainClassifier_with_fakes.main(fold, class_fold)

        for openessid in range(0,5):
            res = GOpenMax.main(fold, openessid, class_fold, 500)
            results[openessid] += [res]
            save_results(results)
