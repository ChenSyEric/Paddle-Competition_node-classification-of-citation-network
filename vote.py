import csv
from collections import Counter

def vote_merge(filelst):
    result = {}
    fw = open('merge.csv', encoding='utf-8', mode='w', newline='')
    csv_writer = csv.writer(fw)
    csv_writer.writerow(['nid', 'label'])
    for filepath in filelst:
        cr = open(filepath, encoding='utf-8', mode='r')
        csv_reader = csv.reader(cr)
        for i, row in enumerate(csv_reader):
            if i == 0:
                continue
            idx, cls = row
            if idx not in result:
                result[idx] = []
            result[idx].append(cls)

    for nid, clss in result.items():
        counter = Counter(clss)
        true_cls = counter.most_common(1)
        csv_writer.writerow([nid, true_cls[0][0]])

if __name__ == '__main__':
    vote_merge([
        ".\submission_lib\submission_0.71873.csv",
        ".\submission_lib\submission_0.6947391.csv",
        ".\submission_lib\submission_0.7052751.csv",
        ".\submission_lib\submission_0.7170214.csv",
        ".\submission_lib\submission_0.7211504.csv",
        ".\submission_lib\submission_0.71509933.csv",
        ".\submission_lib\submission_4.csv",
        ".\submission_lib\submission_dpgcn5.csv",
        ".\submission_lib\submission_dpgcn10.csv",
        ".\submission_lib\submission_GCN.csv",
        ".\submission_lib\submission_GCN1300.csv",
        ".\submission_lib\submission_resgat.csv",
        ".\submission_lib\submission_resgat2.csv",
        ".\submission_lib\submission_resgcn.csv",
        ".\submission_lib\submission_resgcn2.csv",
        ".\submission_lib\submission_0.73228.csv.csv"
                ])


