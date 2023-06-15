import click
import csv
import json
from  project_pat import prediction, depletion, benchmark

@click.command()
@click.option('--output', default='output', prompt='Name of output JSON file (without the ".json")?', help='name of output (default: output')
@click.option('--testpredictionmodel', default='', help='path to csv')
@click.option('--testdepletionchances', default='', help='path to csv')
@click.option('--test1samplebenchmark', default='', help='path to csv')
@click.option('--test2samplebenchmark', default='', help='path to csv')
def main(output, testpredictionmodel, testdepletionchances, test1samplebenchmark, test2samplebenchmark):
    if testpredictionmodel:
        with open(testpredictionmodel, newline='') as csvfile:
            reader = csv.reader(csvfile)
            observed = []
            fitted = []
            stimuli = []
            next(reader)
            label = ''
            for row in reader:
                if row[4]:
                    label = row[4]
                observed.append(float(row[1]))
                fitted.append(float(row[2]))
                stimuli.append(float(row[0]))
            result = prediction.testmodel(observed, fitted, stimuli, label)
            json_formatted_str = json.dumps(result, indent=2)
            print(json_formatted_str)
        with open(output + ".json", "w") as outfile:
            outfile.write(json_formatted_str)
        return
    if testdepletionchances:
        with open(testdepletionchances, newline='') as csvfile:
            reader = csv.reader(csvfile)
            observed = []
            timestamps = []
            next(reader)
            hi_pass_filter = 0.0
            label = ''
            for row in reader:
                if row[2]:
                    hi_pass_filter = float(row[2])
                if row[3]:
                    label = row[3]
                observed.append(float(row[1]))
                timestamps.append(float(row[0]))
            result = depletion.testchances(observed, timestamps, hi_pass_filter, label)
            json_formatted_str = json.dumps(result, indent=2)
            print(json_formatted_str)
        with open(output + ".json", "w") as outfile:
            outfile.write(json_formatted_str)
        return
    if test1samplebenchmark:
        with open(test1samplebenchmark, newline='') as csvfile:
            reader = csv.reader(csvfile)
            a = []
            next(reader)
            target = 0.0
            aname = ''
            for row in reader:
                if row[1]:
                    target = float(row[1])
                if row[2]:
                    aname = row[2]
                a.append(float(row[0]))
            result = benchmark.test1sample(a, target, aname)
            json_formatted_str = json.dumps(result, indent=2)
            print(json_formatted_str)
        with open(output + ".json", "w") as outfile:
            outfile.write(json_formatted_str)
        return
    if test2samplebenchmark:
        with open(test2samplebenchmark, newline='') as csvfile:
            reader = csv.reader(csvfile)
            a = []
            b = []
            next(reader)
            aname = ''
            bname = ''
            for row in reader:
                if row[2]:
                    aname = row[2]
                if row[3]:
                    bname = row[3]
                a.append(float(row[0]))
                b.append(float(row[1]))
            result = benchmark.test2sample(a, b, aname, bname)
            json_formatted_str = json.dumps(result, indent=2)
            print(json_formatted_str)
        with open(output + ".json", "w") as outfile:
            outfile.write(json_formatted_str)
        return
    click.echo(f"Use one of the following flags:")
    click.echo(f"--testpredictionmodel")
    click.echo(f"--testdepletionchances")
    click.echo(f"--test1samplebenchmark")
    click.echo(f"--test2samplebenchmark")

main()
