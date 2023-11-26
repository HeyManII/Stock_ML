import csv
def generateCsv(filePath, data):
    with open(filePath, 'w', newline='') as file:
        writer = csv.writer(file)
        # Write each row of data to the csv file
        for row in data:
            writer.writerow(row)