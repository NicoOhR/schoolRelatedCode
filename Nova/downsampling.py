
import csv

#main:
#  iterate over csv:
#    * for every point, calculate the center coordinate of the voxel it would be in
#    * append the center coordinate of the voxel to output.csv
#  iterate over output.csv:
#    * remove duplicate entries

def removeDuplicate(fileName):
    input = open(fileName)

    inputSet = set(input)

    cleanOutput = open("output.csv",'w')

    for line in inputSet:
        cleanOutput.write(line)


def main():

    stepSize = 0.05
    
    with open("output.csv", 'w') as outputFile:

        outputFile.write('x,y,z')
        
        with open("input.csv") as inputFile:
            csvInputReader = csv.reader(inputFile)

            header = next(inputFile)

            for row in csvInputReader:
                x = float(row[0])
                y = float(row[1])
                z = float(row[2])

                #finding the nearest center coordinante of the voxal
                xCenter = ((x - (x % stepSize)) + stepSize)/2
                yCenter = ((y - (y % stepSize)) + stepSize)/2
                zCenter = ((z - (z % stepSize)) + stepSize)/2

                outputFile.write(f'{xCenter},{yCenter},{zCenter} \n')
            
            inputFile.close()
    
    outputFile.close()
    
    removeDuplicate("output.csv")
    
main()
    
            



