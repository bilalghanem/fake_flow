# psyLex: an open-source implementation of the Linguistic Inquiry Word Count
# Created by Sean C. Rife, Ph.D.
# srife1@murraystate.edu // seanrife.com // @seanrife
# Licensed under the MIT License


# Function to read in an LIWC-style dictionary
import collections, sys, re

def readDict(dictionaryPath):
    catList = collections.OrderedDict()
    catLocation = []
    wordList = {}
    finalDict = []

    # Check to make sure the dictionary is properly formatted
    with open(dictionaryPath, "r") as dictionaryFile:
        for idx, item in enumerate(dictionaryFile):
            if "%" in item:
                catLocation.append(idx)
        if len(catLocation) > 2:
            # There are apparently more than two category sections; throw error and die
            sys.exit("Invalid dictionary format. Check the number/locations of the category delimiters (%).")

    # Read dictionary as lines
    with open(dictionaryPath, "r") as dictionaryFile:
        lines = dictionaryFile.readlines()

    # Within the category section of the dictionary file, grab the numbers associated with each category
    for line in lines[catLocation[0] + 1:catLocation[1]]:
        try:
            catList[re.split(r'\s{20}', line)[0]] = [re.split(r'\s{20}', line.rstrip())[1]]
        except Exception as ex:
            print(ex)

    # Now move on to the words
    for idx, line in enumerate(lines[catLocation[1] + 1:]):
        # Get each line (row), and split it by tabs (\t)
        line = re.sub('(?<=[a-zA-Z*])\s+(?=\d{2})', ' ', line.rstrip())
        workingRow = re.split(' ', line)
        wordList[workingRow[0]] = list(workingRow[1:])

    # Merge the category list and the word list

    for key, values in wordList.items():
        for catnum in values:
            workingValue = catList[catnum][0]
            finalDict.append([key, workingValue])
    return finalDict
