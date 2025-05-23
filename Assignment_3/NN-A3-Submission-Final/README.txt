Recreate the progam using below steps:

FOLDER STRUCTURE:
- CSCI544_HW3_NN.py
- eval.py
- data
    - train
    - dev
    - test 
- output
    - hmm.json
    - vocab.txt
    - greedy.out
    - viterbi.out
    

STEPS:
- Download the CSCI544_HW3_NN.py file into current working directory
- keep the data folder in the same directory
- keep the eval.py file in the same directory
- run the CSCI544_HW3_NN.py file
- An 'output' directory will be automatically created which will contain following files:
    - hmm.json
    - vocab.txt
    - greedy.out
    - viterbi.out
- run the eval.py file command : python3 eval.py -p output/greedy.out -g data/test
- run the eval.py file command : python3 eval.py -p output/viterbi.out -g data/test


ADDITIONAL COMMENTS:
- Currently the test data file has no POS column, so before running eval.py script, 
    ensure to change test data file, such that it has POS column.
- The eval.py script will give the accuracy of the model.