Recreate the program using below steps:

FOLDER STRUCTURE:
.
├── CSCI544_HW4_NN.py
├── data
│   ├── dev
│   ├── test
│   └── train
├── eval.py
└── output
    ├── bilstm-1.pt
    ├── bilstm-2.pt
    ├── bilstm-3.pt
    ├── dev-bonus.out
    ├── dev1.out
    ├── dev2.out
    ├── test-pred-bonus.out
    ├── test1.out
    └── test2.out
    

STEPS: 

- Make sure you have the necessary packages installed. You can install them using the following command:
	pip install torch tqdm datasets numpy pandas

- Download the CSCI544_HW4_NN.py file into current working directory

- keep the data folder in the same directory

- keep the eval.py file in the same directory

- run the CSCI544_HW3_NN.py file

- An 'output' directory will be automatically created which will contain following files:
    - bilstm-1.pt
    - bilstm-2.pt
    - bilstm-3.pt
    - dev1.out
    - test1.out
    - dev2.out
    - test2.out
    - dev-bonus.out
    - test-pred-bonus.out

- For Bi-LSTM testing of Dev & Test Data, run the eval.py file command : 
	python3 eval.py -p output/dev1.out -g data/dev
	python3 eval.py -p output/test1.out -g data/test

- For Glove Embeddings Task, run the eval.py file command : 
	python3 eval.py -p output/dev2.out -g data/dev
	python3 eval.py -p output/test2.out -g data/test

- For Bonus Task, LSTM-CNN Model, run the eval.py file command :
	python3 eval.py -p output/dev-bonus.out -g data/dev		
	python3 eval.py -p output/test-pred-bonus.out -g data/test
	


ADDITIONAL COMMENTS:
- Currently the test data file has no NER column, so before running eval.py script, 
    ensure to change test data file, such that it has POS column.
- The eval.py script will give the accuracy of the model.