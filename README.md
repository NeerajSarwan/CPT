# CPT (Compact Prediction Tree)

This is the Python Implementation of CPT algorithm for Sequence Prediction. The library has been written from scratch in Python and as far as I believe is the first Python implementation of the algorithm.

The repository is also an exercise on my part to code a research paper. The library is not perfect. I have intentionally left out some optimisations such as CFS(compression of frequenct sequences) etc. These features will be later added to the library as an ongoing effort.

The library is created using the below two research papers.

1. [Compact Prediction Tree: A Losless Model for Accurate Sequence Prediction](http://www.philippe-fournier-viger.com/spmf/ADMA2013_Compact_Prediction_tree) 

2. [CPT+: Decreasing the time/space complexity of the Compact Prediction Tree](https://pdfs.semanticscholar.org/bd00/0fe7e222b8095c6591291cd7bef18f970ab7.pdf)


- How to use the library?

There is no requirement of compiling anything but make sure you have Pandas and tqdm installed in your environment specific versions of which are mentioned in the file requirements.txt.

- Sample code for training and getting predictions.

~~~
# When inside the CPT folder

from CPT import CPT

model = CPT()

train, test = model.load_files("./data/train.csv","./data/test.csv", merge = True)

model.train(data)

predictions = model.predict(train,test, k, n)

~~~



