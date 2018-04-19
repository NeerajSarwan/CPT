from CPT import *
model = CPT()
data,target = model.load_files("./data/train.csv","./data/test.csv")
model.train(data)
predictions = model.predict(data,target,5,3)