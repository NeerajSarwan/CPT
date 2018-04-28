from CPT import *
model = CPT()
train,test = model.load_files("./data/train.csv","./data/test.csv",merge = True)
model.train(train)
predictions = model.predict(train,test,5,3)