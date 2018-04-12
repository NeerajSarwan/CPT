
# coding: utf-8

# In[1]:


from PredictionTree import *
import pandas as pd
import operator
from tqdm import tqdm


# In[2]:


def load_files(train_file,test_file):
    data = []
    target = []
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)
    
    for index, row in train.iterrows():
        data.append(row.values)
        
    for index, row in test.iterrows():
        data.append(row.values)
        target.append(list(row.values))
        
    return data, target
    


# In[3]:


def train(data):
    
    alphabet = set()
    
    root = PredictionTree()
    
    cursornode = root
    
    II = {}
    
    LT = {}

    for seqid,row in enumerate(data):
        for element in row:

            # adding to the Prediction Tree

            if cursornode.hasChild(element)== False:
                cursornode.addChild(element)
                cursornode = cursornode.getChild(element)

            else:
                cursornode = cursornode.getChild(element)

            # Adding to the Inverted Index

            if II.get(element) is None:
                II[element] = set()

            II[element].add(seqid)
            
            alphabet.add(element)

        LT[seqid] = cursornode

        cursornode = root
        
    return root, II, LT, alphabet


# In[4]:


#data = load_files("train_wide.csv","test_wide.csv")


# In[5]:


#root, II, LT, alphabet = train(data)


# In[6]:


def score(counttable,key, length, target_size, number_of_similar_sequences, number_items_counttable):
    weight_level = 1/number_of_similar_sequences
    weight_distance = 1/number_items_counttable
    score = 1 + weight_level + weight_distance* 0.001
    
    if counttable.get(key) is None:
        counttable[key] = score
    else:
        counttable[key] = score * counttable.get(key)
        
    return counttable


# In[7]:


def predict(target,data, II, LT, alphabet, k, n): 
    """
    Here target is the test dataset in the form of list of list,
    k is the number of last elements that will be used to find similar sequences and,
    n is the number of predictions required.
    """
    
    predictions = []
    
    for each_target in tqdm(target):
        each_target = each_target[-k:]
        
        intersection = set(range(0,len(data)))
        
        for element in each_target:
            if II.get(element) is None:
                continue
            intersection = intersection & II.get(element)
        
        similar_sequences = []
        
        for element in intersection:
            currentnode = LT.get(element)
            tmp = []
            while currentnode.Item is not None:
                tmp.append(currentnode.Item)
                currentnode = currentnode.Parent
            similar_sequences.append(tmp)
            
        for sequence in similar_sequences:
            sequence.reverse()
            
        counttable = {}

        for  sequence in similar_sequences:
            try:
                index = next(i for i,v in zip(range(len(sequence)-1, 0, -1), reversed(sequence)) if v == each_target[-1])
            except:
                index = None
            if index is not None:
                count = 1
                for element in sequence[index+1:]:
                    if element in each_target:
                        continue
                        
                    counttable = score(counttable,element,len(each_target),len(each_target),len(similar_sequences),count)
                    count+=1
                    
#                     if counttable.get(element) is None:
#                         support = len(II.get(element))/len(LT)
#                     else:
#                         support = counttable.get(element) + len(II.get(element))/len(LT)

#                     counttable[element] = support

        pred = get_n_largest(counttable,n)
        predictions.append(pred)

    return predictions


# In[8]:


def get_n_largest(dictionary,n):
    largest = sorted(dictionary.items(), key = lambda t: t[1], reverse=True)[:n]
    return [key for key,_ in largest]


# In[9]:


# test = pd.read_csv("test_wide.csv")


# # In[10]:


# target = []
# for index, row in test.iterrows():
#     target.append(list(row.values))


# # In[11]:


# preds = predict(target,5,3)


# # In[ ]:


# # n = 5

# # test = pd.read_csv("test_wide.csv")



# # target = list(test.loc[0].values)

# # target = target[-n:]

# # target

# # intersection = set(range(0,len(data)))

# # for element in target:
# #     if II.get(element) is None:
# #         continue
# #     intersection = intersection & II.get(element)

# # intersection

# # similar_sequences = []

# # for element in intersection:
# #     currentnode = LT.get(element)
# #     tmp = []
# #     while currentnode.Item is not None:
# #         tmp.append(currentnode.Item)
# #         currentnode = currentnode.Parent
# #     similar_sequences.append(tmp)
        

# # for sequence in similar_sequences:
# #     sequence.reverse()

# # counttable = {}

# # for  sequence in similar_sequences:
# #     index = next(i for i,v in zip(range(len(sequence)-1, 0, -1), reversed(sequence)) if v == target[-1])
# #     for element in sequence[index+1:]:
# #         if counttable.get(element) is None:
# #             support = len(II.get(element))/len(LT)
# #         else:
# #             support = counttable.get(element) + len(II.get(element))/len(LT)
            
# #         counttable[element] = support

# # predicted = max(counttable.items(),key =operator.itemgetter(1))[0]

# # predicted

# # counttable


# # In[12]:


# final = []

# for pred in preds:
#     if len(pred)==0:
#         pred = [24530]*3
#     if len(pred)==1:
#         pred.append(pred[0]+1)
#         pred.append(pred[0]+2)
#     if len(pred)==2:
#         pred.append(pred[1]+1)
    
#     final.append(pred)
    


# # In[14]:


# ext = []
# for l in final:
#     ext.extend(l)
    


# # In[16]:


# sample_sub = pd.read_csv("sample_submission.csv")


# # In[19]:


# sample_sub.head()


# # In[21]:


# submission = pd.DataFrame({"user_sequence":sample_sub['user_sequence'],"challenge":ext})


# # In[23]:


# def replace(x):
#     return "CI"+str(x)


# # In[24]:


# submission['challenge'] = submission['challenge'].apply(replace)


# # In[25]:


# submission.head()


# # In[26]:


# submission.to_csv("coded.csv", index = False)


# # In[ ]:


# thefile = open("predictions.txt","w")


# # In[ ]:


# for line in final:
#     thefile.write(",".join(map(str,line)))
#     thefile.write("\n")


# # In[ ]:


# thefile.close()


# # In[ ]:


# get_ipython().system('pwd')

