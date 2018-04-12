class PredictionTree():
    Item = None
    Parent = None
    Children = None
    
    def __init__(self,itemValue=None):
        self.Item = itemValue
        self.Children = []
        self.Parent = None
        
    def addChild(self, child):
        newchild = PredictionTree(child)
        newchild.Parent = self
        self.Children.append(newchild)
        
    def getChild(self,target):
        for chld in self.Children:
            if chld.Item == target:
                return chld
        return None
    
    def getChildren(self):
        return self.Children
        
    def hasChild(self,target):
        found = self.getChild(target)
        if found is not None:
            return True
        else:
            return False
        
    def removeChild(self,child):
        for chld in self.Children:
            if chld.Item==child:
                self.Children.remove(chld)