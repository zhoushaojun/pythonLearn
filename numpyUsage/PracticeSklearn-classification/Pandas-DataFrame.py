import pandas as pd
import numpy as np
aa={'one':[1,2,3],'two':[2,3,4],'three':[3,4,5]}
bb=pd.DataFrame(aa,index=[22,33,44],columns=['one','two','three'])

print(bb['one'])
print(bb[['one','two']])

print(np.arange(2))
print(bb[[1,2]])
print(bb[[1]])
print(bb[:1])
print("--")
print(bb[1:3])
print(bb[:1])

print(bb[1:2][['one','two']])

print(bb.iloc[1:2,1:2])