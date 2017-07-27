import numpy as np
import matplotlib.pyplot as plt
mat = np.random.randint(1,10, size=(3,4))
print mat

print np.median(mat, axis=0)
print np.mean(mat, axis=0)
print np.average(mat, axis=0)
print np.var(mat, axis=0)
print np.std(mat, axis=0)
print '####################'
arr = np.empty(shape=(2,3))
print arr

print np.empty_like(arr)

print np.eye(3)
print np.eye(2,3,1)

print np.eye(3, k=1)
print '#########'
print np.identity(3)

print np.ones(shape=(2,2))

print np.full(shape=(2,3),fill_value=3)

arr1 = np.copy(np.full((2,2), 5))
print arr1

print np.arange(0,20,.5)
print np.linspace(0,20, dtype='int8')
print np.logspace(0,20, dtype='int8')
print '##################'
arr3 = np.random.rand(3,3)
print arr3
print np.diag(arr3,1)

print np.tri(3,3)
print np.tril(arr3)
print np.triu(arr3)

print '##################'

arr4 = np.ones(shape=(2,3,4))
print arr4

print np.reshape(arr4, newshape=(2,12))
print np.ravel(arr3) # flatten same

print arr3.flatten() # return copy
print np.transpose(arr3)

print np.concatenate((arr3,arr3), axis=1)
print np.hstack((arr3,arr3))