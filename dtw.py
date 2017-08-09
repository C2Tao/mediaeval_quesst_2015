import theano.tensor as T
import theano
import numpy as np
import sys
A = np.array([[1,2],[1,1]],dtype='float32')
B = np.array([[0,1],[1,2],[1,1],[-1,-1]],dtype='float32')

X = T.fmatrix('X')
Y = T.fmatrix('Y')

#D = theano.shared()

euclidean_distances = T.sqrt((X ** 2).sum(1).reshape((X.shape[0], 1)) + (Y ** 2).sum(1).reshape((1, Y.shape[0])) - 2 * X.dot(Y.T))


#def accumulate(x,y):
    

#D = T.zeros_like(euclidean_distances,dtype = 'float32')



def inner_loop(distance,match,insertion,deletion):
    print 'inner',distance.ndim
    return distance + T.min(T.stack(insertion,deletion,match),axis=0)
    
def outer_loop(distance,past_value):
    print past_value.ndim
    print past_value.shape
    pads = theano.shared(9999.0)
    outputs, updates = theano.scan(fn = inner_loop,
                sequences = [distance,
                dict({inputs = past_value,taps = [-1,0]})],
                outputs_info = [pads])
    return outputs

def total_loop(distance,xlen):
    #pad = T.stack(T.zeros_like(distance[0][:]),T.zeros_like(distance[0][0])))
    #pad = theano.shared(np.zeros((xlen+1),dtype = 'float32'))
    pad = T.zeros_like(distance[0])
    outputs, updates = theano.scan(fn = outer_loop,
                sequences = [distance],
                outputs_info = [pad])
    return outputs

stuff = total_loop(euclidean_distances,4)
#for i in range(X.shape[0]):
#    for j in range(Y.shape[0]):
#        D[i][j] = euclidean_distances[i][j]

def accumulate3(distance):
    def csum(x,y):
        return x+y
    def rsum(X,Y):
        outputs, updates = theano.scan(fn = csum,
                    sequences = X,
                    outputs_info = T.zeros_like(distance[0][0]))
        return outputs + Y
    outputs, updates = theano.scan(fn = rsum,
                sequences = distance,
                outputs_info = T.zeros_like(distance[0]))
    return outputs
def accumulate2(distance):
    def csum(x,y):
        return x+y
    def rsum(X,Y):
        print 'ac2',X.ndim
        print 'ac2',Y.ndim
        outputs, updates = theano.scan(fn = csum,
                    sequences = X,
                    outputs_info = T.zeros_like(distance[0][0]))
        return outputs
    outputs, updates = theano.scan(fn = rsum,
                sequences = distance,
                outputs_info = T.zeros_like(distance[0]))
    return outputs

def accumulate(distance):
    def csum(x,y):
        return x+y
    outputs, updates = theano.scan(fn = csum,
                sequences = distance,
                outputs_info = T.zeros_like(distance[0]))
    return outputs
D = accumulate(euclidean_distances)
D2 = accumulate2(euclidean_distances)
D3 = accumulate3(euclidean_distances)
f_euclidean = theano.function([X, Y], [euclidean_distances,D,D2,D3])



for x in  f_euclidean(A,B):
    print x
