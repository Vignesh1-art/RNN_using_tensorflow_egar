import tensorflow as tf
import numpy as np


class RNN:

    def __init__(self,input_dim,output_dim,hidden_dim):
        self.U=tf.Variable(tf.random.uniform(shape=(hidden_dim,input_dim)),dtype=tf.float32,name="U")
        self.V=tf.Variable(tf.random.uniform(shape=(output_dim,hidden_dim)),dtype=tf.float32,name="V")
        self.W=tf.Variable(tf.random.uniform(shape=(hidden_dim,hidden_dim)),dtype=tf.float32,name="W")
        self.b=tf.Variable(tf.random.uniform(shape=(hidden_dim,1)),dtype=tf.float32,name="b")
        self.c=tf.Variable(tf.random.uniform(shape=(output_dim,1)),dtype=tf.float32,name="c")
        self.state=tf.Variable(tf.random.uniform(shape=(hidden_dim,1)),dtype=tf.float32,name="state")
  
    @tf.function
    def basic_rnn_cell(self,x,s):
        if(len(x.shape)==1):
            x=tf.expand_dims(x,axis=1)
        s=tf.linalg.matmul(self.U,x,name="U-cross-x")+tf.linalg.matmul(self.W,s,name="W-cross-s")+self.b
        y=tf.linalg.matmul(self.V,s,name="V-cross-s")+self.c
        return y,s
    
    @tf.function
    def rnn(self,X):
        outputs=tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
        state=self.state
        input_len=X.shape[0]
        i=0
        while(i<input_len):
            output,state=self.basic_rnn_cell(X[i],state)
            outputs=outputs.write(i,output)
            i+=1     
        return outputs.stack()
    
    @tf.function
    def loss(self,X,Y):
        X=tf.reshape(X,(X.shape[0],X.shape[1]))#Converts to 2D tensor
        diff=tf.subtract(X,Y)
        sq_diff=tf.square(diff)
        sq_diff_sum=tf.reduce_sum(sq_diff)
        return sq_diff_sum

    def get_loss_grads(self,X,Y):
        with tf.GradientTape() as tape:
            X=self.rnn(X)
            l=self.loss(X,Y)
        return(tape.gradient(l,[self.W,self.U,self.b,self.c,self.state,self.V]))

    def fit(self,X,Y,epochs=100,_learning_rate=0.001):
        optimizer=tf.keras.optimizers.Adam(learning_rate=_learning_rate)
        for _ in range(epochs):
            grads=self.get_loss_grads(X,Y)
            optimizer.apply_gradients(zip(grads,[self.W,self.U,self.b,self.c,self.state,self.V]))




            


    
#Testing
x=np.array([[0.01,0.02],[0.01,0.02]],dtype=np.float32).T
y=np.array([1,2],dtype=np.float32).T
o=RNN(2,1,5)
x=tf.constant(x)
y=tf.constant(y)
o.fit(x,y)
print(o.rnn(x))


