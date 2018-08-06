import numpy as np
import math


""" seed: seed values initialize randomization. Saving this value or setting it
to the same number each time guarantees that the algorithm will come up with
the same results (identical on each run).
 
Seed in machine learning means intilization state of a pseudo random number generator. If you 
use the same seed you will get exactly the same pattern of numbers.
This means that whether you're making a train test split, generating a numpy array from some random 
distribution or even fitting a ML model, setting a seed will be giving you the same set of results
 time and again."""
 
 
 
#shuffles samples data set X and y randomly
def shuffle(X, y, seed = None):
    if seed:
        np.random.seed(seed)
        #shuffles wrt the given seed
        
    indices = np.arrange(X.shape[0])
    np.random.shuffle(indices)
    #np.random.shuffle(): Modify a sequence in-place by shuffling its contents
    
    return X[indices], y[indices]
 
    
    
"""
    training set— a subset to train a model.
    test set— a subset to test the trained model.
"""  
    


#Splits data set into training and test subsets
def splitting_data(X, y, test_size = 0.5, shuffle = True, seed = None):
    
    if shuffle:
        X, y = shuffle(X, y , seed)
        
    #split training data from test data by test_size factor.
    split = len(y) - int( len(y) // (1 / test_size) )
   
    X_train_set = X[ :split] #(elements before split factor)
    X_test_set = X[split: ] #(elements after split factor)
    y_test_set = y[split: ]
    y_train_set = y[ :split]
    
    return X_train_set, X_test_set, y_train_set, y_test_set
    
    #can also use train_test_split() method of sklearn directly



#computes accuracy of y_trueoriginal y of training data w.r.t predicted y
def accuracy_score(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred, axis = 0)/ len(y_true)
    return accuracy
    #can use : " accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)" method of sklearn directly
    

#Gaussian Naive Bayes classifier
class GaussianNaiveBayes():
    
    def fit(self, X, y):
        self.X = X
        self.y = y
        
        #all the classes that y can be classified to
        self.classes = np.unique(y)
   
        self.parameters = []
        
        
        
        #Calculates mean and variance of each feature for each class
        for i, c in enumerate(self.classes):
           
            # Selects rows where label equals the given class
            #Different mean and variance for different classes and features
            X_matches_c = X[np.where(y==c)]
           
            
            #appending empty list
            self.parameters.append([])
            
            
            # Add the mean and variance for each feature (column)
            #Different mean and variance for different classes and features
            for col in X_matches_c.T:
                parameters = {"mean": col.mean(), "variance": col.var()}
                
              
                self.parameters[i].append(parameters)
          
            
    
    #Gaussian maximum likelihood of data x given mean and variance       
    def gaussian_likelihood(self, mean, var, x ):
        
        #Added the term in denominator to prevent division by zero
        eps = 1e-4 
        
        #Maximum likelihood: 1/sqare root (2 * pi * variance)  * e^(  -(x - mean)^2 / (2 * mean)    )
        coeffecient_term = 1.0/ math.sqrt(2.0 * math.pi * var + eps)
        
        exponent_term = math.exp( -( math.pow(x - mean, 2) / (2 * var + eps) ) )
        
        return coeffecient_term * exponent_term
        
                
       
    #Gaussioan prior P(y) of class c 
    def gaussian_prior(self, c):
        # prior p(y) = (samples of x where y have class c ) / (total samples in x)
        
        prior = np.mean(self.y == c)
        return prior
    
    
    # Classifies the sample (x) as the class (y) that results in the largest P(Y|X) ie posterior probability
    def classify(self, sample):
        #classify x using bayes rule
        #P(Y|X) = P(X|Y) * P(Y) / P(X) ie, Posterior Probability =   likelihood function * prior probability
        
        posteriors = []
        
        for i, c in enumerate(self.classes):
            
            #initializing posterior as prior
            posterior = self.gaussian_prior(c)
            
            #assume iid assumption: P(x1, x2, ...,xn|y) = P(x1|y) * P(x2|y) * ..... * P(xn|y)
            for feature_value, params in zip(sample, self.parameters[i]):
                """The zip() function take iterables (can be zero or more), makes iterator that 
                aggregates elements based on the iterables passed, and returns an iterator of tuples."""
                
                # # Likelihood of feature value given distribution of feature values given y
                likelihood = self.gaussian_likelihood(params["mean"], params["var"], feature_value)
                posterior *= likelihood
                
            posteriors.append(posterior)
                
        #returns class with maximum posterior probability     
        return self.classes[np.argmax(posteriors)]
        
    
    
    #Predicts the class labels of the samples in data set X
    def prdeiction(self, X):
        y_prediction = [self._classify(sample) for sample in X]
        return y_prediction
        
        
        
    
    
    
    

