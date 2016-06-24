#-----------------------------------

#
#   In this exercise you will put the finishing touches on a perceptron class
#
#   Finish writing the activate() method by using numpy.dot and adding in the thresholded
#   activation function

import numpy

class Perceptron:

    
    def activate(self,inputs):
        '''Takes in @param inputs, a list of numbers.
        @return the output of a threshold perceptron with
        given weights, threshold, and inputs.
        ''' 
               
        
        #YOUR CODE HERE

        #TODO: calculate the strength with which the perceptron
        if len(self.weights) == len(inputs) and self.threshold is not None:
            strength = numpy.dot(inputs, self.weights)
            print strength
            result = strength > self.threshold

        else: 
            raise ValueError("Lists do not have the same shape or threshold is None")

        #TODO: return 0 or 1 based on the         
        return result

        
        
    def __init__(self,weights=None,threshold=None):
        if weights:
            self.weights = weights
        if threshold:
            self.threshold = threshold
            

perceptron = Perceptron([1,1,1], 1)

print perceptron.activate([2,2,2])