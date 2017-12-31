import numpy as np


#  use epsilon to catch div/0 errors
epsilon = 1e-5


class Processor(object):
    """
    A base class for Processor objects.
    
    args
        length (int) the length of the array
                      this corresponds to the second dimension of the shape
                      shape = (num_samples, length)
        use_history (bool) whether to use the history to estimtate parameters
    """
    def __init__(self, length, use_history):
        self.length = length
        self.use_history = use_history
        
    def transform(self, batch):
        """
        Preprocesses array into 2 dimensions.

        args
            batch (np.array)
            
        returns
            transformed_batch (np.array) shape=(num_samples, length)
        """
        if batch.ndim > 2:
            raise ValueError('batch dimension is greater than 2')
        
        #  reshape into two dimensions
        batch = np.reshape(batch, (-1, self.length))
        
        return self._transform(batch)
        
class Normalizer(Processor):
    """
    Processor object for performing normalization to range [0,1]
    Normalization = (val - min) / (max - min)
    
    args
        length (int) the length of the array
                      this corresponds to the second dimension of the shape
                      shape=(num_samples, length)
        use_history (bool) whether to use the history to estimtate parameters
        global_space (GlobalSpace) used to initialize history
    """
    def __init__(self, length, use_history=True, space=None):
        #  initialize the parent Processor class
        super().__init__(length, use_history)
        
        if space:
            #  if we include a space object we use this and don't ever use history
            self.mins = np.array([spc.low for spc in space.spaces])
            self.maxs = np.array([spc.high for spc in space.spaces])
            self.use_space = True
            self.use_history = False
        else:
            self.use_space = False
            self.mins = None
            self.maxs = None
        
    def _transform(self, batch):
        """
        Transforms a batch (shape=(num_samples, length))
        
        if we are using a space -> use mins & maxs set in __init__
        if using history -> min & max over history + batch
        else -> min & max over batch
        
        args
            batch (np.array)
        returns
            transformed_batch (np.array) shape=(num_samples, length)
        """
        
        if self.use_space:
            #  keep the original mins & maxs set in parent __init__
            pass
        
        elif self.use_history:
            #  idea is to create an array (hist) that includes the previous min & max
            #  and the batch.  we then min & max over hist.  
            if self.mins is None and self.maxs is None:
                #  catch the case where we haven't initialized mins or maxs yet
                hist = batch              
            else:
                #  create an array to min & max over from the previous min/max
                #  and our batch
                hist = np.concatenate([self.mins, self.maxs, batch]).reshape(-1, self.length)
            
            #  perform the min & max operatons over the hist array
            self.mins = hist.min(axis=0).reshape(1, self.length)
            self.maxs = hist.max(axis=0).reshape(1, self.length)        
            
        else:
            #  if we aren't using history, we use statistics from the batch
            self.mins = batch.min(axis=0).reshape(1, self.length)
            self.maxs = batch.min(axis=0).reshape(1, self.length)
        
        #  perform the min & max normalization
        return (batch - self.mins) / (self.maxs - self.mins + epsilon)
    
    
class Standardizer(Processor):
    """
    Processor object for performing standardization
    Standardization = scaling for zero mean, unit variance
    
    Algorithm from post by Dinesh 2011 on Stack Exchange:
    https://math.stackexchange.com/questions/20593/calculate-variance-from-a-stream-of-sample-values
    
    Statistics are calculated online, without keeping entire history (ie each batch)
    
    Idea is to keep three running counters
        sum(x)
        sum(x^2)
        N (count)
    
    We can then calculate historical statistics by:
        mean = sum(x) / N
        variance = 1/N * [sum(x^2) - sum(x)^2 / N]
        standard deviation = sqrt(variance)
        
    args
        length (int) the length of the array
                      this corresponds to the second dimension of the shape
                      shape=(num_samples, length)
        use_history (bool) whether to use the history to estimtate parameters    
    """
    def __init__(self, length, use_history=True):
        #  initialize the parent Processor class
        super().__init__(length, use_history)
        
        #  setup initial statistics
        self.count = 0
        self.sum = np.zeros(shape=(1, self.length))
        self.sum_sq = np.zeros(shape=(1, self.length))
        
    def _transform(self, batch):
        """
        Transforms a batch shape=(num_samples, length).
        
        If we are using history, we make use of the three counters.
        
        Else we process the batch using the mean & standard deviation
        from the batch.
        
        args
            batch (np.array)
            
        returns
            transformed_batch (np.array) shape=(num_samples, length)
        """
        if self.use_history:
            #  update our three counters
            self.sum = np.sum(np.concatenate([self.sum, batch]),
                              axis=0).reshape(1, self.length)
            self.sum_sq = np.sum(np.concatenate([self.sum_sq, batch**2]),
                                 axis=0).reshape(1, self.length)
            self.count += batch.shape[0]

            #  calculate the mean, variance and standard deivation
            self.means = self.sum / self.count
            var = (self.sum_sq - self.sum**2 / self.count) / self.count
            self.stds = np.sqrt(var)
        
        else:
            #  calculate mean & standard deivation from the batch
            self.means, self.stds = batch.mean(axis=0), batch.std(axis=0)
        
        #  perform the de-meaning & scaling by standard deivation
        return (batch - self.means) / (self.stds + epsilon)
            
