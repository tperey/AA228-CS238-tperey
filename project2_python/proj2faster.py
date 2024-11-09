import numpy as np
import pandas as pd
import time


# QLearning class
class QLearning:
    def __init__(self, S, A, gamma, Q, alpha):
        self.S = S # State space
        self.A = A # Actino space
        self.gamma = gamma # Discount factor
        self.Q = Q # Q table (to be updated)
        self.alpha = alpha # Learning rate

# Read in csv and convert to numpy matrix
def extractRLdata(file_name):
    frame = pd.read_csv(file_name)
    data_matrix = frame.to_numpy()

    return data_matrix

def updateQ(model: QLearning, s, a, r, s_new):
    gamma, Q, alpha = model.gamma, model.Q, model.alpha

    Q[s,a] += alpha*(r + gamma*np.max(Q[s_new,:]) - Q[s,a])

    return model

def learnQwrapper(model: QLearning, data_matrix, iterations):

    # In this project, all states and actions start at 1. Convert to start at 0.
    matrix[:,0] = matrix[:,0] - 1 # Subtract one from sample states so they range 0 up
    matrix[:,1] = matrix[:,1] - 1 # Subtract one from sample actions so they range 0 up
    matrix[:,3] = matrix[:,3] - 1 # Subtract one from next states so they range 0 up

    # Learn Q
    for it in range(iterations): # Number of times to run through the data

        for sample in data_matrix: # Update model according to all samples
            s = sample[0]
            a = sample[1]
            r = sample[2]
            s_new = sample[3]

            model = updateQ(model, s, a, r, s_new)
        
        # Track iterations
        # if (it%10 == 0): # Print every 10 iterations
        #     print("Current iteration = ", it)
    
    # Extract policy vector
    Q = model.Q
    policy = np.argmax(Q, axis = 1) # Get action (column index) giving max Q for each state (row)

    return policy
    
if __name__ == "__main__": 
    pass

    """ SMALL """
    # Get data
    matrix = extractRLdata("./data/small.csv")
    #print(np.min(matrix[:,0]))

    # Build up Q model
    S = np.arange(1,100 + 1) # State space. Set for small
    A = np.arange(1,4 + 1) # Action space. Set for small
    gamma = 0.95 # Discount factor
    Q = np.zeros((len(S),len(A))) # Matrix of zeros for holding Q values. Uses S and A to get sizes
    alpha = 0.1 # Learning rate

    model = QLearning(S, A, gamma, Q, alpha)

    # Get results
    print("Starting small")
    start = time.time()

    iterations = 100

    learned_policy = learnQwrapper(model, matrix, iterations)
    learned_policy += 1 # Go back to appropriate action representation
    
    elapsed = time.time() - start
    print("   small Runtime: ", elapsed, " s, ", elapsed/60, " min")

    # Save results
    np.savetxt('small.policy', learned_policy, delimiter='\n', fmt = '%d')
    print("Done with small")



    """ MEDIUM """
    matrix = extractRLdata("./data/medium.csv")
    #print(np.min(matrix[:,0]))

    # Build up Q model
    S = np.arange(1,50000 + 1) # State space. Set for medium
    A = np.arange(1,7 + 1) # Action space. Set for medium
    gamma = 1.0 # Discount factor
    Q = np.zeros((len(S),len(A))) # Matrix of zeros for holding Q values. Uses S and A to get sizes
    alpha = 0.4 # Learning rate

    model = QLearning(S, A, gamma, Q, alpha)

    # Get results 
    print("Starting Medium")
    start = time.time()

    iterations = 500

    learned_policy = learnQwrapper(model, matrix, iterations)
    learned_policy += 1 # Go back to appropriate action representation
    elapsed = time.time() - start
    print("   medium Runtime: ", elapsed, " s, ", elapsed/60, " min")

    # Save results
    np.savetxt('medium.policy', learned_policy, delimiter='\n', fmt = '%d')
    print("Done with Medium")



    # """ LARGE """
    matrix = extractRLdata("./data/large.csv")
    # print(np.min(matrix[:,0]))
    # print(np.min(matrix[:,1]))
    # print(np.min(matrix[:,3]))

    # Build up Q model
    S = np.arange(1,302020 + 1) # State space. Set for large
    A = np.arange(1,9 + 1) # Action space. Set for large
    gamma = 0.95 # Discount factor
    Q = np.zeros((len(S),len(A))) # Matrix of zeros for holding Q values. Uses S and A to get sizes
    alpha = 0.2 # Learning rate

    model = QLearning(S, A, gamma, Q, alpha)

    # Get results 
    print("Starting LARGE")
    start = time.time()

    iterations = 100

    learned_policy = learnQwrapper(model, matrix, iterations)
    learned_policy += 1 # Go back to appropriate action representation
    
    elapsed = time.time() - start
    print("   LARGE Runtime: ", elapsed, " s, ", elapsed/60, " min")

    # Save results
    np.savetxt('large.policy', learned_policy, delimiter='\n', fmt = '%d')
    print("Done with LARGE")






    

