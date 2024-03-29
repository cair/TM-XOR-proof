import numpy as np

s = 10

DiaSum = np.zeros([256,1])
M = np.zeros([256,256])
Mx = np.zeros([254,1,8])
AbsorbingStates = np.zeros([2,1,8])
#My = np.zeros([254,1,8])
Input = np.zeros([4,2])
Input[1,1],Input[2,0],Input[3,0],Input[3,1] = 1,1,1,1

k = 0
for i in range(256):
    if i == 105 or i == 150:
        continue
    k += 1
    binary = "{0:08b}".format(i)
    x = 0
    #Mx[k-1,1,0] = i
    for j in binary:
        Mx[k-1,0,x] = int(j)
        x += 1

y = 0
for i in (105,150):
    binary = "{0:08b}".format(i)
    x = 0
    #Mx[k-1,1,0] = i
    for j in binary:
        AbsorbingStates[y,0,x] = int(j)
        x += 1
    y += 1

Mx = np.vstack((AbsorbingStates,Mx))
My = Mx

def CalculateClasueOutputs(TADecisions, Input):   
    TADecisions = np.reshape(TADecisions, (2, 4), order='F')
         
    Clause1 = 1
    if (TADecisions[0,0] == 1 and Input[0] == 0) or (TADecisions[1,0] == 1 and Input[0] == 1) or (TADecisions[0,1] == 1 and Input[1] == 0) or (TADecisions[1,1] == 1 and Input[1] == 1):
        Clause1 = 0
        
    Clause2 = 1
    if (TADecisions[0,2] == 1 and Input[0] == 0) or (TADecisions[1,2] == 1 and Input[0] == 1) or (TADecisions[0,3] == 1 and Input[1] == 0) or (TADecisions[1,3] == 1 and Input[1] == 1):
        Clause2 = 0
        
    return Clause1, Clause2


def ActivationProbability(FeedbackType, v):
    if FeedbackType == 2:
        if v == 0:
            ActProbability = 0.5
        else:
            ActProbability = 1
    else:
        if v == 0:
            ActProbability = 0.5
        else:
            ActProbability = 0.0
            
    return ActProbability 
            

def getLiteral(index, Input):
    if index % 2 == 0:
        literalType = 1
    else: 
        literalType = 0

    if index % 4 < 2:
        INx = 0
    else:
        INx = 1

    if literalType == 1:
        literal = Input[INx]
    else:
        literal = int(not Input[INx])
        
    return literal

def FeedbackProbability(FeedbackType, ClauseOutput, Literal, taDecision, change):
    #print('FeedbackType', FeedbackType, 'ClauseOutput', ClauseOutput, 'Literal', Literal, 'taDecision', taDecision, 'change', change)
    FeedProb = 0.0
    Feedrew = 0.0
    Feedinact = 0.0
    if change == 1:
        if FeedbackType == 1:
            if ClauseOutput == 1:
                if Literal == 1:
                    if taDecision == 0:
                        FeedProb = (s-1)/s
    
            else:
                if Literal == 1:
                    if taDecision == 1:
                        FeedProb = 1/s
    
                else:
                    if taDecision == 1:
                        FeedProb = 1/s
    
        else:
            if ClauseOutput == 1:
                if Literal == 0:
                    if taDecision == 0:
                        FeedProb = 1
    else:
        if FeedbackType == 1:
            if ClauseOutput == 1:
                if Literal == 1:
                    if taDecision == 1:
                        Feedrew = (s-1)/s
                        Feedinact = 1/s
                    else: 
                        Feedinact = 1/s
                else:
                    if taDecision == 0:
                        Feedrew = 1/s
                        Feedinact = (s-1)/s
            else:
                if taDecision == 1:
                    Feedinact = (s-1)/s
                else:
                    Feedrew = 1/s
                    Feedinact = (s-1)/s
        else:
            if ClauseOutput == 1:
                if Literal == 1:
                    Feedinact = 1.0
            else:
                Feedinact = 1.0
                        
                
                
    if change == 1:                
        FeedProb = FeedProb
    else:
        FeedProb = Feedrew + Feedinact
        #print(FeedProb)  
    return FeedProb


for xaxis in range(len(Mx)):
    for yaxis in range(len(My)):
        p = 0
        for case in range(4): # different Inputs
            if case == 0 or case == 3:
                FeedbackType = 2
            else:
                FeedbackType = 1
                
            Clause1, Clause2 = CalculateClasueOutputs(Mx[xaxis,:,:],Input[case,:])
            Clause1Prime, Clause2Prime = CalculateClasueOutputs(My[yaxis,:,:],Input[case,:])
            ActProb = ActivationProbability(FeedbackType, Clause1+Clause2)
            
            if np.array_equal(Mx[xaxis,:,0:4], My[yaxis,:,0:4]):#clause1, no change case
                FeedProb = 1
                for index in range(4):
                    literal = getLiteral(index, Input[case,:])
                    change = 0
                    FeedProb = FeedProb * FeedbackProbability(FeedbackType, Clause1, literal, Mx[xaxis,:,index], change)
                
                Pc1 = (ActProb * FeedProb) + (1 - ActProb)
                
            else:# clause1, change case
                FeedProb = 1
                for index in range(4):
                    literal = getLiteral(index, Input[case,:])
                    if Mx[xaxis,:,index] == My[yaxis,:,index]:
                        change = 0
                    else: 
                        change = 1
                    
                    FeedProb = FeedProb * FeedbackProbability(FeedbackType, Clause1, literal, Mx[xaxis,:,index], change)
                
                Pc1 = ActProb * FeedProb
                                
            if np.array_equal(Mx[xaxis,:,4:8], My[yaxis,:,4:8]):#clause2, no change case
                FeedProb = 1
                for kk in range(4):
                    index = kk+4
                    literal = getLiteral(index, Input[case,:])
                    change = 0
                    FeedProb = FeedProb * FeedbackProbability(FeedbackType, Clause2, literal, Mx[xaxis,:,index], change)
                
                Pc2 = (ActProb * FeedProb) + (1 - ActProb)
                                
            else:#clause2, change case
                FeedProb = 1
                for kk in range(4):
                    index = kk+4
                    literal = getLiteral(index, Input[case,:])
                    if Mx[xaxis,:,index] == My[yaxis,:,index]:
                        change = 0
                    else: 
                        change = 1
                    
                    FeedProb = FeedProb * FeedbackProbability(FeedbackType, Clause2, literal, Mx[xaxis,:,index], change)
                
                Pc2 = ActProb * FeedProb
                
                
            p += 0.25 * Pc1 * Pc2
            #print(Pc1, Pc2)

        M[yaxis, xaxis] = p
        

MM = M.transpose() 
        
for j in range(256):
    DiaSum[j,0] = sum(MM[j,:]) # sum of all row
    
np.savetxt('transitionmatrixstandard.csv', MM)
            

mm=np.linalg.matrix_power(MM, 1000000)
