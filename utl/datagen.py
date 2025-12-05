import numpy as np

def data_gen(inputs, batch_size):
    c = 0
    
    index = list(inputs.keys())

    while (True):
        batch_feats = np.loadtxt(inputs[index[c]][0], dtype=np.float32)
        if inputs[index[c]][1] == 0:
            batch_label = np.zeros((np.shape(batch_feats)[0],1))
        elif inputs[index[c]][1] == 1:
            batch_label = np.ones((np.shape(batch_feats)[0],1))
        
        pos = np.zeros([np.shape(batch_feats)[0], 1])
        for j in range(np.shape(batch_feats)[0]):
            pos[j,0] = j/np.shape(batch_feats)[0]
            
        
        c += batch_size
        if (c+batch_size >= len(inputs)):
            c = 0
            np.random.shuffle(index)
        
        yield {"image":batch_feats, "position":pos}, batch_label
        
            
            
            
            
            
            
        
