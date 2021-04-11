"""
Codey Phoun
CS123B/CS223 Programming Assignment
10/19/2020

Objective:
    Implement the Viterbi algorithm to find the 
    most probable state path and the path probability
    of a given DNA sequence

    Start, transition, and emission probabilities provided
    by Dr. Wesley
"""

import os
import sys
from math import *

class HiddenMarkov:

    #A 
    statesSet = {'S1', 'S2', 'S3'}
    
    """
    NOTE:  The probability numbers indicated below are in log2 form.  That is,
    the probability number =  2^number so for example 2^-1 = 0.5,  and 
    2^-2 = 0.25 ... etc.   The reason for this is that Log2 is used to represent
    probabilities so that underflow or overflow is not encountered.  
    
    If you wish to change the probabilities in the tables below, than simple calculate 
    Log2(probability that you want/need).  For example Log2(0.33) = -0.6252,  and 
    Log2(0.5) = -1,  and so forth. 
    """
    
    #pi:
    start_probability = {'S1': -2, 'S2': -1, 'S3': -2}
    #tau:
    trans_probability = {'S1': {'S1': -1,  'S2': -1.321928,     'S3': -3.321928},
                         'S2'  : {'S1': -float('Inf'),  'S2': -1.,     'S3': -1.},
                         'S3': {'S1': -1.736966,  'S2': -2.321928,     'S3': -1.}}
    #e:
    emit_propability = {'S1': {'A': -1.321928,    'C': -1.736966,  'T': -2.321928,  'G': -3.321928},
                        'S2': {'A': -2.,   'C': -2., 'T': -2., 'G': -2.},
                        'S3': {'A': -3.321928,    'C': -2.321928,  'T': -1.736966,  'G': -1.321928}}

    def __init__(self, input_seq):
        self.input_seq = input_seq

    # Viterbi function written by Codey Phoun
    def viterbi(self, obs, states, start_p, trans_p, emit_p):
        # V is a list of dictionaries containing the hidden state probabilities of each nucleotide
        V = [] 
        path = {}

        states = list(states)

        # initialize V and path
        for i in range(len(obs)):
            path[i] = {}
            V.append({})

        # initialize time 0 with first observation
        for i in range(len(states)):
            V[0][states[i]] = start_p[states[i]] + emit_p[states[i]][obs[0]]
            path[0][states[i]] = 0

        # perform main recursion
        for t in range(1,len(obs)):
            for j in range(len(states)):
                prob, psi = max([(emit_p[states[j]][obs[t]] + trans_p[states[i]][states[j]] + V[t-1][states[i]], i) for i in range(len(states))])
                V[t][states[j]] = prob
                path[t][states[j]] = states[psi]

        # get final path probability and final state
        final_state, prob  = max(V[-1].items(), key = lambda x: x[1])
        final_path = []
        final_path.insert(0, final_state)
        prev = final_state

        # perform backtrace for most probable path
        for i in range(len(path)-1, 0, -1):
            final_path.insert(0, path[i][prev])
            prev = path[i][prev]
        
        # convert from log2
        transformed_prob = 2**prob
        
        print("Most probable state path: " + " ".join(final_path))
        print("Final path probability (log2): " + str(prob))
        print("Final path probability: " + str(transformed_prob))
        
        # write results to output file
        viterbi_out = open("CodeyPhoun_viterbi_output.txt", 'w')
        viterbi_out.write("HMM input sequence: " + input_seq + "\n")
        viterbi_out.write("Most probable state path: " + " ".join(final_path) + "\n")
        viterbi_out.write("Final path probability (log2): " + str(prob) + "\n")
        viterbi_out.write("Final path probability: " + str(transformed_prob) + "\n")
        viterbi_out.close()

        return prob, final_path

if __name__ == '__main__':

    # Set working directory to script's current directory
    file_path = os.path.abspath(sys.argv[0])
    directory = os.path.dirname(file_path)
    os.chdir(directory)

    with open("hmm_input_seq.txt") as f:
        input_seq = f.readline().strip().upper()
        print("HMM input sequence: ", input_seq)
        hm = HiddenMarkov(input_seq)
        hm.viterbi(input_seq, hm.statesSet, hm.start_probability, hm.trans_probability, hm.emit_propability)
