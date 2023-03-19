__all__ = ["FairGen"]

import time

from algo import *

import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from itertools import product
from collections import Counter, OrderedDict

from sklearn.preprocessing import MinMaxScaler
#from pyod.models.iforest import IForest
from sklearn_extra.cluster import KMedoids
from sklearn.neighbors import NearestNeighbors as NN

import 

import warnings
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None 

class FairGen(object):
    def __init__(self, df: object, sensitive_attributes:list, class_name:str, 
                causal_reg:list = [], causal_class:list = [],
                integer_attributes:list = [], weighted_attributes:list = [], random_attributes:list = [],
                mode:str = 'Distance', ds:str = 'Fixed', no_fair:bool = False):
    

        self.df = df
        self.sensitive_attributes = sensitive_attributes
        self.class_name = class_name
        self.causal_reg = causal_reg
        self.causal_class = causal_class
        self.integer_attributes = integer_attributes  
        self.weighted_attributes = weighted_attributes  
        self.random_attributes = random_attributes
        self.mode = mode
        self.ds = ds
        self.no_fair = no_fair

    def fit(self):

        print ("FairGen is running...")
        print ("")
        print ("Target variable:", self.class_name)
        print ("Sensitive attributes:", self.sensitive_attributes)
        print ("")
        print ("Integer strategy for:", self.integer_attributes)
        print ("Weighted strategy for:", self.weighted_attributes)
        print ("Random strategy for:", self.random_attributes)
        print ("")
        print ("Regressor:", self.causal_reg)
        print ("Classifier:", self.causal_class)
        
        self.end_time = 0

        self.genetic_data = []
            
        self.weighted_indexes = []

        self.random_indexes = []
        
        self.integer_indexes = []
        
        self.regular_indexes = []

        self.causal_reg_attributes = []
        
        self.causal_class_attributes = []
                
        self.attributes = [col for col in self.df.columns if col != self.class_name]
        
        self.val_comb = []
        for att in self.sensitive_attributes:
            self.val_comb.append(list(self.df[att].unique()))
        
        self.df_combinations = list(product(*self.val_comb))
        
        print ("Ranking the data...")
        
        self.X_proba = ranker(self.df, self.attributes, self.class_name)
        
        edges = []
        
        if len(self.causal_reg) > 0:
            
            for e in self.causal_reg:
                
                X_index = []
                for feat in e[0]:
                    X_index.append(get_index(feat, self.attributes))
                    edges.append((feat, e[1]))
                y_index = get_index(e[1], self.attributes)
                self.causal_reg_attributes.append(e[1])
                
                regressor = get_causal_regressor (self.X_proba, X_index, y_index)

                e[0] = X_index
                e[1] = y_index
                e.append(regressor)
                
                # Each e in causal has:
                # e[0] = name of ind variable
                # e[1] = name of dep variable
                # we turned e[0] and e[1] into the variable indexes and we added
                # e[2] = predictor
        
        if len(self.causal_class) > 0:

            for e in self.causal_class:

                X_index = []
                for feat in e[0]:
                    X_index.append(get_index(feat, self.attributes))
                    edges.append((feat, e[1]))
                y_index = get_index(e[1], self.attributes)
                self.causal_class_attributes.append(e[1])

                classifier = get_causal_classifier (self.X_proba, X_index, y_index)

                e[0] = X_index
                e[1] = y_index
                e.append(classifier)       
        print ("!!")

        nodes = self.df.columns
    
        dag = nx.DiGraph(edges)

        dag.add_nodes_from(nodes)

        nx.draw_networkx(dag, pos = nx.circular_layout(dag), font_size=10, node_size=350, node_color='#abdbe3')

        plt.title('Assumed Ground Truth', fontsize=13)

        plt.show()
                
        self.values = self.X_proba.copy()
        
        self.target = None
        
        #PART 1 - Discrimination Test
        
        print ("Creating the sensitive dictionary...")
        
        self.sensitive_dict = get_discrimination (self.X_proba, self.sensitive_attributes, self.class_name)
        
        self.X_proba = self.X_proba.iloc[:, :-1] #removing the last column (with the ranker proba)
        
        self.og_df = self.X_proba.copy() 

        self.scaler = MinMaxScaler()

        self.df_scaled = self.scaler.fit_transform(self.og_df)

        
        #Training outlier detection methods for fitness
        
        #self.forest=IForest()

        self.forest = []

        #self.forest.fit(self.X_proba)
            
        self.X_proba = self.X_proba.values
                
        for att in self.integer_attributes:
            self.integer_indexes.append(get_index(att, self.attributes)) 


        for att in self.weighted_attributes:
            self.weighted_indexes.append(get_index(att, self.attributes))



        if self.no_fair:
            for att in self.random_attributes + self.sensitive_attributes:
                self.random_indexes.append(get_index(att, self.attributes))
            self.random_indexes.append(len(self.attributes))
            print ("Random index:", self.random_indexes)
        else:    
            for att in self.random_attributes:
                self.random_indexes.append(get_index(att, self.attributes))


            
        for att in self.attributes:
            if self.no_fair:
                if att not in self.causal_reg_attributes + self.causal_class_attributes:
                    self.regular_indexes.append((get_index(att, self.attributes)))
                    self.regular_indexes.append(len(self.attributes))
            else:
                if att not in self.sensitive_attributes + self.causal_reg_attributes + self.causal_class_attributes:
                    self.regular_indexes.append((get_index(att, self.attributes)))
                
        self.X_proba = self.X_proba.tolist() #list of every record in the dataset

    def balance(self):

        

        #### PART2 - Record Removal [of DN / PP]
        
        # We're working with X_proba (list of records) and sensitive_dictionary (various informations about records)
        #      
        print("")
        print("Removing records...")
        print("")
        
        
        record_informations = []
        # for every record removed, we append a "record information"
        # a "record information" is a tuple of length == len(sensitive_attributes)
        # each value in the tuple is the value the respective attribute had in the removed record
        
        # additionally, we'll make sure to remove the record in every list of records we have

        for att in self.sensitive_attributes:
            #### Removing DN
            for val in self.sensitive_dict[att]['D']['values_list']:
                if len(self.sensitive_dict[att]['D'][val]['N']) > self.sensitive_dict[att]['D'][val]['N_exp']:
                    to_remove = len(self.sensitive_dict[att]['D'][val]['N']) - self.sensitive_dict[att]['D'][val]['N_exp']
                    for record in self.sensitive_dict[att]['D'][val]['N'][:to_remove]:
                        record_info = []
                        for att2 in self.sensitive_attributes:
                            index = get_index(att2, self.attributes)
                            value = record[index]
                            record_info.append(value)
                            if att2 != att:
                                for val2 in self.sensitive_dict[att2]['D']['values_list']:
                                    if record in self.sensitive_dict[att2]['D'][val2]['P']:
                                        self.sensitive_dict[att2]['D'][val2]['P'].remove(record)
                                    if record in self.sensitive_dict[att2]['D'][val2]['N']:
                                        self.sensitive_dict[att2]['D'][val2]['N'].remove(record)
                                for val2 in self.sensitive_dict[att2]['P']['values_list']:
                                    if record in self.sensitive_dict[att2]['P'][val2]['P']:
                                        self.sensitive_dict[att2]['P'][val2]['P'].remove(record)
                                    if record in self.sensitive_dict[att2]['P'][val2]['N']:
                                        self.sensitive_dict[att2]['P'][val2]['N'].remove(record)                                    
                        record_informations.append(tuple(record_info))
                        self.X_proba.remove(record)
                    self.sensitive_dict[att]['D'][val]['N'] = self.sensitive_dict[att]['D'][val]['N'][to_remove:]
            
            ### Removing PP
            for val in self.sensitive_dict[att]['P']['values_list']:
                if len(self.sensitive_dict[att]['P'][val]['P']) > self.sensitive_dict[att]['P'][val]['P_exp']:
                    to_remove = len(self.sensitive_dict[att]['P'][val]['P']) - self.sensitive_dict[att]['P'][val]['P_exp']
                    for record in self.sensitive_dict[att]['P'][val]['P'][:to_remove]:
                        record_info = []
                        for att2 in self.sensitive_attributes:
                            index = get_index(att2, self.attributes)
                            value = record[index]
                            record_info.append(value)
                            if att2 != att:
                                for val2 in self.sensitive_dict[att2]['D']['values_list']:
                                    if record in self.sensitive_dict[att2]['D'][val2]['P']:
                                        self.sensitive_dict[att2]['D'][val2]['P'].remove(record)
                                    if record in self.sensitive_dict[att2]['D'][val2]['N']:
                                        self.sensitive_dict[att2]['D'][val2]['N'].remove(record)
                                for val2 in self.sensitive_dict[att2]['P']['values_list']:
                                    if record in self.sensitive_dict[att2]['P'][val2]['P']:
                                        self.sensitive_dict[att2]['P'][val2]['P'].remove(record)
                                    if record in self.sensitive_dict[att2]['P'][val2]['N']:
                                        self.sensitive_dict[att2]['P'][val2]['N'].remove(record)
                        record_informations.append(tuple(record_info))
                        self.X_proba.remove(record)
                    self.sensitive_dict[att]['P'][val]['P'] = self.sensitive_dict[att]['P'][val]['P'][to_remove:]

        if len(record_informations) == 0:
            print ("No records removed! The dataset is already balanced")
            return
            
        print ("Records removed:", len(record_informations)) #each element (list) of record information => 1 removed record
        print ("")
        print ("Current length of dataset: ", len(self.X_proba))  

        #removed combinations of sensitive attributes:
        common_combs = Counter(tuple(record_informations))
        common_combs = list(OrderedDict(common_combs.most_common()))
        
        # the unique set of "record information", ordered by their frequency
        # i.e. every unique combinations of sensitive attributes removed

        
        for comb in self.df_combinations:
            if comb not in common_combs:
                common_combs.append(comb)
                
        # additional unique combinations of sensitive attributes
        # not among those removed
        # we append them to the very end of the list => lower priority

        
        ### PART3 - Combination Test [Getting the constraints]
        
        
        constraints = []
        # a constraint is a binary tuple
        # tuple[0] = index of a feature; tuple[1] = value of the feature
        # we'll make a constraint for the sensitive attribute(s) and the target variable
        
        
        # "N_cur" => current length of the respective subset
        for att in self.sensitive_attributes:
            for val in self.sensitive_dict[att]['D']['values_list']:
                self.sensitive_dict[att]['D'][val]['N_curr'] = len(self.sensitive_dict[att]['D'][val]['N'])
                self.sensitive_dict[att]['D'][val]['P_curr'] = len(self.sensitive_dict[att]['D'][val]['P'])
                #DN_curr = self.sensitive_dict[att]['D'][val]['N_curr']
                #DN_exp = self.sensitive_dict[att]['D'][val]['N_exp']
                #DP_curr = self.sensitive_dict[att]['D'][val]['P_curr']
                #DP_exp = self.sensitive_dict[att]['D'][val]['P_exp']            
            for val in self.sensitive_dict[att]['P']['values_list']:
                self.sensitive_dict[att]['P'][val]['N_curr'] = len(self.sensitive_dict[att]['P'][val]['N'])
                self.sensitive_dict[att]['P'][val]['P_curr'] = len(self.sensitive_dict[att]['P'][val]['P'])     
                #PN_curr = self.sensitive_dict[att]['P'][val]['N_curr']
                #PN_exp = self.sensitive_dict[att]['P'][val]['N_exp']
                #PP_curr = self.sensitive_dict[att]['P'][val]['P_curr']
                #PP_exp = self.sensitive_dict[att]['P'][val]['P_exp']            

                
                
        # combinations of sensitive attributes values are ordered according to their frequency in common_combs
        # higher frequency == higher priority
        # for each value in the comb, we check if a new record with *that* value is needed
        # (i.e, if N_curr < N_exp or P_curr < P_exp)
        # if every value in the combination do pass the check (either for a negative record or a positive record)
        # we'll make a constraint with those values (and the target variable)
        # we repeat those steps as long as neither a negative or a positive record is needed
        # then we'll try the next combination
        # this way, before creating records with a less frequent combination of sens attributes values,
        # we are sure we exhausted the more frequent combinations
        
        self.start_time = time.time()
        for comb in common_combs:
            while True:
                starting_const_len = len(constraints)
                check = True
                ok_comb_neg = True
                ok_comb_pos = True
                constraint_neg = [(len(self.attributes), 0)]
                constraint_pos = [(len(self.attributes), 1)]
                for i in range(len(comb)):
                    att = self.sensitive_attributes[i]
                    val = comb[i]
                    
                    if val in self.sensitive_dict[att]['D']['values_list']:
                        if self.sensitive_dict[att]['D'][val]['N_curr'] < self.sensitive_dict[att]['D'][val]['N_exp']:
                            constraint_neg.append((get_index(att, self.attributes), val))
                        else:
                            ok_comb_neg = False 
                    elif val in self.sensitive_dict[att]['P']['values_list']:
                        if self.sensitive_dict[att]['P'][val]['N_curr'] < self.sensitive_dict[att]['P'][val]['N_exp']:
                            constraint_neg.append((get_index(att, self.attributes), val))
                        else:
                            ok_comb_neg = False
                            
                    if val in self.sensitive_dict[att]['D']['values_list']:
                        if self.sensitive_dict[att]['D'][val]['P_curr'] < self.sensitive_dict[att]['D'][val]['P_exp']:
                            constraint_pos.append((get_index(att, self.attributes), val))
                        else:
                            ok_comb_pos = False                        
                    elif val in self.sensitive_dict[att]['P']['values_list']:
                        if self.sensitive_dict[att]['P'][val]['P_curr'] < self.sensitive_dict[att]['P'][val]['P_exp']:
                            constraint_pos.append((get_index(att, self.attributes), val))
                        else:
                            ok_comb_pos = False   
                
                if ok_comb_neg == True:
                    constraints.append(tuple(constraint_neg))
                    for tup in constraint_neg[1:]:
                        att = self.attributes[tup[0]]
                        val = tup[1]
                        if val in self.sensitive_dict[att]['D']['values_list']:
                            self.sensitive_dict[att]['D'][val]['N_curr'] = self.sensitive_dict[att]['D'][val]['N_curr'] + 1
                        elif val in self.sensitive_dict[att]['P']['values_list']:
                            self.sensitive_dict[att]['P'][val]['N_curr'] = self.sensitive_dict[att]['P'][val]['N_curr'] + 1

                if ok_comb_pos == True:
                    constraints.append(tuple(constraint_pos))
                    for tup in constraint_pos[1:]:
                        att = self.attributes[tup[0]]
                        val = tup[1]
                        if val in self.sensitive_dict[att]['D']['values_list']:
                            self.sensitive_dict[att]['D'][val]['P_curr'] = self.sensitive_dict[att]['D'][val]['P_curr'] + 1
                        elif val in self.sensitive_dict[att]['P']['values_list']:
                            self.sensitive_dict[att]['P'][val]['P_curr'] = self.sensitive_dict[att]['P'][val]['P_curr'] + 1
                
                if ok_comb_neg == False and ok_comb_pos == False:
                    break
                    
                if len(constraints) == starting_const_len:
                    break

        #print("Printing stuff for debugging purposes...")
        #for att in sensitive_attributes:
        #   for val in sensitive_dict[att]['D']['values_list']:
        #      DN_curr = sensitive_dict[att]['D'][val]['N_curr']
        #     DN_exp = sensitive_dict[att]['D'][val]['N_exp']
            #    DP_curr = sensitive_dict[att]['D'][val]['P_curr']
            #   DP_exp = sensitive_dict[att]['D'][val]['P_exp']            
            #  print(att, val, "DN_cur", DN_curr, "DN_exp", DN_exp, "DP_cur", DP_curr, "DP_exp", DP_exp)
            #for val in sensitive_dict[att]['P']['values_list']: 
            #   PN_curr = sensitive_dict[att]['P'][val]['N_curr']
            #  PN_exp = sensitive_dict[att]['P'][val]['N_exp']
            # PP_curr = sensitive_dict[att]['P'][val]['P_curr']
                #PP_exp = sensitive_dict[att]['P'][val]['P_exp']            
                #print(att, val, "PN_cur", PN_curr, "PN_exp", PN_exp, "PP_cur", PP_curr, "PP_exp", PP_exp)         

        ### PART4 - Creating new records using the constraints to balance the dataset
        
        
        if self.no_fair:
            kmedoids = KMedoids(n_clusters=1, random_state=42).fit(self.df_scaled)
            medoid = kmedoids.cluster_centers_[0] #medoid of entire dataset
            const = []
            new_records = GA(self.values, const, len(record_informations), self.forest, medoid, 
                            self.weighted_indexes, self.integer_indexes, self.random_indexes, self.regular_indexes, self.causal_reg, 
                            self.causal_class, self.mode, self.ds, self.scaler)
            for all_records in new_records:
                for record in all_records:
                    self.X_proba.append(record) #balanced dataset (OG + Syntethic)
                    self.genetic_data.append(record) #other dataset with ONLY synthetic data
            
            
        else:   
            constraints = Counter(tuple(constraints))

            for const in constraints.keys():

                subset = self.og_df.copy() #subgroup of records with const values (e.g.: Black, Male, Positive)
                for tup in const:
                    subset = subset[subset[subset.columns[tup[0]]] == tup[1]]

                subset = self.scaler.transform(subset)

                if len(subset) > 0:
                    kmedoids = KMedoids(n_clusters=1, random_state=42).fit(subset)
                    medoid = kmedoids.cluster_centers_[0] #medoid of subgroup
                else:
                    kmedoids = KMedoids(n_clusters=1, random_state=42).fit(self.df_scaled)
                    medoid = kmedoids.cluster_centers_[0] #medoid of entire dataset

                #we use a GA for every const [eg. Black, Male, Positve] and each associated number [e.g. 5]
                #e.g., we create 5 records following the consts
                #if we have 20 unique consts, we use the GA 20 times

                new_records = GA(self.values, const, constraints[const], self.forest, medoid, 
                                self.weighted_indexes, self.integer_indexes, self.random_indexes, self.regular_indexes, self.causal_reg, 
                                self.causal_class, self.mode, self.ds, self.scaler)

                for all_records in new_records:
                    for record in all_records:
                        #target = const[0][1]
                        #for tup in const[1:]:
                         #   att = self.attributes[tup[0]]
                          #  val = tup[1]
                           # if val in self.sensitive_dict[att]['D']['values_list']:
                            #    if target == 0: 
                             #       self.sensitive_dict[att]['D'][val]['N'].append(record)
                              #  else:
                               #     self.sensitive_dict[att]['D'][val]['P'].append(record)
                            #elif val in self.sensitive_dict[att]['P']['values_list']:
                             #   if target == 0: 
                              #      self.sensitive_dict[att]['P'][val]['N'].append(record)
                               # else:
                                #    self.sensitive_dict[att]['P'][val]['P'].append(record)
                            #else:
                             #   print ("ERROR! ERROR!")
                              #  print ("Value", val, "shouldn't exist for attribute", att)

                        self.X_proba.append(record) #balanced dataset (OG + Syntethic)
                        self.genetic_data.append(record) #other dataset with ONLY synthetic data
        
        print ("=== NEW DATASET ===")
        self.end_time = time.time() - self.start_time
        self.final_df = pd.DataFrame.from_records(self.X_proba)
        self.final_df.columns = self.attributes + [self.class_name]
        
        self.genetic_df = pd.DataFrame.from_records(self.genetic_data)
        self.genetic_df.columns = self.attributes + [self.class_name]
        
        
        get_discrimination (self.final_df, self.sensitive_attributes, self.class_name)



        print("")
        print("OG dataset length:", len(self.df))
        print("Records generated:", len(self.genetic_df))
        print("New dataset length:", len(self.final_df))
        print("Time:", self.end_time)
        
    def get_syntethic (self):
        
        try:
            return self.genetic_df
        except:
            print("To get syntethic data, run FairGen.fit() and FairGen.balance()")

    def get_final_df (self):
        try:
            return self.final_df
        except:
            print("To get the final, balanced dataset, run FairGen.fit() and FairGen.balance()")

    def get_results (self):
        try:
            return self.final_df, self.genetic_df, self.end_time
        except:
            print("To get the results, run FairGen.fit() and FairGen.balance()")



            
