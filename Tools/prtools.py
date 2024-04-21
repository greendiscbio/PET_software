##########################################################################################################################################

import pandas
import numpy
import matplotlib
from random import uniform
from math import log,sqrt
from operator import itemgetter
from collections import Counter
from igraph import *

##########################################################################################################################################

class Datasets:
    """
    Class that allows defining the input datasets.
    #Methods
        filter_diagnosis(): filters a dataframe based on a given value for the diagnosis of the disease.
        define_binary(): converts a quantitative dataset into a binary dataset given a threshold.
    """

    def filter_diagnosis(self,df,diagnostic):
        """
        Function that filters a dataframe based on a given value for the diagnosis of the disease.
        :param df: (Dataframe) dataframe with one column associated to the diagnosis of the disease.
        :param diagnostic: (List) values of diagnosis that are to be filtered.
        """
        filtered_df = df.loc[df['diagnostic'].isin(diagnostic)]
        filtered_df.drop(columns=['diagnostic'],inplace=True)
        return(filtered_df)
    
    def define_binary(self,df,binary_threshold):
        """
        Function that converts a quantitative dataset into a binary dataset given a threshold.
        :param df: (Dataframe) dataframe with several patients and their brain regions associated to the number of hypometabolic voxels.
        :param binary_threshold: (Integer) threshold above which a value is considered to be hypometabolic.
        """
        binary_df = (df>binary_threshold).astype('int')
        return(binary_df)

##########################################################################################################################################
 
class RuleDefinition(Datasets):
    """
    Class that allows defining rules.
    #Methods
        define_rules_absolute(): defines a set of rules of the type 'if region A is hypometabolic/non-hypometabolic, then region B is hypometabolic/non-hypometabolic with a probability of p'
        define_rules_normalised(): defines a set of absolute rules but normalises the probabilities so to outline real relations and discard noise.
        define_rules_relative(): defines a set of rules of the type 'region A and region B have equal/unequal metabolism with a probability of p'.
        define_rules_normalised_constant(): defines a set of normalised rules over different binary thresholds, outlining those that are constant.
        define_ordered_regions(): defines valid regions based on a threshold of hypometabolic or non-hypometabolic values.
        define_probabilities_absolute(): defines the probabilities of having a hypometabolic/non-hypometabolic region when another region is hypometabolic/non-hypometabolic.
        define_probabilities_normalised(): defines the absolute probabilities but normalises them so to outline real relations and discard noise.
        define_probabilities_relative(): defines the probability of having one region with the same/opposite metabolism of another region.
        define_ordered_rules(): defines a set of rules based on previously defined probabilities.
        print_rules_absolute(): prints previously defined rules.
        print_rules_normalised(): prints previously defined rules.
        print_rules_relative(): prints previously defined rules.
        retrieve_connected_regions(): retrieves regions connected through previously defined rules.
        filter_rules_by_regions(): either selects or removes those previously defined rules that involve a certain region.
        probabilities_to_similarities(): converts an asymmetric matrix of probabilities to a symmetric matrix of similarities.
    """

    def define_rules_absolute(self,df,binary_threshold,strategy,premise,conclusion,relevance_threshold,probability_order=True,ordered_regions=None):
        """
        Function that defines a set of rules of the type 'if region A is hypometabolic/non-hypometabolic, then region B is hypometabolic/non-hypometabolic with a probability of p'.
        :param df: (Dataframe) dataframe with several patients and their brain regions associated to the number of hypometabolic voxels.
        :param binary_threshold: (Integer) threshold above which a value is considered to be hypometabolic.
        :param strategy: (String) decision on which relations are measured.
        :param premise: (String) if region X is hypometabolic/non-hypometabolic...
        :param conclusion: (String) ...then region Y is probably hypometabolic/non-hypometabolic.
        :param relevance_threshold: (Float) frequency above which a causative relation is considered.
        :param probability_order: (Boolean) whether to order or not the rules by their probability.
        :param ordered_regions: (List) list of regions that determines the order in which the rules will appear.        
        """
        binary_df = self.define_binary(df,binary_threshold)
        if probability_order==True: ordered_regions,percentage_dict = self.define_ordered_regions(binary_df,premise)
        probabilities = self.define_probabilities_absolute(binary_df,ordered_regions,strategy,premise,conclusion)
        ordered_rules = self.define_ordered_rules(probabilities,relevance_threshold,probability_order)
        return(ordered_rules)

    def define_rules_normalised(self,df,binary_threshold,strategy,premise,conclusion,relevance_threshold,probability_order=True,ordered_regions=None,return_probabilities=False):
        """
        Function that defines a set of absolute rules but normalises the probabilities so to outline real relations and discard noise.
        :param df: (Dataframe) dataframe with several patients and their brain regions associated to the number of hypometabolic voxels.
        :param binary_threshold: (Integer) threshold above which a value is considered to be hypometabolic.
        :param strategy: (String) decision on which relations are measured.
        :param premise: (String) if region X is hypometabolic/non-hypometabolic...
        :param conclusion: (String) ...then region Y is probably hypometabolic/non-hypometabolic.
        :param relevance_threshold: (Float) frequency above which a causative relation is considered.
        :param probability_order: (Boolean) whether to order or not the rules by their probability.
        :param ordered_regions: (List) list of regions that determines the order in which the rules will appear.               
        :param return_probabilities: (Boolean) if True, only the probability matrix is returned.
        """
        binary_df = self.define_binary(df,binary_threshold)
        if probability_order==True: ordered_regions,percentage_dict = self.define_ordered_regions(binary_df,premise)
        probabilities = self.define_probabilities_normalised(binary_df,ordered_regions,strategy,premise,conclusion)
        if return_probabilities==True: return(probabilities)
        ordered_rules = self.define_ordered_rules(probabilities,relevance_threshold,probability_order)
        return(ordered_rules)
   
    def define_rules_relative(self,df,binary_threshold,type_relation,relevance_threshold,probability_order=True,ordered_regions=None):
        """
        Function that defines a set of rules of the type 'region A and region B have equal/unequal metabolism with a probability of p'.
        :param df: (Dataframe) dataframe with several patients and their brain regions associated to the number of hypometabolic voxels.
        :param binary_threshold: (Integer) threshold above which a value is considered to be hypometabolic.
        :param type_relation: (String) whether the relation is of equality or unequality.
        :param relevance_threshold: (Float) frequency above which a causative relation is considered.
        :param probability_order: (Boolean) whether to order or not the rules by their probability.
        :param ordered_regions: (List) list of regions that determines the order in which the rules will appear.            
        """
        binary_df = self.define_binary(df,binary_threshold)
        if probability_order==True: ordered_regions,percentage_dict = self.define_ordered_regions(binary_df,'hypometabolic')
        probabilities = self.define_probabilities_relative(binary_df,ordered_regions,type_relation)
        ordered_rules = self.define_ordered_rules(probabilities,relevance_threshold,probability_order)
        return(ordered_rules)

    def define_rules_normalised_constant(self,df,binary_thresholds,strategy,premise,conclusion,relevance_threshold,probability_order=True,ordered_regions=None):
        """
        Function that defines a set of normalised rules over different binary thresholds, outlining those that are constant.
        :param df: (Dataframe) dataframe with several patients and their brain regions associated to a number of hipometabolic voxels.
        :param binary_thresholds: (List) list of the binary thresholds that will be considered.
        :param strategy: (String) decision on which relations are measured.
        :param premise: (String) if region X is hypometabolic/non-hypometabolic...
        :param conclusion: (String) ...then region Y is probably hypometabolic/non-hypometabolic.
        :param relevance_threshold: (Float) frequency above which a causative relation is considered.
        :param probability_order: (Boolean) whether to order or not the rules by their probability.
        :param ordered_regions: (List) list of regions that determines the order in which the rules will appear.
        """
        binary_df = self.define_binary(df,binary_thresholds[0])
        if probability_order==True: ordered_regions = self.define_ordered_regions(binary_df,premise)[0]
        for binary_threshold in binary_thresholds:
            temp_binary_df = self.define_binary(df,binary_threshold)
            temp_percentage_dict = self.define_ordered_regions(temp_binary_df,premise)[1]
            try: percentage_dict = dict(Counter(percentage_dict)+Counter(temp_percentage_dict))
            except: percentage_dict = temp_percentage_dict
            temp_probabilities_df = self.define_probabilities_normalised(temp_binary_df,ordered_regions,strategy,premise,conclusion)
            try: probabilities_df = probabilities_df + temp_probabilities_df
            except: probabilities_df = temp_probabilities_df
        for key in percentage_dict.keys(): percentage_dict[key]/=len(binary_thresholds)
        probabilities_df = probabilities_df/len(binary_thresholds)
        ordered_rules = self.define_ordered_rules(probabilities_df,relevance_threshold,probability_order)
        return(ordered_regions,percentage_dict,ordered_rules)
  
    def define_ordered_regions(self,binary_df,premise):
        """
        Function that defines ordered regions based on a threshold of hypometabolic or non-hypometabolic values.
        :param binary_df: (Dataframe) dataframe with several patients and their brain regions classified as hypometabolic or not hypometabolic.
        :param premise: (String) if region X is hypometabolic/non-hypometabolic...
        """
        all_regions = list(binary_df.columns)
        percentage_dict = {}
        for i in all_regions:
            target = numpy.array(binary_df[i])
            if premise=='hypometabolic': percentage = list(target).count(1)/len(target)
            elif premise=='non-hypometabolic': percentage = list(target).count(0)/len(target)
            if 0<percentage<1: percentage_dict[i]=percentage
        ordered_regions = sorted(percentage_dict,key=percentage_dict.get,reverse=True)
        return(ordered_regions,percentage_dict)

    def define_probabilities_absolute(self,binary_df,ordered_regions,strategy,premise,conclusion):
        """
        Function that defines the probabilities of having a hypometabolic/non-hypometabolic region when another region is hypometabolic/non-hypometabolic.
        :param binary_df: (Dataframe) dataframe with several patients and their brain regions clasified as hypometabolic or non-hypometabolic.
        :param ordered_regions: (List) list with the brain regions previously selected as valid.
        :param strategy: (String) decision on which relations are measured.
        :param premise: (String) if region X is hypometabolic/non-hypometabolic...
        :param conclusion: (String) ...then region Y is probably hypometabolic/non-hypometabolic.
        """
        probabilities_matrix = numpy.zeros((len(ordered_regions),len(ordered_regions)))
        for i in ordered_regions:
            metabolism_premise = numpy.array(binary_df[i])
            if premise=='hypometabolic': count_premise = list(metabolism_premise).count(1)
            elif premise=='non-hypometabolic': count_premise = list(metabolism_premise).count(0)
            exploration = ordered_regions.copy()
            if strategy=='non-hierarchical': exploration.remove(i)
            elif strategy=='hierarchical': exploration = exploration[exploration.index(i)+1:]
            for j in exploration:
                metabolism_conclusion = numpy.array(binary_df[j])
                metabolism_argument = numpy.core.defchararray.add(metabolism_premise.astype(str),metabolism_conclusion.astype(str))
                if premise=='hypometabolic' and conclusion=='hypometabolic': count_argument = list(metabolism_argument).count('11')
                if premise=='hypometabolic' and conclusion=='non-hypometabolic': count_argument = list(metabolism_argument).count('10')
                if premise=='non-hypometabolic' and conclusion=='hypometabolic': count_argument = list(metabolism_argument).count('01')
                if premise=='non-hypometabolic' and conclusion=='non-hypometabolic': count_argument = list(metabolism_argument).count('00')
                try: 
                    probability_argument = count_argument/count_premise
                    probabilities_matrix[ordered_regions.index(i),ordered_regions.index(j)] = probability_argument
                except: pass
        probabilities_df = pandas.DataFrame(probabilities_matrix,index=ordered_regions,columns=ordered_regions)
        probabilities_df = round(probabilities_df,2)
        return(probabilities_df)

    def define_probabilities_normalised(self,binary_df,ordered_regions,strategy,premise,conclusion):
        """
        Function that defines the absolute probabilities but normalises them so to outline real relations and discard noise.
        :param binary_df: (Dataframe) dataframe with several patients and their brain regions clasified as hypometabolic or non-hypometabolic.
        :param ordered_regions: (List) list with the brain regions previously selected as valid.
        :param strategy: (String) decision on which relations are measured.
        :param premise: (String) if region X is hypometabolic/non-hypometabolic...
        :param conclusion: (String) ...then region Y is probably hypometabolic/non-hypometabolic.
        """
        probabilities_matrix = numpy.zeros((len(ordered_regions),len(ordered_regions)))
        for i in ordered_regions:
            metabolism_premise = numpy.array(binary_df[i])
            if premise=='hypometabolic': 
                count_premise = list(metabolism_premise).count(1)
                #count_not_premise = list(metabolism_premise).count(0)
            elif premise=='non-hypometabolic': 
                count_premise = list(metabolism_premise).count(0)
                #count_not_premise = list(metabolism_premise).count(0)
            exploration = ordered_regions.copy()
            if strategy=='non-hierarchical': exploration.remove(i)
            elif strategy=='hierarchical': exploration = exploration[exploration.index(i)+1:]
            for j in exploration:
                metabolism_conclusion = numpy.array(binary_df[j])
                metabolism_argument = numpy.core.defchararray.add(metabolism_premise.astype(str),metabolism_conclusion.astype(str))
                if premise=='hypometabolic' and conclusion=='hypometabolic': 
                    count_argument = list(metabolism_argument).count('11')
                    count_not_argument = list(metabolism_argument).count('01')
                if premise=='hypometabolic' and conclusion=='non-hypometabolic': 
                    count_argument = list(metabolism_argument).count('10')
                    count_not_argument = list(metabolism_argument).count('00')
                if premise=='non-hypometabolic' and conclusion=='hypometabolic': 
                    count_argument = list(metabolism_argument).count('01')
                    count_not_argument = list(metabolism_argument).count('11')
                if premise=='non-hypometabolic' and conclusion=='non-hypometabolic': 
                    count_argument = list(metabolism_argument).count('00')
                    count_not_argument = list(metabolism_argument).count('10')
                #probability_argument = count_argument/count_premise
                #probability_not_argument = count_not_argument/count_not_premise
                #probability_normalised = max(probability_argument-probability_not_argument,0)
                try:
                    probability_normalised = max(((count_argument-count_not_argument)/count_premise),0)
                    probabilities_matrix[ordered_regions.index(i),ordered_regions.index(j)] = probability_normalised
                except: pass
        probabilities_df = pandas.DataFrame(probabilities_matrix,index=ordered_regions,columns=ordered_regions)
        probabilities_df = round(probabilities_df,2)
        return(probabilities_df)
    
    def define_probabilities_relative(self,binary_df,ordered_regions,type_relation):
        """
        Function that defines the probability of having one region with the same/opposite metabolism of another region.
        :param binary_df: (Dataframe) dataframe with several patients and their brain regions clasified as hypometabolic or non-hypometabolic.
        :param ordered_regions: (List) list with the brain regions previously selected as valid.
        :param type_relation: (String) whether the relation is of equality or unequality.
        In this case, causality makes no sense, which explains why there is no strategy defined
        """
        probabilities_matrix = numpy.zeros((len(ordered_regions),len(ordered_regions)))
        for i in ordered_regions:
            metabolism_first = numpy.array(binary_df[i])
            exploration = ordered_regions[ordered_regions.index(i)+1:]
            for j in exploration:
                metabolism_second = numpy.array(binary_df[j])
                metabolism_both = numpy.core.defchararray.add(metabolism_first.astype(str),metabolism_second.astype(str))
                if type_relation=='equal': count_both = list(metabolism_both).count('11')+list(metabolism_both).count('00')
                elif type_relation=='unequal': count_both = list(metabolism_both).count('10')+list(metabolism_both).count('01')
                try:
                    probability_both = count_both/len(metabolism_first)
                    probabilities_matrix[ordered_regions.index(i),ordered_regions.index(j)] = probability_both
                except: pass
        probabilities_df = pandas.DataFrame(probabilities_matrix,index=ordered_regions,columns=ordered_regions)
        probabilities_df = round(probabilities_df,2)
        return(probabilities_df)
    
    def define_ordered_rules(self,probabilities_df,relevance_threshold,probability_order):
        """
        Function that defines a set of ordered rules based on previously defined probabilities.
        :param probabilities_df: (Dataframe) dataframe with the probability of having a hypometabolic/non-hypometabolic region when another region is hypometabolic/non-hypometabolic.
        :param relevance_threshold: (Float) frequency above which a causative relation is considered.
        :param probability_order: (Boolean) whether to order or not the rules by their probability.
        """
        rules = {}
        for i in probabilities_df.index:
            for j in probabilities_df.columns:
                if probabilities_df[j][i]>relevance_threshold:
                    rules[i,j]=probabilities_df[j][i]
        if probability_order==True: ordered_rules = sorted(rules.items(),key=itemgetter(1),reverse=True)
        else: ordered_rules = list(rules.items())
        return(ordered_rules)
    
    def print_rules_absolute(self,ordered_rules,premise,conclusion):
        """
        Function that prints previously defined rules.
        :param ordered_rules: (List) pairs of regions that represent rules and their associated probabilities.
        :param premise: (String) if region X is hypometabolic/non-hypometabolic...
        :param conclusion: (String) ...then region Y is probably hypometabolic/non-hypometabolic.
        """
        for rule in ordered_rules:
            print('When '+str(rule[0][0])+' is '+str(premise)+', '+str(rule[0][1])+' is '+str(conclusion)+' with a probability of '+str(rule[1]))

    def print_rules_normalised(self,ordered_rules,premise,conclusion):
        """
        Function that prints previously defined rules.
        :param ordered_rules: (List) pairs of regions that represent rules and their associated probabilities.
        :param premise: (String) if region X is hypometabolic/non-hypometabolic...
        :param conclusion: (String) ...then region Y is probably hypometabolic/non-hypometabolic.
        """
        for rule in ordered_rules:
            print('When '+str(rule[0][0])+' is '+str(premise)+', '+str(rule[0][1])+' is '+str(conclusion)+' with a normalised probability of '+str(rule[1]))
    
    def print_rules_relative(self,ordered_rules,type_relation):
        """
        Function that prints previously defined rules.
        :param ordered_rules: (List) pairs of regions that represent rules and their associated probabilities.
        :param type_relation: (String) whether the relation is of equality or unequality.
        """
        for rule in ordered_rules:
            print(str(rule[0][0])+' and '+str(rule[0][1])+' have '+str(type_relation)+' metabolism with a probability of '+str(rule[1]))

    def retrieve_connected_regions(self,ordered_rules):
        """
        Function that retrieves regions connected through previously defined rules.
        :param ordered_rules: (List) pairs of regions that represent rules and their associated probabilities.
        """
        connected_regions = []
        for rule in ordered_rules: 
            connected_regions.append(rule[0][0])
            connected_regions.append(rule[0][1])
        connected_regions = list(set(connected_regions))
        return(connected_regions)

    def filter_rules_by_regions(self,ordered_rules,operation,direction,filtered_regions):
        """
        Function that either selects or removes those previously defined rules that involve a certain region.
        :param ordered_rules: (List) pairs of regions that represent rules and their associated probabilities.
        :param operation: (String) whether to select or remove the indicated rules.
        :param direction: (String) whether to consider those rules in which the region is premise, conclusion or either.
        :param filtered_regions: (List) regions to be selected or removed from the rules.
        """
        rules_to_filter = []
        for rule in ordered_rules:
            if direction=='premise':
                if rule[0][0] in filtered_regions: rules_to_filter.append(rule)
            elif direction=='conclusion':
                if rule[0][1] in filtered_regions: rules_to_filter.append(rule)
            elif direction=='either':
                if rule[0][0] in filtered_regions or rule[0][1] in filtered_regions:
                    rules_to_filter.append(rule)
        if operation=='select': new_rules = rules_to_filter
        elif operation=='remove':
            new_rules = ordered_rules.copy()
            for rule in rules_to_filter: new_rules.remove(rule)
        return(new_rules)
    
    def probabilities_to_similarities(self,probabilities_df):
        """
        Function that converts an asymmetric matrix of probabilities to a symmetric matrix of similarities.
        :param probabilities_df: (DataFrame) dataframe with the probability of having a hypometabolic/non-hypometabolic region when another region is hypometabolic/non-hypometabolic.
        """
        distance_df = pandas.DataFrame(index=probabilities_df.index,columns=probabilities_df.index)
        for i in probabilities_df.columns:
            for j in probabilities_df.index:
                if i==j: distance_df[i][j] = 1
                if i!=j: distance_df[i][j] = (probabilities_df[i][j]+probabilities_df[j][i])/2
        return(distance_df)

    
##########################################################################################################################################

class RuleApplication:
    """
    Class that allows applying rules.
    #Methods
        apply_rules(): applies previously defined rules on the predicted data.
        compare_with_real(): compares the predicted output with the real outputs, giving the corresponding metrics and confusion data.
    """

    def apply_rules(self,predicted_binary_df,ordered_rules,premise,conclusion):
        """
        Function that applies previously defined rules on the predicted data.
        :param predicted_binary_df: (Dataframe) dataframe with several patients and their brain regions classified as hypometabolic or non-hypometabolic.
        :param ordered_rules: (List) pairs of regions that represent rules and their associated probabilities.
        :param premise: (String) if region X is hypometabolic/non-hypometabolic...
        :param conclusion: (String) ...then region Y is probably hypometabolic/non-hypometabolic.
        """
        corrected_binary_df = predicted_binary_df.copy()
        for patient in corrected_binary_df.index:
            for rule in ordered_rules:
                if uniform(0,1)<rule[1]:
                    if premise=='hypometabolic' and conclusion=='hypometabolic':
                        if corrected_binary_df[rule[0][0]][patient]==1: corrected_binary_df[rule[0][1]][patient]=1
                    if premise=='hypometabolic' and conclusion=='non-hypometabolic':
                        if corrected_binary_df[rule[0][0]][patient]==1: corrected_binary_df[rule[0][1]][patient]=0
                    if premise=='non-hypometabolic' and conclusion=='hypometabolic':
                        if corrected_binary_df[rule[0][0]][patient]==0: corrected_binary_df[rule[0][1]][patient]=1
                    if premise=='non-hypometabolic' and conclusion=='non-hypometabolic':
                        if corrected_binary_df[rule[0][0]][patient]==0: corrected_binary_df[rule[0][1]][patient]=0
        return(corrected_binary_df)

    def compare_with_real(self,predicted_binary_df,real_binary_df):
        """
        Function that compares the predicted output with the real outputs, giving the corresponding metrics and confusion data.
        :param predicted_binary_df: (Array) predicted values for the output.
        :param real_binary_df: (Dataframe) real values for the output.
        """
        comparison = predicted_binary_df.apply(lambda col: col.astype(str)) + real_binary_df.apply(lambda col: col.astype(str))
        comparison = comparison.replace({'11':'TP','00':'TN','10':'FP','01':'FN'})
        comp_list = list(comparison.values.flatten())
        TP = comp_list.count('TP'); TN = comp_list.count('TN'); FP = comp_list.count('FP'); FN = comp_list.count('FN')
        accuracy = (TP+TN)/(TP+TN+FP+FN+1e-10); precision = TP/(TP+FP+1e-10); recall = TP/(TP+FN+1e-10); f1 = 2*(precision*recall)/(precision+recall+1e-10)
        confusion = pandas.DataFrame({'TP':[TP],'TN':[TN],'FP':[FP],'FN':[FN],'accuracy':[accuracy],'f1':[f1],'precision':[precision],'recall':[recall]})
        confusion = round(confusion,3)
        return(confusion)

##########################################################################################################################################

class RulePlotting(RuleDefinition):
    """
    Class that allows drawing the rules in networks.
    #Methods
        plot_rules(): draws previously defined rules in a network.
        plot_clusters(): draws previously defined clusters.
        fix_distances(): separates the vertex so that they do not overlap.
        keep_fixing_distances(): determines whether to continue moving the vertex.
    """
    
    def plot_rules(self,vertex,lobes,percentages,positions,edges,color_dict,file_name,color_connected_regions=False):
        """
        Function that draws previously defined rules in a networks.
        :param vertex: (List) list of the regions that will act as vertex.
        :param lobes: (Dictionary) correspondencies between each brain region and the lobe it belongs to.
        :param percentages: (Dictionary) correspondencies between each brain region and the percentage of patients that have hypometabolism in it.
        :param positions: (Dictionary) correspondencies between each brain region and its spatial coordinates.
        :param edges: (List) list of the relations between regions and their associated probability.
        :param color_dict: (Dictionary) dictionary with the attribute values and associated colors.
        :param file_name: (String) name that will be given to the image.
        :param color_connected_regions: (Dictionary) whether to color only the connected regions.
        """
        g = Graph(directed=True)
        for i in range(len(vertex)):
            g.add_vertex(name=vertex[i])
            g.vs[i]['lobe'] = lobes[vertex[i]]
            g.vs[i]['percentage'] = percentages[vertex[i]]
            if positions is not None: g.vs[i]['position'] = positions[vertex[i]]
        for i in range(len(edges)):
            g.add_edge(edges[i][0][0],edges[i][0][1])
            g.es[i]['probability'] = edges[i][1]
            g.es[i]['input_lobe'] = lobes[edges[i][0][0]]
        visual_style = {}
        visual_style['vertex_label'] = [i.replace('_l','').replace('_r','') for i in g.vs['name']]
        visual_style['vertex_label_size'] = [abs(1/log(i))*50 for i in g.vs['percentage']]
        visual_style['vertex_size'] = [abs(1/log(i))*125 for i in g.vs['percentage']]
        visual_style['vertex_color'] = [color_dict[i] for i in g.vs['lobe']]
        visual_style['vertex_frame_width'] = 0
        visual_style['edge_width'] = [i*10 for i in g.es['probability']]
        visual_style['edge_arrow_size'] = [i+1 for i in g.es['probability']]
        visual_style['edge_arrow_width'] = [i+1 for i in g.es['probability']]
        visual_style['edge_curved'] = -0.5
        visual_style['edge_color'] = [color_dict[i] for i in g.es['input_lobe']] 
        visual_style['bbox'] = (2500,2500)
        visual_style['margin'] = 200
        if positions is not None: visual_style['layout'] = g.vs['position']
        if color_connected_regions==True:
            indexes = [vertex.index(i) for i in self.retrieve_connected_regions(edges)]
            for i in range(len(visual_style['vertex_color'])):
                if i not in indexes: visual_style['vertex_color'][i]='grey'
        p = plot(g, **visual_style)
        p.save(file_name)

    def plot_clusters(self,vertex,clusters,positions,file_name):
        """
        Function that draws previously defined clusters.
        :param vertex: (List) list of the regions that will act as vertex.
        :param clusters: (Dictionary) correspondencies between each brain region and the cluster it belongs to.
        :param positions: (Dictionary) correspondencies between each brain region and its spatial coordinates.
        :param file_name: (String) name that will be given to the image.
        """
        g = Graph(directed=True)
        for i in range(len(vertex)):
            g.add_vertex(name=vertex[i])
            g.vs[i]['cluster'] = clusters[vertex[i]]
            if positions is not None: g.vs[i]['position'] = positions[vertex[i]]
        visual_style = {}
        visual_style['vertex_label'] = [i.replace('_l','').replace('_r','') for i in g.vs['name']]
        visual_style['vertex_label_size'] = 35
        visual_style['vertex_size'] = 100
        visual_style['vertex_frame_width'] = 0
        visual_style['bbox'] = (2500,2500)
        visual_style['margin'] = 200
        color_dict = {}
        unique_clusters = list(set(list(clusters.values())))
        viridis = matplotlib.pyplot.cm.get_cmap('gist_rainbow',len(unique_clusters))
        for i in range(len(unique_clusters)): color_dict[unique_clusters[i]]=viridis(i)
        visual_style['vertex_color'] = [color_dict[i] for i in g.vs['cluster']]
        if positions is not None: visual_style['layout'] = g.vs['position']
        p = plot(g, **visual_style)
        p.save(file_name)
    
    def fix_distances(self,coordinates_2d,minimum_distance):
        """
        Function that separates the vertex so that they do not overlap.
        :param coordinates_2d: (Dictionary) dictionary with the regions and their associated 2d coordinates.
        :param minimum_distance: (Integer) minimum desired distance between the vertex.
        """
        for i in coordinates_2d.keys():
            for j in coordinates_2d.keys():
                if i!=j:
                    ci = coordinates_2d[i]
                    cj = coordinates_2d[j]
                    distance = sqrt((ci[0]-cj[0])**2+(ci[1]-cj[1])**2)
                    if distance<minimum_distance:
                        abs_ci = abs(ci[0])+abs(ci[1])
                        abs_cj = abs(cj[0])+abs(cj[1])
                        while distance<minimum_distance:
                            if abs_ci>abs_cj: ci*=1.05; cj*=0.95
                            else: ci*=0.95; cj*=1.05
                            distance = sqrt((ci[0]-cj[0])**2+(ci[1]-cj[1])**2)
                    coordinates_2d[i] = ci
                    coordinates_2d[j] = cj
        if self.keep_fixing_distances(coordinates_2d,minimum_distance)==True: self.fix_distances(coordinates_2d,minimum_distance)
    
    def keep_fixing_distances(self,coordinates_2d,minimum_distance):
        """
        Function that determines whether to continue moving the vertex
        :param coordinates_2d: (Dictionary) dictionary with the regions and their associated 2d coordinates.
        :param minimum_distance: (Integer) minimum desired distance between the vertex.
        """
        for i in coordinates_2d.keys():
            for j in coordinates_2d.keys():
                if i!=j:
                    ci = coordinates_2d[i]
                    cj = coordinates_2d[j]
                    distance = sqrt((ci[0]-cj[0])**2+(ci[1]-cj[1])**2)
                    if distance<minimum_distance: return(True)
        return(False)

##########################################################################################################################################