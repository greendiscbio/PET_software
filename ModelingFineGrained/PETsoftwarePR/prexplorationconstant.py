"""
This file retrieves graph-like representations of hypometabolism-hypometabolism rules.
It considers a set of binary thresholds for making the results more consistent.
"""

#Import all packages and module dependencies

import sys
sys.path.insert(0,'../../Tools')
from prtools import *
da = Datasets()
rd = RuleDefinition()
rp = RulePlotting()
from regionsdict import *

#Load the datasets

Y_aal_quan = pandas.read_csv('../../Data/Y_aal_quan.csv')
Y_aal_quan.drop(['f1mo_l','f1mo_r'],axis=1,inplace=True)
Y_aal_quan_diag = pandas.read_csv('../../Data/Y_aal_quan_diag.csv')
Y_aal_quan_diag.drop(['f1mo_l','f1mo_r'],axis=1,inplace=True)
Y_aal_quan_AD = da.filter_diagnosis(Y_aal_quan_diag,[5,7,17])
Y_aal_quan_FTD = da.filter_diagnosis(Y_aal_quan_diag,[3])
Y_aal_quan_CT = da.filter_diagnosis(Y_aal_quan_diag,[13,16])

#Define and plot the rules for the different conditions

rp.fix_distances(aal_coordinates_2d,8)
viridis = matplotlib.pyplot.cm.get_cmap('viridis',8)
color_dict = {'frontal_l':viridis(7),'frontal_r':viridis(7),
              'parietal_l':viridis(6),'parietal_r':viridis(6),
              'occipital_l':viridis(5),'occipital_r':viridis(5),
              'temporal_l':viridis(4),'temporal_r':viridis(4),
              'insular_l':viridis(3),'insular_r':viridis(3),
              'subcortical_l':viridis(2),'subcortical_r':viridis(2)}

#aal

ordered_regions,percentage_dict,ordered_rules = rd.define_rules_normalised_constant(Y_aal_quan,[30,60,100,150,210],'non-hierarchical','hypometabolic','hypometabolic',0.25)
ordered_rules = rd.filter_rules_by_regions(ordered_rules,'remove','either',['f1mo_l','f1mo_r'])
rp.plot_rules(ordered_regions,aal_region_lobe,percentage_dict,aal_coordinates_2d,ordered_rules,color_dict,'../../Results/Rules/aal.png')
del ordered_regions,percentage_dict,ordered_rules

#aal_AD

ordered_regions,percentage_dict,ordered_rules = rd.define_rules_normalised_constant(Y_aal_quan_AD,[30,60,100,150,210],'non-hierarchical','hypometabolic','hypometabolic',0.25)
ordered_rules = rd.filter_rules_by_regions(ordered_rules,'remove','either',['f1mo_l','f1mo_r'])
rp.plot_rules(ordered_regions,aal_region_lobe,percentage_dict,aal_coordinates_2d,ordered_rules,color_dict,'../../Results/Rules/aal_AD.png')
filtered_rules = rd.filter_rules_by_regions(ordered_rules,'select','premise',['t3_r'])
rp.plot_rules(ordered_regions,aal_region_lobe,percentage_dict,aal_coordinates_2d,filtered_rules,color_dict,'../../Results/Rules/aal_AD_t3_r.png',True)
filtered_rules = rd.filter_rules_by_regions(ordered_rules,'select','premise',['ag_r'])
rp.plot_rules(ordered_regions,aal_region_lobe,percentage_dict,aal_coordinates_2d,filtered_rules,color_dict,'../../Results/Rules/aal_AD_ag_r.png',True)
filtered_rules = rd.filter_rules_by_regions(ordered_rules,'select','premise',['o2_r'])
rp.plot_rules(ordered_regions,aal_region_lobe,percentage_dict,aal_coordinates_2d,filtered_rules,color_dict,'../../Results/Rules/aal_AD_o2_r.png',True)
del ordered_regions,percentage_dict,ordered_rules,filtered_rules

#aal_FTD

ordered_regions,percentage_dict,ordered_rules = rd.define_rules_normalised_constant(Y_aal_quan_FTD,[30,60,100,150,210],'non-hierarchical','hypometabolic','hypometabolic',0.25)
ordered_rules = rd.filter_rules_by_regions(ordered_rules,'remove','either',['f1mo_l','f1mo_r'])
rp.plot_rules(ordered_regions,aal_region_lobe,percentage_dict,aal_coordinates_2d,ordered_rules,color_dict,'../../Results/Rules/aal_FTD.png')
filtered_rules = rd.filter_rules_by_regions(ordered_rules,'select','premise',['t3_l'])
rp.plot_rules(ordered_regions,aal_region_lobe,percentage_dict,aal_coordinates_2d,filtered_rules,color_dict,'../../Results/Rules/aal_FTD_t3_l.png',True)
filtered_rules = rd.filter_rules_by_regions(ordered_rules,'select','premise',['f1_l'])
rp.plot_rules(ordered_regions,aal_region_lobe,percentage_dict,aal_coordinates_2d,filtered_rules,color_dict,'../../Results/Rules/aal_FTD_f1_l.png',True)
filtered_rules = rd.filter_rules_by_regions(ordered_rules,'select','premise',['in_l'])
rp.plot_rules(ordered_regions,aal_region_lobe,percentage_dict,aal_coordinates_2d,filtered_rules,color_dict,'../../Results/Rules/aal_FTD_in_l.png',True)
del ordered_regions,percentage_dict,ordered_rules,filtered_rules

#aal_CT

ordered_regions,percentage_dict,ordered_rules = rd.define_rules_normalised_constant(Y_aal_quan_CT,[30,60,100,150,210],'non-hierarchical','hypometabolic','hypometabolic',0.25)
ordered_rules = rd.filter_rules_by_regions(ordered_rules,'remove','either',['f1mo_l','f1mo_r'])
rp.plot_rules(ordered_regions,aal_region_lobe,percentage_dict,aal_coordinates_2d,ordered_rules,color_dict,'../../Results/Rules/aal_CT.png')
del ordered_regions,percentage_dict,ordered_rules
