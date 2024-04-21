"""
This file stores important information about the brain regions, including their associated lobes, spatial coordinates and volume
"""

#Import the packages and module dependencies

import pandas
import numpy
from sklearn.decomposition import PCA

#Load the data

aal_information = pandas.read_csv('../../Data/aal_information.csv')
brodmann_information = pandas.read_csv('../../Data/brodmann_information.csv')
aal_information['Coordinates_3d'] = aal_information[['X', 'Y','Z']].values.tolist()

#Define the functions
  
def inverse_dictionary(original_dictionary):
    new_dictionary = {}
    for key,value in original_dictionary.items():
        try: new_dictionary[value].append(key)
        except: new_dictionary[value]=[key]
    return(new_dictionary)

#Define the dictionaries

aal_region_lobe = aal_information[['Region','Lobe']].set_index('Region').to_dict()['Lobe']
aal_lobe_region = inverse_dictionary(aal_region_lobe)
aal_coordinates_3d = aal_information[['Region','Coordinates_3d']].set_index('Region').to_dict()['Coordinates_3d']
aal_coordinates_2d = dict(zip(aal_coordinates_3d.keys(),PCA(n_components=2).fit_transform(list(aal_coordinates_3d.values()))))
aal_volume = aal_information[['Region','Volume']].set_index('Region').to_dict()['Volume']

brodmann_region_lobe = brodmann_information[['Region','Lobe']].set_index('Region').to_dict()['Lobe']
brodmann_lobe_region = inverse_dictionary(brodmann_region_lobe)