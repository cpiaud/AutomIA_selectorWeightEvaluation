import configparser
import pandas as pd
import os
def load_config(script_dir,config_file_name):
    config_file=os.path.join(script_dir, 'configs', config_file_name)
    config = configparser.ConfigParser()
    config.read(config_file)
    if config_file_name=='config_gpt.txt':

       file_path = config.get('base', 'file_path')
       output_file = config.get('base', 'output_file')
       columns_csv=config.get('columns_csv', 'cols').split(', ')

       return file_path,output_file,columns_csv
    else:
        
        file_path = config.get('base', 'file_path')
        output_file = config.get('base', 'output_file')
        num_clusters = config.getint('base', 'num_clusters')
        properties_str = config['properties_dict']
        print(properties_str.items())
        properties_dict = {}
        for key, value in properties_str.items():
            cleaned_key = key.strip().strip("'")
            cleaned_value = int(value.strip().rstrip(','))
            properties_dict[cleaned_key] = cleaned_value

        prop_columns = config.get('prop_columns', 'prop_columns').split(', ')
        columns_csv=config.get('columns_csv', 'cols').split(', ')

        return file_path,columns_csv, properties_dict, prop_columns, num_clusters, output_file
def load_data(file_path,usecols):
    try:
        data = pd.read_csv(file_path, sep=";", usecols=usecols)
        return data
    except FileNotFoundError:
        print(f"Erreur: Le fichier {file_path} n'a pas été trouvé.")
        return None