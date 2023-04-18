import json
import os
import import_data.find_paths as find_paths

def loadjson_as_dict(path):
    f = open(path)

    return json.load(f)


def savedict_as_json(path, 
    file, 
    data
):
    filename = file.replace('dist', 'feat').replace('.csv', '')
    file_path_name = f'{path}/{filename}.json'
    with open(file_path_name, 'w') as fp:
        json.dump(data, fp)
    return
