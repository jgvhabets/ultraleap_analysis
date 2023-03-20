import json


def loadjson_as_dict(path):
    f = open(path)
  
    data = json.load(f)

    return data


def savedict_as_json(path, 
    file, 
    data
):
    filename = file.replace('dist', 'feat')
    file_path_name = f'{path}/{filename}.json'
    with open(file_path_name, 'w') as fp:
        json.dump(data, fp)
    return