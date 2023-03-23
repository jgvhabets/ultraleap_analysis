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

def get_scores(sub, cond, cam, task, side, block):

    read_scores = pd.read_excel(
        os.path.join(
            find_paths.find_onedrive_path('patientdata'),
            'scores_JB_JH_JR.xlsx',
        ),
        usecols='A:I',
    )

    read_scores.set_index('sub_cond_cam', inplace = True)

    if side == 'left': side='lh'
    elif side == 'right': side='rh'

    # read scores for all blocks of a subject in the same cond, cam per side
    ext_scores = read_scores.loc[f'{sub}_{cond}_{cam}'][f'{task}_{side}']

    if type(ext_scores) != float:

        if isinstance(ext_scores, int):
            ls_int_sc = [ext_scores,]
        else:
            ls_int_sc = [int(s) for s in ext_scores if s in ['0', '1', '2', '3', '4']]


        if block == 'b1':
            score = ls_int_sc[0]
        elif block == 'b2':
            try:
                score = ls_int_sc[1]
            except IndexError:
                score = ls_int_sc[0]
        elif block == 'b3':
            score = ls_int_sc[2]
        else:
            print(f'no scores for block {block} or block does not exist')
            score = np.nan
        return score