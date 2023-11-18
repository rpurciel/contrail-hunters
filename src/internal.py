import os
import glob

def str_to_bool(string):
    if string in ['true', 'True', 'TRUE', 't', 'T', 'yes', 'Yes', 'YES', 'y', 'Y']:
        return True
    elif string in ['false', 'False', 'FALSE', 'f', 'F', 'no', 'No', 'NO', 'n', 'N']:
        return False
    else:
        if string == True:
            return True
        elif string == False:
            return False
        else:
            return False #fallback to false

def clean_idx(directory):
    idx_files = glob.glob(os.path.join(directory, "*.idx"))

    if not idx_files:
        return False
    else:
        for file in idx_files:
            os.remove(file)
        return True

