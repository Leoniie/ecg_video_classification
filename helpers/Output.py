import numpy as np
import pandas as pd

def output_generator(y, df_test):
    # Output Formatting
    y = pd.DataFrame(y, dtype=np.dtype('U25'))

    Id = list(range(0, df_test.shape[0]))
    Id = pd.DataFrame(Id)

    output = pd.concat([Id, y], axis=1)
    output.columns = ["id", "y"]

    # Output generation
    auth.authenticate_user()
    gauth = GoogleAuth()
    gauth.credentials = GoogleCredentials.get_application_default()
    drive = GoogleDrive(gauth)

    output.to_csv(path_or_buf='solution.csv', sep=',', na_rep='', float_format='U25',
                  header=True, index=False,
                  mode='w', encoding=None, compression=None,
                  quoting=None, quotechar='"', line_terminator='\n', chunksize=None, tupleize_cols=None,
                  date_format=None, doublequote=True, escapechar=None, decimal='.')

    file = drive.CreateFile({'parents': [{u'id': '12oEvaQx6AYlWSggKR2WXcxXhRuBWnBIk'}]})
    file.SetContentFile('solution.csv')
    file.Upload()
