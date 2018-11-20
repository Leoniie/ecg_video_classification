import pandas as pd
import numpy as np
import os


def inputter_csv(file):
    path = os.path.abspath( file )
    array = pd.read_csv( path )
    array = array[ 'y' ].values
    array = np.reshape( array, (array.shape[ 0 ], 1) )

    print( "Input of {} done! \n Array has shape of {}.".format( file, array.shape ) )

    return array


def inputter_avi(file):
    # pass is a placeholder for the real function
    pass


def outputter(array, path):
    y = pd.DataFrame( array, dtype=np.dtype( 'U25' ) )
    ids = list( range( 0, y.shape[ 0 ] ) )
    ids = pd.DataFrame( ids )
    output = pd.concat( [ ids, y ], axis=1 )
    output.columns = [ "id", "y" ]

    # TODO: name of the output file is 'daytime_solution.csv'

    output.to_csv( path_or_buf='output\\solution.csv', sep=',', na_rep='', float_format='U25',
                   header=True, index=False,
                   mode='w', encoding=None, compression=None,
                   quoting=None, quotechar='"', line_terminator='\n', chunksize=None, tupleize_cols=None,
                   date_format=None, doublequote=True, escapechar=None, decimal='.' )
