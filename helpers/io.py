def Inputter():

    # pass is a placeholder for the real function
    pass

    # TODO: define input arguments

    # TODO: reformat the input to numpy array

    # TODO: return numpy array


def Outputter():

    # pass is a placeholder for the real function
    pass

    # TODO: define the arguments

    # TODO: name of the output file is 'daytime_solution.csv'

    # TODO: generate output file in folder './output'

    # no return required

def unzip(path_from, path_to):
    import zipfile
    zip_ref = zipfile.ZipFile( path_from, 'r' )
    zip_ref.extractall( path_to )
    zip_ref.close()