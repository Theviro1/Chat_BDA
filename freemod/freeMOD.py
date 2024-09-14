from freemod import freeMOD_infer, freeMOD_load

def prepare():
    # user provide: input_case.txt\formulas_raw.txt\ineqs.txt

    # check the syntax in formulas_raw.txt
    legal = freeMOD_load.handle_format()
    if not legal: return False
    # generate input.txt base on input_case.txt
    freeMOD_load.handle_input()
    # generate formulas.txt base on input_case.txt and formulas_raw.txt
    freeMOD_load.handle_i()
    freeMOD_load.handle_sum()
    # generate default_bound.txt base on formulas.txt
    freeMOD_load.create_default_bound()
    # generate bound.txt base on ineqs.txt and default_bound.txt
    freeMOD_load.handle_ineqs()
    
    # now we have input_case.txt\formulas_raw.txt\ineqs.txt\formulas.txt\default_bound.txt\bound.txt, everything is ready
    # load to neo4j graph
    freeMOD_load.load_database()
    return True

def infer():
    return freeMOD_infer.run()

def run():
    success = prepare()
    if success: return infer()
    else: print('loading failed, please follow the instruction and fix your input files')