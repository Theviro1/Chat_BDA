from freemod import freeMOD_infer, freeMOD_load

def prepare():
    # user provide: input_case.txt\formulas_raw.txt\ineqs.txt

    # generate formulas.txt base on input_case.txt and formulas_raw.txt
    freeMOD_load.handle_sum()
    # generate default_bound.txt base on formulas.txt
    freeMOD_load.create_default_bound()
    # generate bound.txt base on ineqs.txt and default_bound.txt
    freeMOD_load.handle_ineqs()
    
    # now we have input_case.txt\formulas_raw.txt\ineqs.txt\formulas.txt\default_bound.txt\bound.txt, everything is ready
    # load to neo4j graph
    freeMOD_load.load_database()

def run():
    freeMOD_infer.run()