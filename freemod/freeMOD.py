from freemod import freeMOD_infer, freeMOD_load

from utils.logs import FreeMODLogger

logger = FreeMODLogger()

def prepare():
    # user provide: input_case.txt\formulas_raw.txt\ineqs.txt

    # check the syntax in formulas_raw.txt
    logger.info('checking format in formula knowledge files...')
    legal = freeMOD_load.handle_format()
    if not legal: return False

    try:
        # generate input.txt base on input_case.txt
        logger.info('creating final input based on input_case...')
        freeMOD_load.handle_input()
        # generate formulas.txt base on input_case.txt and formulas_raw.txt
        logger.info('handling all the "_i" stuff in formulas...')
        freeMOD_load.handle_i()
        freeMOD_load.handle_sum()
        # generate default_bound.txt base on formulas.txt
        logger.info('creating default bound...')
        freeMOD_load.create_default_bound()
        # generate bound.txt base on ineqs.txt and default_bound.txt
        logger.info('updating bound based on ineqs...')
        freeMOD_load.handle_ineqs()
    
        # now we have input_case.txt\formulas_raw.txt\ineqs.txt\formulas.txt\default_bound.txt\bound.txt, everything is ready
        # load to neo4j graph
        logger.info('loading data into neo4j...')
        freeMOD_load.load_database()
    except:
        logger.error('error in loading process, please check your input and try again')
        return False
    logger.info('loading successfully finished')
    return True

def infer():
    logger.info('executing automated inferring machine..')
    return freeMOD_infer.run()

def run():
    success = prepare()
    if success: 
        logger.info('inferring successfully finished')
        return infer()
    else: logger.error('loading failed, please follow the instruction and fix your input files')