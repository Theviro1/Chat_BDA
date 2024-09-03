import numpy as np
import collections
from py2neo import Graph
import scipy.optimize
import sympy
import scipy
import ast
import json
import re
from freemod.config import NEO4J_URL, NEO4J_PASSWORD, NEO4J_USERNAME, INPUT_CASE_PATH, OUTPUT_PATH, ERR_RANGE

input_relas = []
input_paras = []
input_pid_sparse = []
input_pid_origin = []
input_rid_params = []

def parse_case(file_path):
    with open(file_path, 'r') as f:
        case = f.read()
    return ast.literal_eval(case)

def load_inputs():
    global input_paras, input_relas, input_pid_origin, input_pid_sparse, input_rid_params
    # load values
    input_case = parse_case(INPUT_CASE_PATH)
    # load paras&relas from neo4j
    graph = Graph(NEO4J_URL, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    all_paras = []
    para_num = graph.run("Match (n:Parameter) return count(n)").data()[0]['count(n)']
    for node in graph.run("Match (n:Parameter) return properties(n) order by n.PID").data():
        all_paras.append(node['properties(n)'])
    from collections import Counter, defaultdict
    all_relas = []
    for node in graph.run("Match (n:Relation) return properties(n) order by n.RID").data():
        all_relas.append(node['properties(n)'])
    # reset values base on input
    input_paras = all_paras
    input_pid_sparse = [0] * para_num
    input_pid_origin = [0] * para_num
    # refresh other attributes, set the value as DefaultValue
    for param in input_paras:
        param['Value'] = param['ParaName']
        param['RecordValue'] = param['Value']
        param['ConflictInfo'] = None
        param['InferredValue'] = None
        param['InferredPath'] = None
        param['isVisited'] = 0

    for name in input_case.keys():
        for param in input_paras:
            if param['ParaName'] == name:
                param['Value'] = input_case[name]
                param['DefaultValue'] = input_case[name]  # set default value as the input value
                input_pid_sparse[param['PID']] = 1
                input_pid_origin[param['PID']] = 1
    
    # load relas and set attributes
    input_relas = all_relas
    for rela in input_relas:
        rela['isConflict'] = 1
    
    # set initial inputs relationship in input_rid_params
    input_rid_params = collections.defaultdict(lambda: defaultdict(list))
    inputted = [input_paras[i] for i in range(para_num) if input_pid_origin[i] == 1]
    for input in inputted:
        for rid in input['Rn']:
            input_rid_params[rid]['PIDs'].append(input['PID'])
    for rid in range(len(input_relas)):
        input_rid_params[rid]['cnt_PIDs'] = len(input_rid_params[rid]['PIDs'])



def full_exp(rid):
    # check whether all the params are inferred/inputted in a formula
    for pid in input_relas[rid]['Pn']:
        if input_pid_sparse[pid] == 0:
            return False
    return True

def get_exp(rid, with_x=False, pid=-1)->str:
    # return the value form of a formula
    lost_pid = pid
    lost_name = input_paras[pid]['ParaName']
    lost_value = str(input_paras[pid]['Value'])
    formula = str(input_relas[rid]['Formula'])
    # replace other params with exact value
    Pn_sort_by_len = [[input_paras[pid]['ParaName'], pid] for pid in input_relas[rid]['Pn']]
    Pn_sort_by_len.sort(key=lambda x: len(x[0]), reverse=True)
    Pn_sort_by_len = [x[1] for x in Pn_sort_by_len]
    for pid in Pn_sort_by_len:
        name = input_paras[pid]['ParaName']
        if pid != lost_pid:    
            value = str(input_paras[pid]['Value'])
            formula = formula.replace(name, value)
    # replace the target one with 'x'
    if with_x: 
        formula = formula.replace(lost_name, 'x')
    else:
        formula = formula.replace(lost_name, lost_value)
    return formula


def check_exp(rid):
    # check whether formula is satisfied
    # only check full formula
    if not full_exp(rid): return True
    formula = get_exp(rid)
    if '=' in formula:
        left, right = [part.strip() for part in formula.split('=')][:2]
        result = sympy.simplify(left + ' - ({}) '.format(right)).evalf()
        if not -ERR_RANGE <= result <= ERR_RANGE:
            print_conflict(rid)
            return False  # exp is not satisfied
        return True
    elif '>' in formula:
        result = sympy.simplify(formula)
        if result == False:
            print_conflict(rid)
            return False  # exp is not satisfied
        return True


def cal_exp_eq(rid, pid):
    # calculate the value of pid base on formula rid, rid is required to be an equality expr
    formula = get_exp(rid, with_x=True, pid=pid)
    lost_pid = pid
    # use sympy to calculate lost value
    left, right = [part.strip() for part in formula.split('=')][:2]
    results = sympy.solve(left + ' - ( {} ) '.format(right), 'x')
    # if the function is solvable
    if len(results) > 0:
        # obtain calculate result
        result = results[0].evalf()
        # 1. if pid has never been inferred before: set InferredValue, add formula to it's InferPath
        if input_paras[lost_pid]['InferredValue'] is None:
            # if result is clipped due to boundary limitation, this means formula is incorrect and raise a conflict
            input_paras[lost_pid]['InferredValue'] = result
            input_paras[lost_pid]['InferredPath'] = rid
            old = result
            new = float(np.clip(old, input_paras[lost_pid]['Domain'][0], input_paras[lost_pid]['Domain'][1]))
            if old != new:
                input_paras[lost_pid]['ConflictInfo'] = ('out of bound', result)
                return True
            return False
            # 2. if pid has been inferred and the value is not conflict: do nothing
        elif result - ERR_RANGE <= input_paras[lost_pid]['InferredValue'] <= result + ERR_RANGE:
            return False
        # 3. if pid has been inferred and the value is conflict: set Conflict as the tuple of two conflict formula rid
        else:
            input_paras[lost_pid]['ConflictInfo'] = ('formula collapse', result)
            return True
        #  if the function cannot be solved
    else:
        input_paras[lost_pid]['ConflictInfo'] = ('unsolvable', None)
        return True
        

def do_exp(rid, pid):
    # deal with different kind of exp in different way, such as those contains 'Max/Min' '>' '='
    formula = input_relas[rid]['Formula']
    if '=' in formula:
        is_conflict = cal_exp_eq(rid, pid)
        if is_conflict:
            print_conflict(rid, pid)
        return is_conflict
    if '>' in formula:
        pass  # TODO


# functions of print
def reset_isVisited():
    for param in input_paras:
        param['isVisited'] = 0

def print_conflict(rid, lost_pid=-1):
    print('during inferring encounter an error:', end='')
    reset_isVisited()
    if lost_pid == -1:
        print('after inferred error\n')
        for pid in input_relas[rid]['Pn']:
            if input_pid_origin[pid] == 1:
                print(f'original input {input_paras[pid]["ParaName"]} = {input_paras[pid]["Value"]}')
            else:
                print_conflict_dfs(input_paras[pid]['InferredPath'], pid)
                reset_isVisited()
        exp = get_exp(rid)
        print(f'\nafter infering, in formula {input_relas[rid]["Formula"]} there is a conflict:')
        print(f'formula {exp} cannot be satisfied')
    # encounter an error when calculating lost_pid
    else:
        name, result = input_paras[lost_pid]['ConflictInfo']
        print(name + '\n')
        if name == 'out of bound':
            print_conflict_dfs(rid, lost_pid)
            print(f'\nvalue of {input_paras[lost_pid]["ParaName"]} get {result} which is out of the bound [{input_paras[lost_pid]["Domain"][0]}, {input_paras[lost_pid]["Domain"][1]}]')
        elif name == 'unsolvable':
            print_conflict_dfs(rid, lost_pid)
            print(f'\nvalue of {input_paras[lost_pid]["ParaName"]} is not solvable')
        elif name == 'formula collapse':
            print('in a different formula, we have:')
            print_conflict_dfs(input_paras[lost_pid]['InferredPath'], lost_pid)
            reset_isVisited()
            input_paras[lost_pid]['InferredValue'] = result
            print('\nin this formula we have:')
            print_conflict_dfs(rid, lost_pid)
            print(f'\nvalue of {input_paras[lost_pid]["ParaName"]} has conflict')

def print_conflict_dfs(rid, lost_pid):
    # reset visited
    other_pid = [pid for pid in input_relas[rid]['Pn'] if pid != lost_pid]
    for pid in other_pid:
        if input_paras[pid]['isVisited'] == 1:
            continue
        if input_pid_origin[pid] == 1:
            print(f'original input {input_paras[pid]["ParaName"]} = {input_paras[pid]["Value"]}')
            input_paras[pid]['isVisited'] = 1
            continue
        print_conflict_dfs(input_paras[pid]['InferredPath'], pid)
        input_paras[pid]['isVisited'] = 1
    print(f'using formula {input_relas[rid]["Formula"]} inferred {input_paras[lost_pid]["ParaName"]} = {input_paras[lost_pid]["InferredValue"]}')
            


def update_infer(pids):
    inferred_pids = pids
    for pid in range(len(input_paras)):
        if input_paras[pid]['InferredValue'] is not None:
            input_paras[pid]['Value'] = input_paras[pid]['InferredValue']
            input_pid_sparse[pid] = 1
    for lost_pid in inferred_pids:
        for rid in input_paras[lost_pid]['Rn']:
            if lost_pid not in input_rid_params[rid]['PIDs']:
                input_rid_params[rid]['PIDs'].append(lost_pid)
                input_rid_params[rid]['cnt_PIDs'] += 1

def auto_infer():
    flag = 1
    while(flag):
        flag = 0
        inferred_pids = []  # record of all the pids inferred in this round
        for rid in range(len(input_relas)):
            # to all the relas, if it can be inferred, calculate the lost param's value
            if input_relas[rid]['CountPn'] - input_rid_params[rid]['cnt_PIDs'] == 1:
                lost_pid = [pid for pid in input_relas[rid]['Pn'] if pid not in input_rid_params[rid]['PIDs']][0]
                is_conflict = do_exp(rid, lost_pid)
                if is_conflict:
                    return False
                else: 
                    inferred_pids.append(lost_pid)
                    flag = 1
        # 1. no more params can be inferred, quit loop
        # 2. conflict happened, quit loop
        # after a round is finished, all the conflicts are solved, set those inferred value as the real value, and update the record in input_rid_params
        update_infer(inferred_pids)
        # double check whether other relas is satisfied
        for rid in range(len(input_relas)):
            if not check_exp(rid): return False
    print('finish auto infer')
    return True


eq_formula_infos = []
eq_formula_exps = []
eq_param_dict = {}
eq_param_init = []

def get_exps():
    # after inferring, replace all the inferred param with it's value.
    for rid in range(len(input_relas)):
        if full_exp(rid):
            continue
        exp = get_exp(rid)
        left, right = [part.strip() for part in exp.split('=')][:2]
        exp = left + ' - ( {} )'.format(right)
        eq_formula_exps.append(exp)
    # return the tuple of (function, [symbols]) of an expression using sympy processing
    for exp in eq_formula_exps:
        exp_ = sympy.simplify(exp)
        symbols = list(exp_.free_symbols)
        func = sympy.lambdify(symbols, exp_)
        symbols = [str(symbol) for symbol in symbols]
        info = (func, symbols)
        eq_formula_infos.append(info)
    # get all the params and return a diction, matching parameter's name with it's positional number
    symb_dict = ['+', '-', '*', ',' , '/', '<', '>', '(', ')', '=', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
    func_dict = ['Min', 'Sum', 'Max']
    i = 0
    for exp in eq_formula_exps:
        eles = exp.split(' ')
        all_paras = set(eles) - set(symb_dict) - set(func_dict)
        for para in all_paras:
            if re.match(r'^-?\d+(\.\d+)?(e-?\d+)?$', para):
                continue
            if para not in eq_param_dict.keys():
                pid = [param['PID'] for param in input_paras if param['ParaName']==para][0]
                eq_param_init.append(input_paras[pid]['DefaultValue'])  # these params is not in input_pid_sparse, their value have never been inputted or inferred, take their default value as initial value
                # create a diction, key is the param name, value is it's positional number, it will be quite useful in scipy
                eq_param_dict[para] = i
                i += 1
 

def equations(vars):
    # get the value of var base on it's positional number. For example if eq_param_dict['a'] = 0, vars[0] will be the input value of 'a' 
    results = []
    for func, params in eq_formula_infos:
        input_dict = {}
        # get all the param name in this formula, get their positional number and subsequently obtain their value in list 'vars'
        for param in params:
            value = vars[eq_param_dict[param]]
            if param not in input_dict:
                input_dict[param] = value
        result = func(**input_dict)
        results.append(result)
    return results


def auto_gen():
    get_exps()
    # using least square to calculate approximate value
    solution = scipy.optimize.least_squares(equations, eq_param_init)
    for i in range(len(solution.x)):
        val = solution.x[i]
        key = [k for k, v in eq_param_dict.items() if v==i][0]
        pid = [param['PID'] for param in input_paras if param['ParaName']==key][0]
        input_paras[pid]['Value'] = val
        input_pid_sparse[pid] = 1
    for rid in range(len(input_relas)):
        if not check_exp(rid):  return False
    print('finish auto generate')
    return True
    
    
def output():
    out_dict = {}
    for param in input_paras:
        if input_pid_sparse[param['PID']] == 1:
            out_dict[param['ParaName']] = float(param['Value'])
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(out_dict, f, indent=4)


def run():
    load_inputs()
    inferable = auto_infer()
    if not inferable: 
        print('input unsolvable')
        return 
    solvable = auto_gen()
    if not solvable: 
        print('formulas unsolvable')
        return
    output()