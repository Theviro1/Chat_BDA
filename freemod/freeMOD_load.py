import collections
import json
import ast
from uu import Error
import numpy as np
from py2neo import Node, Relationship, Graph, NodeMatcher, RelationshipMatcher
from freemod.config import *
import re
import sympy

def input_case(file_path):
    with open(file_path, 'r') as f:
        case = f.read()
    return ast.literal_eval(case)

def handle_format():
    with open(FORMULAS_RAW_PATH, 'r') as f:
        formulas = f.readlines()
    pattern = r'([=\(\)\+\-\*/,<>] |\w+ )*([=\(\)\+\-\*/,<>]|\w+)'
    format_formulas = []
    indexs = []
    i = 1
    for formula in formulas:
        # lexer check
        if re.match(pattern, formula.strip()) is None:
            indexs.append(i)
            print(f'invalid syntax at line {i}, please check your file!')
            continue
        # gramma check
        gramma = formula.replace('Sum', '')
        if '=' in formula:
            left, right = [part.strip() for part in gramma.split('=')][:2]
            gramma = left + ' - ({}) '.format(right)
        try:
            sympy.simplify(gramma)
        except Exception as e:
            indexs.append(i)
            print(f'in line {i},' + str(e))
        # if success, strip it
        format_formulas.append(formula.strip() + '\n')
        i += 1
    if len(indexs) != 0: 
        print(f'please check line {indexs}')
        return False
    with open(FORMULAS_RAW_PATH, 'w') as f:
        f.writelines(format_formulas)
    return True

def get_params(formula)->list[str]:
    symb_dict = ['+', '-', '*', ',' , '/', '<', '>', '(', ')', '=', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
    func_dic = ['Min', 'Sum', 'Max', 'ceiling', 'floor']
    elements = formula.strip().split(' ')
    all_params = set(elements) - set(symb_dict) - set(func_dic)
    params = []
    for param in all_params:
        if re.match(r'^-?\d+(\.\d+)?(e-?\d+)?$', param):
            continue
        params.append(param)
    return params

record = []
# This function aim to deal with the expressions that contains _i in formulas_raw.txt, try infer all the _i that can be inferred and record them in an global array, they can be used in handle_sum()
def handle_i():
    # relations of _i is complicated, try infer how many _i can be inferred, and pretend they are inputted
    case = input_case(INPUT_PATH)
    with open(FORMULAS_RAW_PATH, 'r') as f:
        formulas = f.readlines()
    formulas_with_i = []
    # get all the formula that contains _i, except Sum expression, assume left part of the Sum expression(which is result) doesn't contain _i, thus all _i is input not output, neither can it infer any _i
    for formula in formulas:
        if 'Sum' in formula: continue
        params = get_params(formula)
        for param in params:
            if param.endswith('_i'):
                formulas_with_i.append(formula)
                break
    # try inferring those inferable and add to record in loop
    while(True):
        infer_record = []  # params inferred in this round
        extend_formulas = []  # formulas after extend
        for formula_with_i in formulas_with_i:
            params = get_params(formula_with_i)
            params_with_i = [param for param in params if param.endswith('_i')]
            i = 1
            # try reform all the formula with _i, search whether there is matching input 
            while(True):
                flag = 0
                reformed_formula_with_i = formula_with_i
                for param_with_i in params_with_i:
                    old = param_with_i
                    new = param_with_i[0:-2] + '_' + str(i)
                    if new in record or new in case.keys():
                        flag = 1  # find one param _num exist in input, replace whole expression _i with _num
                    reformed_formula_with_i = reformed_formula_with_i.replace(old+' ', new+' ')
                    reformed_formula_with_i = reformed_formula_with_i.replace(old+'\n', new+'\n')
                if flag == 0: break  # this means none of the param endswith _i can find a _num one in input, num is to big, break
                extend_formulas.append(reformed_formula_with_i)
                i += 1
        for extend_formula in extend_formulas:
            params = get_params(extend_formula)
            cnt_p = len([p for p in params if p in record or p in case.keys()])  # number of known params
            cnt_all = len(params)  # all params
            if cnt_all - cnt_p == 1:
                lost_param = [p for p in params if p not in record and p not in case.keys()][0]  # infer the lost param
                infer_record.append(lost_param)
        if len(infer_record) == 0: break
        record.extend(infer_record)


# This function aim to deal with the Sum expressions in formulas_raw.txt and replace them with exact _num in input_case.txt, and reform a '+' expression
def handle_sum():
    # for sum expression, left is a normal expression but 
    case = input_case(INPUT_PATH)
    with open(FORMULAS_RAW_PATH, 'r') as f:
        formulas = f.readlines()
    reformed_formulas = []
    for formula in formulas:
        if 'Sum' in formula:  # formula shape: .. = Sum ( .. ), try remove 'Sum' and parents, replace them with simple '+'
            sum_elements = []  # record all the param contain _i
            sum_formulas = []  # record right_exp replaced _i with _num
            left_exp = formula.split('=')[0].strip()  # get left part of the formula, which is the result of right
            right_exp = formula.split('=')[1].replace('Sum', '').strip()  # get right part of the formula, and remove symbol 'Sum'
            elements = right_exp.split(' ')  # get each element in right_exp
            # record all the params contain _i 
            for element in elements:
                if element.endswith('_i'):
                    sum_elements.append(element)
            # initialize loop
            i, flag = 1, 0
            while(True):
                # try replace _i with _num, and check whether it's inputted, if not than stop.
                right_exp_reform = right_exp
                for sum_element in sum_elements:
                    old = sum_element
                    new = sum_element[0:-2] + '_' + str(i)  # replace _i with exact number _num
                    # we can make sure all the params in reformed_formula are in input case, it won't cause key error in freeMOD_infer
                    if new not in case.keys() and new not in record:
                        flag = 1
                        break
                    right_exp_reform = right_exp_reform.replace(old + ' ', new + ' ')  # using space to avoid incorrect replace, for instance a_inter and a_i appear at same time, a_inter might be replaced and get a_1nter
                    right_exp_reform = right_exp_reform.replace(old + '\n', new + '\n')  # special condition: _i appears as the last param
                if flag == 1: break
                sum_formulas.append(right_exp_reform)
                i += 1
            # i still equals 1 means user input nothing, this formula will be deleted, left_exp will choose the default value
            if i == 1:
                print(f'no proper inputs can be found in {formula.strip()}, try skip this formula')
                continue
            right_formula = ' + '.join(sum_formulas)
            reformed_formula = left_exp + ' = ' + right_formula + '\n'
            reformed_formulas.append(reformed_formula)
        else:
            # formula shape: .._i .. = .._i.., both left&right has '_i', replace both with _num input
            extend_formulas = []
            params = get_params(formula)
            params_with_i = [param for param in params if param.endswith('_i')]
            # if no _i contains in this formula, keep the formula still and continue
            if len(params_with_i) == 0:
                reformed_formulas.append(formula)
                continue
            # deal with _i based on record and input_case.txt
            i, flag = 1, 0
            while(True):
                extend_formula = formula
                for param_with_i in params_with_i:
                    old = param_with_i
                    new = param_with_i[0:-2] + '_' + str(i)
                    if new not in case.keys() and new not in record:
                        flag = 1
                        break
                    extend_formula = extend_formula.replace(old + ' ', new + ' ')
                    extend_formula = extend_formula.replace(old + '\n', new + '\n')
                if flag == 1: break
                extend_formulas.append(extend_formula)
                i += 1
            # i still equals 1 means user input nothing, this formula will be deleted
            if i == 1: 
                print(f'no proper inputs can be found in {formula.strip()}, try skip this formula')
                continue
            reformed_formulas.extend(extend_formulas)
    # mostly user will only input right_exp by input _num params, left_exp might choose the value in default case, and then try to solve the conflict and fix left value.
    # remember only input case contain params with _num, default case doesn't. 
    # In this function, only those params in input case can generate a reformed formula, so you don't have to worry some of the _num neither can be found in input case nor can be found in default case
    with open(FORMULAS_PATH, 'w') as f:
        f.writelines(reformed_formulas)



# This function deal with the inequality relations in 'ineqs.txt', base on ineqs.txt fix default_bound.txt into bound.txt
def handle_ineqs():
    # simply fix bound.txt, don't fix input_case.
    default_bound = input_case(DEFAULT_BOUND_PATH)
    pattern = r'(((-?\d+(\.\d+)?)\s*([<>]=?|geq|leq)\s*)?(\w+)(\s*([<>]=?|geq|leq)\s*(-?\d+(\.\d+)?))?)'  # To match inequlity expression
    with open(INEQS_PATH, 'r', encoding='utf-8') as f:
        s = f.read().strip()
    matches = re.findall(pattern, s)
    for match in matches:
        expression, pn, left_value, right_value, left_sign, right_sign = match[0], match[5], float(match[2]), float(match[8]), match[4], match[7]
        # clip the limitation
        # if it's consist of .. > pn >= ..
        if left_sign == '>' and default_bound[pn][1] >= left_value:
            default_bound[pn][1] = left_value - INF_ZERO
        if left_sign == '>=' and default_bound[pn][1] > left_value:
            default_bound[pn][1] = left_value
        if right_sign == '>' and default_bound[pn][0] <= right_value:
            default_bound[pn][0] = right_value + INF_ZERO
        if right_sign == '>=' and default_bound[pn][1] < right_value:
            default_bound[pn][0] = right_value 
        # if it's consist of .. <= pn < ..
        if left_sign == '<' and default_bound[pn][0] <= left_value:
            default_bound[pn][0] = left_value + INF_ZERO
        if left_sign == '<=' and default_bound[pn][0] < left_value:
            default_bound[pn][0] = left_value
        if right_sign == '<' and default_bound[pn][1] >= right_value:
            default_bound[pn][1] = right_value - INF_ZERO
        if right_sign == '<=' and default_bound[pn][1] > right_value:
            default_bound[pn][1] = right_value
        # no way it's consist of .. < pn > .. or .. > pn < .., even if it is, these codes can still handle it in a correct way
    with open(BOUND_PATH, 'w') as f:
        json.dump(default_bound, f, indent=4)


def create_default_bound():
    symb_dict = ['+', '-', '*', ',' , '/', '<', '>', '(', ')', '=', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
    func_dic = ['Min', 'Sum', 'Max', 'ceiling', 'floor']
    # create default_bound.txt, base on formulas.txt
    default_bound = collections.defaultdict(dict)
    with open(FORMULAS_PATH, 'r') as f:
        formulas = f.readlines()
    for formula in formulas:
        elements = formula.strip('\n').split(' ')
        params = set(elements) - set(symb_dict) - set(func_dic)
        # create bound.txt, set the bound as wide as we can
        for param in params:
            if param not in default_bound:
                default_bound[param] = [DEFAULT_LOWER_BOUND, DEFAULT_UPPER_BOUND]
    with open(DEFAULT_BOUND_PATH, 'w') as f:
        json.dump(default_bound, f, indent=4)



def handle_input():
    inputs = collections.defaultdict()
    with open(INPUT_CASE_PATH, 'r') as f:
        input_cases = f.readlines()
        for input_case in input_cases:
            symbol, value = input_case.split('=')[:2]
            inputs[symbol.strip()] = float(value.strip())
    with open(INPUT_PATH, 'w') as f:
        json.dump(inputs, f, indent=4)


def load_database():
    symb_dict = ['+', '-', '*', ',' , '/', '<', '>', '(', ')', '=', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
    func_dic = ['Min', 'Sum', 'Max', 'ceiling', 'floor']
    para_list = collections.defaultdict(dict)
    formu_list = collections.defaultdict(dict)
    i,j = 0, 0
    bound = input_case(BOUND_PATH)

    with open(FORMULAS_PATH,'r') as f:
        formulas = f.readlines()
        for formula in formulas:
            if formula.strip() == '': continue
            formula = formula.strip('\n')
            formu_list[formula]['RID'] = j
            j += 1
            eles = formula.split(' ')
            all_paras = set(eles) - set(symb_dict) - set(func_dic)
            paras = []
            for para in all_paras:
                if re.match(r'^-?\d+(\.\d+)?(e-?\d+)?$', para):
                    continue
                paras.append(para)
            for para in paras:
                if para not in para_list:   # create if appeared first time
                    para_list[para]['PID'] = i
                    i += 1
                    lower = bound[para][0]
                    upper = bound[para][1]
                    para_list[para]['DefaultValue'] = float(np.clip(np.random.normal(loc=(lower+upper)/2, scale=STD_DEV), lower, upper))
                    para_list[para]['Rn'] = []
                    para_list[para]['CountRn'] = 0
                    para_list[para]['Domain'] = [lower, upper]
                para_list[para]['Rn'].append(formu_list[formula]['RID'])
                para_list[para]['CountRn'] += 1     # a param can only appear a single time in a formula
            formu_list[formula]['Pn'] = [para_list[pn]['PID'] for pn in paras]
            formu_list[formula]['CountPn'] = len(formu_list[formula]['Pn'])

    # load into neo4j
    graph = Graph(NEO4J_URL, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    graph.run('MATCH (n) DETACH DELETE n')  # clear database first
    for para, info in zip(para_list.keys(), para_list.values()):
        node = Node("Parameter", PID=info['PID'], ParaName=para, DefaultValue=info['DefaultValue'], Domain=info['Domain'], Rn=info['Rn'], CountRn=info['CountRn'])
        graph.create(node)
    for formu, info in zip(formu_list.keys(), formu_list.values()):
        node = Node("Relation", RID=info['RID'], Formula=formu, Pn=info['Pn'], CountPn=info['CountPn'])
        graph.create(node)
