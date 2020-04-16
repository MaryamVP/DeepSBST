import os,random,re

from processing_source_code import *

def rename_local_variable(method_string):
    local_var_list = extract_local_variable(method_string)
    if len(local_var_list) == 0:
        return string

    mutation_index = random.randint(0, len(local_var_list) - 1)
    return method_string.replace(local_var_list[mutation_index],word_synonym_replacement(local_var_list[mutation_index]))

def add_local_variable(method_string):
    local_var_list = extract_local_variable(method_string)
    if len(local_var_list) == 0:
        return method_string

    mutation_index = random.randint(0, len(local_var_list) - 1)
    match_ret      = re.search('.+' + local_var_list[mutation_index] + '.+;', method_string)
    if match_ret:
        var_definition      = match_ret.group()
        new_var_definition  = var_definition.replace(local_var_list[mutation_index],word_synonym_replacement(local_var_list[mutation_index]))
        method_string       = method_string.replace(var_definition,var_definition + '\n' + new_var_definition)
        return method_string
    else:
        return method_string

def rename_api(method_string):
    match_ret      = re.findall('\.\s*\w+\s*\(', method_string)
    if match_ret != []:
        api_name = random.choice(match_ret)[1:-1]
        return method_string.replace(api_name,word_synonym_replacement(api_name))
    else:
        return method_string

def rename_method_name(method_string):
    method_name = extract_method_name(method_string)
    if method_name:
        return method_string.replace(method_name, word_synonym_replacement(method_name))
    else:
        return method_string

def rename_argument(method_string):
    arguments = extract_argument(method_string)
    arguments = arguments.replace('final','')
    arguments = arguments.replace('static', '')
    if len(arguments) == 0:
        return method_string

    return method_string.replace(arguments,word_synonym_replacement(arguments))

def return_optimal(method_string):
    if 'return ' in method_string:
        return_statement  = method_string[method_string.find('return ') : method_string.find(';', method_string.find('return ') + 1)]
        return_object     = return_statement.replace('return ','')
        optimal_statement = 'if (' + return_object + ' == null){\n\t\t\treturn 0;\n\t\t}\n' + return_statement
        method_string = method_string.replace(return_statement,optimal_statement)
    return method_string

'''
def enhance_for_loop(method_string):
    for_loop_list = extract_for_loop(method_string)
    if for_loop_list == []:
        return string
    mutation_index = random.randint(0, len(for_loop_list) - 1)
    for_text = for_loop_list[mutation_index]
    for_info = for_text[for_text.find('(') + 1 : for_text.find(')')]
    if ':' in for_info:
        loop_bar = for_info.split(':')[-1].strip()
        loop_var = for_info.split(' ')[1].strip()
        if loop_bar == None or loop_bar == '' or loop_var == None or loop_var == '':
            return string
        new_for_info = 'int i = 0; i < ' + loop_bar + '.size(); i ++'
        method_string = method_string.replace(for_info, new_for_info)
        method_string = method_string.replace(loop_var,loop_bar + '.get(i)')
        return method_string

    else:
        return method_string
'''



if __name__ == '__main__':

    method_string = '''
    public static Calendar toCalendar(final Date date) {
    final Calendar c = Calendar.getInstance(); c.setTime(date);
    return c;
    }
    '''

    refactors_list = [rename_argument,rename_local_variable,add_local_variable,rename_api,rename_method_name,return_optimal]
    refactor       = random.choice(refactors_list)

    try :
        print(refactor(method_string))
    except Exception as error:
        print('refactor:\t',refactor)
        print('error:\t',error)
        print('method_string:\n',method_string)