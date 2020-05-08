import os,random

from refactoring_methods import *


def generate_adversarial(k, path, code, file_name):
        final_refactor = ''
        function_list = []

        Class_list, code =  extract_class(code)

        for class_name in Class_list:
            function_list, class_name = extract_function(class_name)

        print(len(function_list))

        for func in function_list:

            refactored_code = func

            for t in range(k):
                refactors_list = [rename_argument,return_optimal,add_argumemts,enhance_for_loop,enhance_filed,enhance_if,rename_api,
                                    rename_local_variable,add_local_variable,rename_method_name]#,add_print]

                refactor       = random.choice(refactors_list)

                try:
                    # print('REFACTOR METHOD IS:', refactor)
                    refactored_code = refactor(refactored_code)

                except Exception as error:
                    refactored_code = refactored_code
                    print('error:\t',error)

            final_refactor = final_refactor + '\n' + refactored_code

            wr_path = path + '/new_' + file_name
            f = open(wr_path,'w')
            f.write(final_refactor)


if __name__ == '__main__':

    K = 1

    mode = 'validation' # Options: train, test
    source = '/Users/Vesal/Desktop/code2seq-master/data/java-small/'

    for path, d, file_names in os.walk(source + mode):
        for filename in file_names:
            if '.java' in filename:
                try:
                    open_file = open(path +'/'+ filename,'r', encoding = 'ISO-8859-1')
                    code = open_file.read()
                    generate_adversarial(K, path, code, filename)
                except Exception as error:
                        print(error)
