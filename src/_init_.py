# SET DEBUG TO TRUE IF YOU WANT TO SEE PRINTED ALL THE STEPS
DEBUG = False

# IMPORTING THE LOCAL LIBRARY
from functions import *


# MODIFY THESE PATHS AND FILE NAMES ACCORDING TO YOUR WORKING ENVIRONMENT
resources_path = "../resources/"
datasets_path = "../datasets/"
results_path = "../results/"

dataset_fnames = ["bicknell.dataset.txt", "logmet_crowd2.txt", "mcelree_cmcl2.txt", "traxler_cmcl2.txt", "dealmeida.dataset.txt", "matsuki.dataset", "mcelree.dataset.txt", "paczynski_dataset2.txt", "traxler.dataset.txt", "warren2.dataset.txt"]
matrix_prefixes = ["thesis_3012_plmi_freq"]

fillers_path = resources_path + "fillers.txt"
datasets_path = [datasets_path + dataset_path for dataset_path in dataset_fnames]
matrix_path = [resources_path + matrix_path for matrix_path in matrix_prefixes]


# VECTOR SYNTACTIC RELATIONS FILTER
relation_dict = {"bicknell.dataset.txt":("obj", "obj"), "traxler.dataset.txt":("obj", "obj"), "warren2.dataset.txt":("obj", "obj"), "dealmeida.dataset.txt":("obj", "obj"), "mcelree.dataset.txt":("obj", "obj"), "mcelree_cmcl2.txt": ('sbj-1', 'obj-1'), "traxler_cmcl2.txt": ('sbj-1', 'obj-1'), "logmet_crowd2.txt": ('sbj-1', 'obj-1'), "paczynski_dataset2.txt": ('sbj-1', 'sbj'), "matsuki.dataset": ('with', 'with')}

# DICTIONARY OF PROTOTYPES
proto_mat = {}


# LOAD TOP FILLERS
fillers = load_fillers(fillers_path)


# OUTPUT FILE
output = open(results_path + "initial_results.txt", "w")
    

# FOR EVERY MATRIX, FOR EVERY DATASET, FOR EVERY k...
for matrix in matrix_path:
    dsm = load_matrix(matrix)

    for dataset_fname in datasets_path:
        dataset = get_dataset(dataset_fname)
        relations = relation_dict[dataset_fname.split("/")[-1]]

        # OUTPUT FILE
        output = open(results_path + dataset_fname.split("/")[-1], "w")

	for filler_number in [20]:
            for target in dataset:
		for candidate in dataset[target]:
                    key1 = target+"-"+relations[0]
                    proto1 = get_proto(target, relations[0], fillers, filler_number, dsm)

                    if proto1 != False:
                        proto_mat[key1] = proto1
                    else:
                        print("Prototype for %s was not created" % key1)

                    key2 = candidate[0]+"-"+relations[1]
                    proto2 = get_proto(candidate[0], relations[1], fillers, filler_number, dsm)

                    if proto2 != False:
                        proto_mat[key2] = proto2
                    else:
                        print("Prototype for %s was not created" % key2)

            lines, gold, vc_sum, vc_prod = calculate_dataset(proto_mat, dataset, relations, dsm)

            results = zip(lines, vc_sum, vc_prod)

            for result in results:
                output.write("\t".join([str(x) for x in result]) + "\n")





        
