import numpy as np
import pandas as pd
import zipfile
import urllib.request
import urllib3

from openpyxl import load_workbook
import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

GEX_fn = "./GDSC_ALL/GDSC_GEX.npz"
Data_All = np.load(GEX_fn)

cl_feature_arr = Data_All['X']  # (962,11737)
Y = Data_All['Y']  # already normalized -logIC50 (962,265)
drug_lists = Data_All['drug_ids']  # (265,) # number as 1,2,3..265 string in GDSC, like 211	TL-2-105
# comes from Cell_line_RMA_proc_basalExp.txt, including all the genes in essential_gens(1856)
gene_symbols = Data_All['GEX_gene_symbols']
# (962,) # also corresponding to Cell_line_RMA_proc_basalExp.txt and IC50 file, cell_ids with gene_symbols
cell_ids = Data_All['cell_ids']  # like ['1240121', '1240122'],Cell line cosmic identifiers
cell_names = Data_All['cell_names']  # corresponds to IC50 file, like ['BICR22', 'BICR78']
P = cell_ids
Y = pd.DataFrame(Y, index=cell_ids, columns=drug_lists)
X = pd.DataFrame(cl_feature_arr, index=cell_ids, columns=gene_symbols)  # 【962，17737】

toxic_drug_fn = "GDSC_ALL/toxic_drug_ids.csv"
toxic_drugs = pd.read_csv(toxic_drug_fn)
toxic_drugs = toxic_drugs['drug_id']
toxic_drugs = np.array([str(id) for id in toxic_drugs])

drugs_GE0_fn = "/home/liux3941/RL/RL_GDSC/GDSC_ALL/gdsc_drugMedianGE0.txt"
drugs_GE0 = pd.read_csv(drugs_GE0_fn, sep="\t", names=['drug_id'])
drugs_GE0 = np.array([str(id) for id in drugs_GE0['drug_id']])


Y_a = Y - 11.47
# toxic_from_y = Y_a.columns[np.nanmedian(Y, axis=1) < 0]
toxic_from_y = Y.columns[np.nanmax(Y, axis=0) > 15]
