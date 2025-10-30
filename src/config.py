data_path = r'C:\Users\RiL\Desktop\Prog\scoring_ifrs9\data\raw\data_final.csv'
model_path = "models/model_pd_v1.pkl"

y = 'loan_status'
Test_size = 0.2
random_state = 44
k_neigh = 10

var_to_imput = ['open_acc_6m', 'total_bal_il', 'inq_fi']


aux_var_num = ['dti', 'tot_cur_bal', 'int_rate', 'annual_inc', 'mort_acc']
aux_var_ord = ['grade']
grade = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
aux_var_nom = ['home_ownership']

# 'var' pour l'imputation déterministe (toutes les colonnes)
features = [
    "term", "int_rate", "installment", "grade", "emp_length", "home_ownership", "annual_inc", "loan_status", "dti", "tot_cur_bal", "open_acc_6m",
    "total_bal_il", "inq_fi", "mort_acc", "num_sats"
]

num_features = ["int_rate", "installment", "annual_inc","dti", "tot_cur_bal","open_acc_6m",
                "total_bal_il", "inq_fi", "mort_acc", "num_sats"]
cat_features = ["term", "grade", "emp_length", "home_ownership"]

cat_order = [
    [' 36 months', ' 60 months'],
    ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
    ['< 1 year', '1 year', '2 years', '3 years', '4 years', '5 years',
    '6 years', '7 years', '8 years', '9 years', '10+ years'],
    ['OWN', 'MORTGAGE', 'RENT', 'ANY', 'OTHER', 'NONE']
]

train_path = "data/processed/train_imp.parquet"
test_path = "data/processed/test_imp.parquet"
processed_path = "data/processed"