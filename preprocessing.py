import os
import pandas as pd

# Define paths
raw_filepath = 'loan.csv'
processed_dir = 'processed_loan_data'
processed_combined_filepath = os.path.join(processed_dir, 'combined_loan.csv')

def preprocess_and_save_data(filepath, processed_dir):
    # Load and preprocess the dataset
    df = pd.read_csv(filepath, low_memory=False)
    # Copy Dataframe
    data = df.copy()
    data = data.drop(['id', 'member_id', 'emp_title', 'issue_d', 'zip_code', 'emp_length', 'title', 'earliest_cr_line', 'last_pymnt_d', 'hardship_start_date', 'desc', 'hardship_end_date', 'payment_plan_start_date', 'next_pymnt_d', 'settlement_date', 'last_credit_pull_d', 'debt_settlement_flag_date', 'sec_app_earliest_cr_line'], axis=1)
    data = data.drop(['url', 'mths_since_last_delinq', 'mths_since_last_major_derog', 'mths_since_last_record', 'annual_inc_joint', 'dti_joint', 'verification_status_joint', 'mths_since_recent_bc_dlq', 'mths_since_recent_revol_delinq', 'revol_bal_joint', 'sec_app_inq_last_6mths', 'sec_app_mort_acc', 'sec_app_open_acc', 'sec_app_revol_util', 'sec_app_open_act_il',
                      'sec_app_num_rev_accts', 'sec_app_chargeoff_within_12_mths', 'sec_app_collections_12_mths_ex_med', 'sec_app_mths_since_last_major_derog', 'hardship_type', 'hardship_reason', 'hardship_status', 'deferral_term', 'hardship_amount', 'hardship_length', 'hardship_dpd', 'hardship_loan_status', 'orig_projected_additional_accrued_interest', 'hardship_payoff_balance_amount',
                      'hardship_last_payment_amount', 'settlement_status', 'settlement_amount', 'settlement_percentage', 'settlement_term'], axis=1)
    data = data.fillna(0)

    columns = data.columns
    df = data.copy()
    list_obj = []
    list_1 = []
    list_10 = []
    list_100 = []
    list_10000 = []
    for i in range(len(columns)):
        if (df.loc[:, columns[i]].dtype == 'object') and (columns[i] != 'addr_state'):
            list_obj.append(columns[i])
            value = list(df.drop_duplicates(columns[i]).loc[:, columns[i]])
            for j in range(len(value)):
                df.loc[df[columns[i]] == value[j], columns[i]] = j
        elif (df.loc[:, columns[i]].dtype == 'float64') or (df.loc[:, columns[i]].dtype == 'int64'):
            if (df[columns[i]].mean() > 10.0) and (df[columns[i]].mean() <= 100.0):
                list_10.append(columns[i])
                df[columns[i]] = df[columns[i]] / 10
            elif (df[columns[i]].mean() > 100.0) and (df[columns[i]].mean() <= 1000.0):
                list_100.append(columns[i])
                df[columns[i]] = df[columns[i]] / 100
            elif (df[columns[i]].mean() > 1000.0):
                list_10000.append(columns[i])
                df[columns[i]] = df[columns[i]] / 10000
            elif (df[columns[i]].mean() <= 10.0):
                list_1.append(columns[i])

    state_set = list(set(df.loc[:, 'addr_state']))

    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    for j in range(len(state_set)):
        data_new = df.loc[df['addr_state'] == state_set[j]].drop(['addr_state'], axis=1)
        with open(os.path.join(processed_dir, f'loan_{state_set[j]}.csv'), 'w', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(data_new.columns)
            for i in range(data_new.shape[0]):
                csv_writer.writerow(data_new.iloc[[i][0]])

    # Combine all preprocessed data into a single DataFrame and save
    combined_df = pd.concat([pd.read_csv(os.path.join(processed_dir, f'loan_{state}.csv')) for state in state_set])
    combined_df.to_csv(processed_combined_filepath, index=False)

    print('Preprocessing done and data saved.')
    return combined_df

# Preprocess data
if os.path.exists(processed_combined_filepath):
    print('Loading preprocessed data...')
    combined_df = pd.read_csv(processed_combined_filepath)
else:
    print('Preprocessing data...')
    combined_df = preprocess_and_save_data(raw_filepath, processed_dir)

