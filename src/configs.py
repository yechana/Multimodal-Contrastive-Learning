import pickle

class config_train_1:
    def __init__(self):
        self.split='no_test'
        self.num_workers=8
        self.batch_size=32
        '''
        Key is dataframe name, should match name of attribute in dataset class,
        Which in turn should be IDENTICAL to the actual name of the csv
        Value is list of columns to use
        
        Can use this for reading data like so:
        for module in first set of keys:
            for csv in inner set of keys:
                read data, use module[csv] as usecols argument
        '''
        self.data_to_use = {
            'core': {
                'admissions': ['subject_id','ethnicity','insurance','admission_location',
                              'discharge_location'],
                'patients': ['subject_id','gender','anchor_age'],
                'transfers': ['subject_id','eventtype', 'careunit']
            },
            'ed': {
                'diagnosis': ['subject_id','icd_code','icd_version'],
                'vitalsign': ['subject_id', 'temperature', 'heartrate',
                              'resprate', 'o2sat', 'sbp', 'dbp','rhythm',
                              'pain']                
            },
            'hosp': {
                'diagnoses_icd': ['subject_id','icd_code','icd_version','seq_num'],
                'drgcodes': ['subject_id', 'drg_type', 'drg_code', 'drg_severity',
                             'drg_mortality'],
                'services': ['subject_id', 'curr_service'],
                'microbiologyevents': ['subject_id', 'spec_itemid', 'spec_type_desc', 'ab_itemid',
                                       'test_itemid', 'test_name', 'org_itemid', 'org_name', 
                                       'dilution_value', 'interpretation'],
                'emar': ['subject_id', 'medication'],
                'labevents': ['subject_id', 'flag','priority','itemid'],
                'pharmacy': ['subject_id', 'medication','status'],
                'poe': ['subject_id', 'order_type'],
                'prescriptions': ['subject_id', 'gsn'],
                'procedures_icd': ['subject_id', 'icd_code', 'icd_version']
            }
        }
        self.prepared_data=True
        
class config_train_2:
    def __init__(self):
        self.split='no_test'
        self.num_workers=8
        self.batch_size=32
        '''
        Key is dataframe name, should match name of attribute in dataset class,
        Which in turn should be IDENTICAL to the actual name of the csv
        Value is list of columns to use
        
        Can use this for reading data like so:
        for module in first set of keys:
            for csv in inner set of keys:
                read data, use module[csv] as usecols argument
        '''
        self.data_to_use = {
            'core': {
                'admissions': ['subject_id','ethnicity','insurance','admission_location',
                              'discharge_location'],
                'patients': ['subject_id','gender','anchor_age'],
                'transfers': ['subject_id','eventtype', 'careunit']
            },
            'ed': {
                'diagnosis': ['subject_id','icd_code','icd_version'],
                'vitalsign': ['subject_id', 'temperature', 'heartrate',
                              'resprate', 'o2sat', 'sbp', 'dbp','rhythm',
                              'pain']                
            },
            'hosp': {
                'diagnoses_icd': ['subject_id','icd_code','icd_version','seq_num'],
                'drgcodes': ['subject_id', 'drg_type', 'drg_code', 'drg_severity',
                             'drg_mortality'],
                'services': ['subject_id', 'curr_service'],
                'microbiologyevents': ['subject_id', 'spec_itemid', 'spec_type_desc', 'ab_itemid',
                                       'test_itemid', 'test_name', 'org_itemid', 'org_name', 
                                       'dilution_value', 'interpretation'],
                'emar': ['subject_id', 'medication'],
                'labevents': ['subject_id', 'flag','priority','itemid'],
                'pharmacy': ['subject_id', 'medication','status'],
                'poe': ['subject_id', 'order_type'],
                'prescriptions': ['subject_id', 'gsn'],
                'procedures_icd': ['subject_id', 'icd_code', 'icd_version']
            }
        }
        self.prepared_data=True
        with open('../data/saved/tab_preprocessor_newsplit.pkl', 'rb') as f:
            self.tab_preprocessor = pickle.load(f)
        self.cat_cols = ['admission_location', 'discharge_location', 'insurance', 'ethnicity',
            'gender', 'anchor_age', 'eventtype', 'careunit', 'icd_code',
            'icd_version', 'rhythm', 'pain', 'seq_num',
            'icd_code_diagnoses_icd', 'icd_version_diagnoses_icd', 'drg_type',
            'drg_code', 'drg_severity', 'drg_mortality', 'curr_service',
            'spec_itemid', 'spec_type_desc', 'test_itemid', 'test_name',
            'org_itemid', 'org_name', 'ab_itemid', 'dilution_value',
            'interpretation', 'medication', 'itemid', 'flag', 'priority',
            'medication_pharmacy', 'status', 'order_type', 'gsn','icd_code_procedures_icd',
            'icd_version_procedures_icd']
        self.cont_cols = ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp']
        

class config_mortality:
    def __init__(self):
        self.num_workers=16
        self.batch_size=128
        self.use_all_data=True