import pandas as pd
from pymongo import MongoClient

class PatientDataProcessor:
    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.client = MongoClient('localhost', 27017)
        self.db = self.client['medical_records']
        self.collection = self.db['patients']
    
    def process_and_save(self):
        for file_path in self.file_paths:
            self._process_single_file(file_path)
    
    def _process_single_file(self, file_path):
        df = pd.read_excel(file_path)
        for _, row in df.iterrows():
            patient_id = row['patient_id']
            existing_patient = self.collection.find_one({'patient_id': patient_id})
            
            if existing_patient:
                updated_patient = self._update_patient(existing_patient, row)
                self.collection.replace_one({'patient_id': patient_id}, updated_patient)
            else:
                self.collection.insert_one(row.to_dict())
    
    def _update_patient(self, existing_patient, new_data):
        for key, value in new_data.items():
            if key != 'patient_id':
                if key in existing_patient:
                    if pd.isna(existing_patient[key]) and not pd.isna(value):
                        existing_patient[key] = value
                else:
                    existing_patient[key] = value
        return existing_patient
    
    def close_connection(self):
        self.client.close()

if __name__ == "__main__":
    file_paths = [
        './样本示例/个人史+过敏史+婚育史+家族史.xlsx', 
        './样本示例/体格检查+诊疗经过+出院诊断+出院医嘱_副本.xlsx',
        './样本示例/主诉+现病史+既往史.xlsx'
        ]
    processor = PatientDataProcessor(file_paths)
    processor.process_and_save()
    processor.close_connection()