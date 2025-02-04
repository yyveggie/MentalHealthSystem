import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import pandas as pd
import os
from typing import List, Any, Optional, Dict
from datetime import datetime
import hashlib

from database.mongo_handler import MongoDBStorage


class DataAnonymizer:
    def __init__(
        self,
        file_paths: List[str],
        align_key: str,
        anonymize_fields: List[str],
        mask_char: str = '*',
    ):
        """
        初始化数据脱敏处理器
        
        Args:
            file_paths: Excel文件路径列表
            align_key: 用于对齐的键(比如'user_id'或'phone')
            anonymize_fields: 需要脱敏的字段列表
            mask_char: 已废弃参数，保留以兼容旧代码
        """
        self.file_paths = file_paths
        self.align_key = align_key
        self.anonymize_fields = anonymize_fields
        self.all_columns = set()
        
    def _mask_value(self, value: Any) -> str:
        """
        使用MD5对值进行加密
        """
        if pd.isna(value):
            return value
            
        value_str = str(value)
        # 使用MD5进行加密
        return hashlib.md5(value_str.encode('utf-8')).hexdigest()
            
    def _anonymize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        对数据框进行脱敏处理
        """
        df_anonymized = df.copy()
        
        for field in self.anonymize_fields:
            if field in df_anonymized.columns:
                df_anonymized[field] = df_anonymized[field].apply(self._mask_value)
                
        return df_anonymized
    
    def _check_and_remove_duplicates(self, df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """
        检查并移除重复记录
        
        Args:
            df: 输入数据框
            
        Returns:
            tuple: (去重后的数据框, 删除的重复记录数)
        """
        initial_count = len(df)
        df_no_duplicates = df.drop_duplicates(subset=[self.align_key], keep='first')
        removed_count = initial_count - len(df_no_duplicates)
        
        return df_no_duplicates, removed_count

    def _remove_empty_records(self, df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """
        删除指定特征均为空值的样本
        
        Args:
            df: 输入数据框
            
        Returns:
            tuple: (清理后的数据框, 删除的记录数)
        """
        features_to_check = [
            '个人史', '过敏史', '婚育史', '家族史', 
            '体格检查', '诊疗经过', '出院诊断', '出院医嘱',
            '主诉', '现病史', '既往史'
        ]
        
        # 确保所有需要检查的特征都存在于数据框中
        existing_features = [f for f in features_to_check if f in df.columns]
        
        if not existing_features:
            print("警告：未找到任何指定的特征列")
            return df, 0
            
        # 计算这些特征是否全部为空
        all_empty = df[existing_features].isna().all(axis=1)
        
        # 保留至少有一个特征非空的记录
        df_cleaned = df[~all_empty]
        removed_count = len(df) - len(df_cleaned)
        
        print(f"删除了 {removed_count} 条所有指定特征均为空的记录")
        return df_cleaned, removed_count

    def merge_and_process(self) -> pd.DataFrame:
        """
        合并所有表格并处理数据
        """
        merged_data = None
        
        print("正在分析表格结构...")
        for file_path in self.file_paths:
            df = pd.read_excel(file_path)
            if self.align_key not in df.columns:
                raise ValueError(f"对齐键 '{self.align_key}' 在文件 {file_path} 中未找到")
            self.all_columns.update(df.columns)
            print(f"从 {os.path.basename(file_path)} 中发现 {len(df.columns)} 个字段")

        print("\n开始合并表格...")
        for file_path in self.file_paths:
            df = pd.read_excel(file_path)
            print(f"处理文件: {os.path.basename(file_path)}")
            
            if merged_data is None:
                merged_data = df
            else:
                merged_data = pd.merge(
                    merged_data, 
                    df,
                    on=self.align_key,
                    how='outer',
                    suffixes=('', '_duplicate')
                )
                
                duplicate_cols = [col for col in merged_data.columns if col.endswith('_duplicate')]
                for dup_col in duplicate_cols:
                    original_col = dup_col[:-10]
                    merged_data[original_col] = merged_data[original_col].fillna(merged_data[dup_col])
                    merged_data = merged_data.drop(columns=[dup_col])
        for col in self.all_columns:
            if col not in merged_data.columns:
                merged_data[col] = None

        print(f"\n合并完成，共处理 {len(merged_data)} 条记录")
        
        # 添加清理空记录的步骤
        merged_data, removed_count = self._remove_empty_records(merged_data)
        print(f"清理后剩余 {len(merged_data)} 条记录")
        
        return merged_data

    def save_to_mongodb(self, 
                        merged_df: pd.DataFrame,
                        mongodb_storage: MongoDBStorage,
                        database: str,
                        metadata: Optional[Dict[str, Any]] = None):
        """
        将原始数据和脱敏数据保存到MongoDB
        
        Args:
            merged_df: 合并后的数据框
            mongodb_storage: MongoDB存储实例
            database: 数据库名称
            metadata: 额外的元数据信息
        """
        try:
            # 检查并移除原始数据中的重复记录
            raw_data_no_duplicates, raw_removed_count = self._check_and_remove_duplicates(merged_df)
            if raw_removed_count > 0:
                print(f"从原始数据中移除了 {raw_removed_count} 条重复记录")
            
            # 更新元数据以包含空记录删除信息
            metadata = metadata or {}
            metadata.update({
                'source_files': self.file_paths,
                'align_key': self.align_key,
                'removed_duplicates': raw_removed_count,
            })
            
            # 保存去重后的原始数据
            mongodb_storage.save_data(
                database=database,
                collection='raw_data',
                data=raw_data_no_duplicates,
                is_anonymized=False,
                metadata={
                    'source_files': self.file_paths,
                    'align_key': self.align_key,
                    'removed_duplicates': raw_removed_count,
                    **(metadata or {})
                }
            )
            
            # 对去重后的数据进行脱敏
            anonymized_df = self._anonymize_dataframe(raw_data_no_duplicates)
            
            # 保存脱敏数据
            mongodb_storage.save_data(
                database=database,
                collection='anonymized_data',
                data=anonymized_df,
                is_anonymized=True,
                metadata={
                    'source_files': self.file_paths,
                    'align_key': self.align_key,
                    'anonymized_fields': self.anonymize_fields,
                    'removed_duplicates': raw_removed_count,
                    **(metadata or {})
                }
            )
            
            print(f"数据已成功保存到MongoDB，最终保存了 {len(raw_data_no_duplicates)} 条去重记录")
            
        except Exception as e:
            print(f"保存到MongoDB时发生错误: {str(e)}")
            raise

    def run(self, mongodb_config: Optional[Dict[str, Any]] = None):
        """
        运行完整的处理流程
        
        Args:
            mongodb_config: MongoDB配置信息，包含host、port、database等
        """
        try:
            print("开始处理文件...")
            merged_df = self.merge_and_process()
            
            if mongodb_config:
                print("开始保存数据到MongoDB...")
                mongodb_storage = MongoDBStorage(
                    host=mongodb_config.get('host', 'localhost'),
                    port=mongodb_config.get('port', 27017)
                )
                try:
                    self.save_to_mongodb(
                        merged_df=merged_df,
                        mongodb_storage=mongodb_storage,
                        database=mongodb_config['database'],
                        metadata=mongodb_config.get('metadata')
                    )
                finally:
                    mongodb_storage.close()
            
            print("处理完成!")
        except Exception as e:
            print(f"处理过程中发生错误: {str(e)}")
            raise


if __name__ == "__main__":
    file_paths = [
        './data/file/个人史+过敏史+婚育史+家族史.xlsx', 
        './data/file/体格检查+诊疗经过+出院诊断+出院医嘱.xlsx',
        './data/file/主诉+现病史+既往史.xlsx'
    ]
    
    # MongoDB配置
    mongodb_config = {
        'host': 'localhost',
        'port': 27017,
        'database': 'medical_records',
        'metadata': {
            'project': '病历数据处理',
            'version': '1.0',
            'processing_date': datetime.now().strftime('%Y-%m-%d')
        }
    }
    
    processor = DataAnonymizer(
        file_paths=file_paths,
        align_key='patient_id',
        anonymize_fields=['patient_id', '住院号', '患者姓名'],
        mask_char='*',
    )
    
    processor.run(mongodb_config=mongodb_config)