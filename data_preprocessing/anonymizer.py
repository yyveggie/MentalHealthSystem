import pandas as pd
import os
from typing import List, Any
from datetime import datetime

class DataAnonymizer:
    def __init__(
        self,
        file_paths: List[str],
        align_key: str,
        anonymize_fields: List[str],
        mask_char: str = '*',
        output_dir: str = './anonymized_data'
    ):
        """
        初始化数据脱敏处理器
        
        Args:
            file_paths: Excel文件路径列表
            align_key: 用于对齐的键(比如'user_id'或'phone')
            anonymize_fields: 需要脱敏的字段列表
            mask_char: 脱敏替换字符
            output_dir: 输出目录
        """
        self.file_paths = file_paths
        self.align_key = align_key
        self.anonymize_fields = anonymize_fields
        self.mask_char = mask_char
        self.output_dir = output_dir
        self.all_columns = set()
        os.makedirs(output_dir, exist_ok=True)
        
    def _mask_value(self, value: Any) -> str:
        """
        对单个值进行脱敏处理
        """
        if pd.isna(value):
            return value
            
        value_str = str(value)
        length = len(value_str)
        
        if length <= 2:
            return self.mask_char * length
        elif length <= 4:
            return value_str[0] + self.mask_char * (length - 1)
        else:
            return value_str[:2] + self.mask_char * (length - 3) + value_str[-1]
            
    def _anonymize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        对数据框进行脱敏处理
        """
        df_anonymized = df.copy()
        
        for field in self.anonymize_fields:
            if field in df_anonymized.columns:
                df_anonymized[field] = df_anonymized[field].apply(self._mask_value)
                
        return df_anonymized

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
        return merged_data

    def save_merged_data(self, merged_df: pd.DataFrame):
        """
        保存合并后的数据
        """
        anonymized_df = self._anonymize_dataframe(merged_df)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"merged_anonymized_{timestamp}.xlsx"
        output_path = os.path.join(self.output_dir, output_file)
        anonymized_df.to_excel(output_path, index=False)
        print(f"\n已保存脱敏文件: {output_path}")
        print("\n数据统计:")
        print(f"总记录数: {len(anonymized_df)}")
        print(f"总字段数: {len(anonymized_df.columns)}")
        non_null_counts = anonymized_df.count()
        print("\n各字段非空值数量:")
        for col in anonymized_df.columns:
            print(f"{col}: {non_null_counts[col]}")

    def run(self):
        """
        运行完整的处理流程
        """
        try:
            print("开始处理文件...")
            merged_df = self.merge_and_process()
            print("开始保存脱敏后的数据...")
            self.save_merged_data(merged_df)
            print("处理完成!")
        except Exception as e:
            print(f"处理过程中发生错误: {str(e)}")
            raise


if __name__ == "__main__":
    file_paths = [
        './database/file/个人史+过敏史+婚育史+家族史.xlsx', 
        './database/file/体格检查+诊疗经过+出院诊断+出院医嘱_副本.xlsx',
        './database/file/主诉+现病史+既往史.xlsx'
        ]
    
    processor = DataAnonymizer(
        file_paths=file_paths,
        align_key='patient_id',  # 用于对齐的键
        anonymize_fields=['patient_id', '住院号', '患者姓名'],  # 需要脱敏的字段
        mask_char='*',  # 脱敏替换字符
        output_dir='./database/anonymized_output'  # 输出目录
    )
    
    processor.run()