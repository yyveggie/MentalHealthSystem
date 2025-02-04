import pandas as pd
import os
from typing import List, Any, Optional, Dict
from datetime import datetime

class DataAnonymizer:
    def __init__(
        self,
        file_paths: List[str],
        align_key: str,
        anonymize_fields: List[str],
        mask_char: str = '*',
        output_dir: str = './output'
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
        
        # 确保输出目录存在
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
        
        # 确保所有要检查的特征都存在于数据框中
        existing_features = [f for f in features_to_check if f in df.columns]
        
        if not existing_features:
            print("警告：未找到任何指定的特征列")
            return df, 0
            
        initial_count = len(df)
        
        # 检查这些特征是否全部为空
        mask = df[existing_features].notna().any(axis=1)
        df_filtered = df[mask].copy()
        
        removed_count = initial_count - len(df_filtered)
        if removed_count > 0:
            print(f"删除了 {removed_count} 条所有指定特征均为空的记录")
            
        return df_filtered, removed_count

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

    def save_to_excel(self, merged_df: pd.DataFrame, metadata: Optional[Dict[str, Any]] = None):
        """
        将原始数据和脱敏数据保存到Excel文件
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 检查并移除原始数据中的重复记录
            raw_data_no_duplicates, raw_removed_count = self._check_and_remove_duplicates(merged_df)
            if raw_removed_count > 0:
                print(f"从原始数据中移除了 {raw_removed_count} 条重复记录")
            
            # 删除特定特征均为空的记录
            raw_data_filtered, empty_removed_count = self._remove_empty_records(raw_data_no_duplicates)
            
            # 保存去重和过滤后的原始数据
            raw_output_path = os.path.join(self.output_dir, f'raw_data_{timestamp}.xlsx')
            raw_data_filtered.to_excel(raw_output_path, index=False)
            print(f"原始数据已保存到: {raw_output_path}")
            
            # 对过滤后的数据进行脱敏
            anonymized_df = self._anonymize_dataframe(raw_data_filtered)
            
            # 保存脱敏数据
            anonymized_output_path = os.path.join(self.output_dir, f'anonymized_data_{timestamp}.xlsx')
            anonymized_df.to_excel(anonymized_output_path, index=False)
            print(f"脱敏数据已保存到: {anonymized_output_path}")
            
            # 更新元数据以包含空记录删除信息
            if metadata is None:
                metadata = {}
            metadata.update({
                '删除重复记录数': raw_removed_count,
                '删除空特征记录数': empty_removed_count,
                '最终记录数': len(raw_data_filtered)
            })
            
            # 保存处理元数据
            metadata_df = pd.DataFrame([{
                '处理时间': timestamp,
                '源文件数量': len(self.file_paths),
                '源文件列表': ', '.join(self.file_paths),
                '对齐键': self.align_key,
                '脱敏字段': ', '.join(self.anonymize_fields),
                **metadata
            }])
            metadata_output_path = os.path.join(self.output_dir, f'processing_metadata_{timestamp}.xlsx')
            metadata_df.to_excel(metadata_output_path, index=False)
            print(f"处理元数据已保存到: {metadata_output_path}")
            
            print(f"数据已成功保存，最终处理了 {len(raw_data_filtered)} 条记录")
            
        except Exception as e:
            print(f"保存数据时发生错误: {str(e)}")
            raise

    def run(self, metadata: Optional[Dict[str, Any]] = None):
        """
        运行完整的处理流程
        
        Args:
            metadata: 额外的元数据信息
        """
        try:
            print("开始处理文件...")
            merged_df = self.merge_and_process()
            
            print("开始保存数据到Excel...")
            self.save_to_excel(
                merged_df=merged_df,
                metadata=metadata
            )
            
            print("处理完成!")
        except Exception as e:
            print(f"处理过程中发生错误: {str(e)}")
            raise

if __name__ == "__main__":
    file_paths = [
        '/Users/mushroom/Library/CloudStorage/OneDrive-共享的库-onedrive/Database/TongJi_Data/个人史-过敏史-婚育史-家族史.xlsx', 
        '/Users/mushroom/Library/CloudStorage/OneDrive-共享的库-onedrive/Database/TongJi_Data/体格检查-诊疗经过-出院诊断-出院医嘱.xlsx',
        '/Users/mushroom/Library/CloudStorage/OneDrive-共享的库-onedrive/Database/TongJi_Data/主诉+现病史+既往史.xlsx'
    ]
    
    # 元数据信息
    metadata = {
        '项目名称': '病历数据处理',
        '版本': '1.0',
        '处理日期': datetime.now().strftime('%Y-%m-%d')
    }
    
    processor = DataAnonymizer(
        file_paths=file_paths,
        align_key='patient_id',
        anonymize_fields=['patient_id', '住院号', '患者姓名'],
        mask_char='*',
        output_dir='./data/processed_data'
    )
    
    processor.run(metadata=metadata)