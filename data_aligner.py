import pandas as pd
import os
from typing import List
from datetime import datetime
from typing import Optional, Dict, Any

class DataAligner:
    def __init__(self, file_paths: List[str], align_key: str, output_dir: str = './output'):
        """
        初始化数据对齐处理器
        
        Args:
            file_paths: Excel文件路径列表
            align_key: 用于对齐的键(比如'user_id'或'phone')
            output_dir: 输出目录
        """
        self.file_paths = file_paths
        self.align_key = align_key
        self.output_dir = output_dir
        self.all_columns = set()

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

    def merge_and_process(self) -> pd.DataFrame:
        """
        合并所有表格并进行数据对齐
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
                
                # 处理重复列
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
        将对齐后的数据保存到Excel文件
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            # 保存合并后的数据
            output_path = os.path.join(self.output_dir, f'aligned_data_{timestamp}.xlsx')
            merged_df.to_excel(output_path, index=False)
            print(f"对齐数据已保存到: {output_path}")
            
            # 更新元数据
            if metadata is None:
                metadata = {}
            metadata.update({
                '最终记录数': len(merged_df)
            })
            
            # 保存处理元数据
            metadata_df = pd.DataFrame([{
                '处理时间': timestamp,
                '对齐键': self.align_key,
                **metadata
            }])
            metadata_output_path = os.path.join(self.output_dir, f'processing_metadata_{timestamp}.xlsx')
            metadata_df.to_excel(metadata_output_path, index=False)
            print(f"处理元数据已保存到: {metadata_output_path}")
            
            print(f"数据已成功保存，最终处理了 {len(merged_df)} 条记录")
            
        except Exception as e:
            print(f"保存数据时发生错误: {str(e)}")
            raise

class DataProcessor:
    def __init__(
        self,
        file_paths: List[str],
        align_key: str,
        output_dir: str = './output'
    ):
        self.aligner = DataAligner(file_paths, align_key, output_dir)

    def run(self, metadata: Optional[Dict[str, Any]] = None):
        """
        运行完整的处理流程
        """
        try:
            print("开始处理文件...")
            merged_df = self.aligner.merge_and_process()
            
            print("开始保存数据到Excel...")
            self.aligner.save_to_excel(merged_df=merged_df, metadata=metadata)
            
            print("处理完成!")
        except Exception as e:
            print(f"处理过程中发生错误: {str(e)}")
            raise

if __name__ == "__main__":
    file_paths = [
        '/path/to/your/excel1.xlsx', 
        '/path/to/your/excel2.xlsx',
        '/path/to/your/excel3.xlsx'
    ]
    
    # 元数据信息
    metadata = {
        '项目名称': '数据对齐项目',
        '版本': '1.0',
        '处理日期': datetime.now().strftime('%Y-%m-%d')
    }
    
    processor = DataProcessor(
        file_paths=file_paths,
        align_key='patient_id',  # 对齐的键
        output_dir='./data/processed_data'
    )
    
    processor.run(metadata=metadata)