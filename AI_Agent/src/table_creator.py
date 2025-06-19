import os
import sys
import pandas as pd
import json
from src.semantic_model import Table, Column,TableProvider

def csv_to_table(csv_file_path: str, table_description: str) -> Table:
    df = pd.read_csv(csv_file_path, encoding='utf-8', quotechar='"', skipinitialspace=True)
    table_name = os.path.splitext(os.path.basename(csv_file_path))[0]
    columns = []
    for _, row in df.iterrows():
        column = Column(
            name=row['column_name'],
            data_type = row['data_format'],
            constraints=[],
            description = (
                "" if pd.isna(row["column_description"]) else row["column_description"]
            ) + (
                "" if pd.isna(row["value_description"]) else " Value description: " + row["value_description"]
            )
        )
        columns.append(column)
    
    table = Table(
        name=table_name,
        description=table_description,
        columns=columns
    )

    return table
    
def load_table_data_from_csv(csv_file_path, table_description):
    return csv_to_table(csv_file_path, table_description)

def create_table_provider_from_folder(folder_path, include_descriptions = False):
    tables_dict = {}
    
    table_descriptions = {}
    if include_descriptions:
        descriptions_file = os.path.join(folder_path, 'tables_descriptions.json')
        if os.path.exists(descriptions_file):
            with open(descriptions_file, 'r', encoding='utf-8') as f:
                table_descriptions = json.load(f)

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            table_name = os.path.splitext(file_name)[0]
            description = table_name
            if include_descriptions:
                description = table_descriptions.get(table_name, table_name)
            
            file_path = os.path.join(folder_path, file_name)
            tables_dict[table_name] = load_table_data_from_csv(file_path, description)

    return TableProvider(tables_dict=tables_dict)