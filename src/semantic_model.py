from typing import List
from abc import ABC, abstractmethod
from typing import List, Dict
from enum import Enum
from pydantic import BaseModel
from typing import Optional

class Column(BaseModel):
    name: str
    data_type: str
    constraints: List[str]
    description: str
    example_values: Optional[List[str]] = None


    def __str__(self):
        constraints_str = ", ".join(self.constraints) if self.constraints else "None"
        return f"Column(name='{self.name}', type='{self.data_type}', constraints=[{constraints_str}], desc='{self.description}')"
    
    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if not isinstance(other, Column):
            return False
        return self.name == other.name and self.data_type == other.data_type and self.constraints == other.constraints

    # Needed so that we can use Column objects as keys in dictionaries
    def __hash__(self):
        return hash((self.name, self.data_type, tuple(self.constraints)))

class Constraint(Enum):
    PRIMARY_KEY = "PRIMARY KEY"
    FOREIGN_KEY = "FORREIGN KEY"
    UNIQUE = "UNIQUE"
    NOT_NULL = "NOT NULL"


class RelationshipType(Enum):
    ONE_TO_ONE = "1:1"
    ONE_TO_MANY = "1:N"
    MANY_TO_MANY = "N:N"
    MANY_TO_ONE = "N:1"

class Relationship(BaseModel):
    type: RelationshipType
    from_table: str
    to_table: str
    via: Optional[str] = None


class Table():
    
    def __init__(self, name: str, description: str, columns: List[Column], relationships: List[Relationship]=[]):
        self.name = name
        self.primary_key = [column for column in columns if Constraint.PRIMARY_KEY.value in column.constraints]
        self.foreign_keys = [column for column in columns if Constraint.FOREIGN_KEY.value in column.constraints]
        self.description = description
        self.columns = columns
        self.relationships = relationships

    def __repr__(self):
        return f"Table(name='{self.name}', description={self.description})"
    
    def _get_table_columns(self) -> Dict[str, Column]:
        return {column.name: column for column in self.columns}
    
    def _get_columns(self, column_names: List[str]) -> List[Column]:
        columns = self._get_table_columns()
        return [columns[name] for name in column_names]
    
    def get_table_short_description(self) -> str:
        """
        Returns an overview short description of the table wihtout columns
        """
        return f"""Table: {self.name}
Description: {self.description}
Primary Key: {', '.join([col.name for col in self.primary_key])}
Relationships: {', '.join([rel.from_table + '-' + rel.type.value + '->' + rel.to_table for rel in self.relationships])}"""
    
    def get_table_context_for_llm(self):
        """
        Returns the table context to add to a prompt for an LLM
        """
        column_lines = "\n".join(
            f"{col.name} {col.data_type}: {col.description}"
            + (f"; Example Values: {col.example_values}" if col.example_values else "")
            for col in self.columns
        )

        return f"""Table: {self.name}
    Description: {self.description}
    Primary Key: {', '.join(col.name for col in self.primary_key)}
    Relationships: {', '.join(rel.from_table + '-' + rel.type.value + '->' + rel.to_table for rel in self.relationships)}
    Columns:
    {column_lines}
    """


class TableProvider:
    def __init__(self, tables_dict: Dict[str, Table]):
        """
        Initialize the TableProvider with a dictionary of pre-initialized tables

        Args:
            tables_dict: Dictionary mapping table names to Table objects
        """
        self._tables = {name.upper(): table for name, table in tables_dict.items()}
    
    def __repr__(self):
        return f"TableProvider({list(self._tables.keys())})"

    def get_tables(self, table_names: List[str]) -> List[Table]:
        """
        Get multiple tables by their names

        Args:
            table_names: List of table names to retrieve

        Returns:
            Dictionary of requested tables

        Raises:
            ValueError: If any requested table name doesn't exist
        """
        normalized_names = [name.upper() for name in table_names]
        invalid_tables = [name for name in normalized_names if name not in self._tables]

        if invalid_tables:
            valid_tables = list(self._tables.keys())
            raise ValueError(
                f"Invalid table names: {invalid_tables}. "
                f"Valid table names are: {valid_tables}"
            )

        return [self._tables[name] for name in normalized_names]

    def get_table(self, table_name: str) -> Table:
        """
        Get a single table by name

        Args:
            table_name: Name of the table to retrieve

        Returns:
            Requested Table object

        Raises:
            ValueError: If the table name doesn't exist
        """
        tables = self.get_tables([table_name])
        return tables[0]

    @property
    def available_tables(self) -> List[str]:
        """
        Get list of all available table names

        Returns:
            List of table names
        """
        return list(self._tables.keys())

    def db_info(self):
        """
        Returns a list of tables with a short description and PK columns and their relationships
        """

        all_table_names = self.available_tables
        all_tables = self.get_tables(all_table_names)
        table_info = "\n\n".join([t.get_table_short_description() for t in all_tables])

        return table_info
    
    def tables_info(self, table_names: List[str]):
        """
        Returns detailed information about the tables and its columns
        """
        tables = self.get_tables(table_names)
        return "\n\n".join([table.get_table_context_for_llm() for table in tables])
    

