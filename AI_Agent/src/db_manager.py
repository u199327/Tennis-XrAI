# db_config.py
import os
import sqlite3
import psycopg2
import pandas as pd
import pandera as pa
from dotenv import load_dotenv
from dataclasses import dataclass
import oracledb
from typing import Optional, Any, Dict, Union
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

@dataclass
class DBConfig:
    """Database configuration class to manage Oracle credentials and connection settings"""
    host: str
    service_name: str  # Oracle uses service_name instead of database
    user: str
    password: str
    port: int
    min_connections: int = 1
    max_connections: int = 10
    encoding: str = 'UTF-8'

    @classmethod
    def from_env(cls):
        """Create a DBConfig instance from environment variables"""
        load_dotenv()

        return cls(
            host=os.getenv('DB_HOST', 'localhost'),
            service_name=os.getenv('DB_SERVICE_NAME'),  # Oracle service name
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            port=int(os.getenv('DB_PORT', 1521)),  # Default Oracle port is 1521
            min_connections=int(os.getenv('DB_MIN_CONNECTIONS', 1)),
            max_connections=int(os.getenv('DB_MAX_CONNECTIONS', 10)),
            encoding=os.getenv('DB_ENCODING', 'UTF-8')
        )

    def validate(self):
        """Validate the configuration settings"""
        required_fields = ['service_name', 'user', 'password', 'host', 'port']
        missing_fields = [field for field in required_fields
                         if not getattr(self, field)]

        if missing_fields:
            raise ValueError(
                f"Missing required configuration fields: {', '.join(missing_fields)}"
            )

        if self.min_connections > self.max_connections:
            raise ValueError(
                "min_connections cannot be greater than max_connections"
            )

        return True

    def get_connection_params(self) -> Dict:
        """Return a dictionary of connection parameters"""
        # Oracle TNS connection string
        # dsn = oracledb.makedsn(
        #     host=self.host,
        #     port=self.port,
        #     service_name=self.service_name
        # )
        return {
            'user': self.user,
            'password': self.password,
            'host': self.host,
            'port': self.port,
            'service_name': self.service_name
        }

class BaseDatabaseManager(ABC):
    """Abstract base class for database managers"""

    @abstractmethod
    def _execute_query(self, query: str) -> Any:
        """Execute a database query and return results"""
        pass
    
    def safe_execute_query(self,
                           query: str
                        ) -> Any:
        """Execute a database query and return results with security checks"""
         # List of dangerous SQL keywords that could modify data or schema
        dangerous_keywords = [
            'DROP', 'DELETE', 'TRUNCATE', 'UPDATE', 'INSERT',
            'ALTER', 'CREATE', 'REPLACE', 'MERGE'
        ]

        # Convert query to uppercase for keyword checking
        query_upper = query.upper()

        # Check for dangerous keywords
        for keyword in dangerous_keywords:
            if keyword in query_upper:
                raise f"Error: Query contains forbidden keyword: {keyword}"
        
        return self._execute_query(query)
class DatabaseManager(BaseDatabaseManager):
    """Database manager class for handling Oracle database operations"""

    def __init__(self, config: DBConfig):
        """Initialize DatabaseManager with configuration"""
        self.config = config
        self.config.validate()
        self._connection_pool = None
        self._initialize_connection_pool()

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()

    def _initialize_connection_pool(self):
        """Initialize the Oracle connection pool with configuration settings"""
        try:
            # Initialize oracle client
            oracledb.init_oracle_client()

            # Create the connection pool
            self._connection_pool = oracledb.create_pool(
                min=self.config.min_connections,
                max=self.config.max_connections,
                increment=1,
                **self.config.get_connection_params()
            )
            logging.info("Oracle connection pool created successfully")
        except Exception as e:
            logging.error(f"Error creating Oracle connection pool: {e}")
            raise

    def _execute_query(self,
                     query: str,
                     ) -> Any:
        """Execute a database query and return results"""
        connection = None
        cursor = None
        try:
            connection = self._connection_pool.acquire()
            cursor = connection.cursor()
            cursor.execute(query)
            results = cursor.fetchall()
            return results
        
        except Exception as e:
            if connection:
                connection.rollback()
            logging.error(f"Database error: {e}")
            raise
        finally:
            if cursor:
                cursor.close()
            if connection:
                self._connection_pool.release(connection)


    def close(self):
        """Close all database connections"""
        if self._connection_pool:
            self._connection_pool.close()
            logging.info("All Oracle database connections closed")



class DatabaseManagerSQLite(BaseDatabaseManager):
    def __init__(self, db_path: Union[str, None] = None):
        """
        Initialize the DatabaseManager with a path to the SQLite database.
        :param db_path: Path to the SQLite database file.
        """
        self.db_path = db_path

    def _connect(self):
        """
        Create and return a connection to the SQLite database.
        :return: SQLite database connection object.
        """
        return sqlite3.connect(self.db_path)
    
    def show_tables(self):
        """
        Show the tables in the database.
        """
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        cursor.close()
        conn.close()
        return [table[0] for table in tables]

    def delete_table(self, table_name: str):
        """
        Delete a table from the database.
        :param table_name: Name of the table to delete.
        """
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute(f"DROP TABLE IF EXISTS {table_name};")
        conn.commit()
        cursor.close()
        conn.close()
        print(f"Table '{table_name}' deleted successfully")

    def write_data(self, table_name: str, data: pd.DataFrame):
        """
        Validate and append data into the specified table in the database.
        :param table_name: Name of the table to write data into.
        :param data: DataFrame to be written.
        :param schema: Pandera SchemaModel for validating the data.
        """

        # Connect to the database
        conn = self._connect()

        try:
            # Append the data to the database
            data.to_sql(table_name, conn, if_exists='append', index=False)
            print(f"Data successfully appended to table '{table_name}'")
        except Exception as e:
            print(f"Error appending data to table '{table_name}': {e}")
        finally:
            # Close the connection
            conn.close()

    def read_data(self, table_name: str) -> pd.DataFrame:
        """
        Read data from the specified table in the database.
        :param table_name: Name of the table to read data from.
        :return: DataFrame containing the data from the table.
        """
        # Connect to the database
        conn = self._connect()

        try:
            # Read the data from the database
            query = f"SELECT * FROM {table_name}"
            data = pd.read_sql_query(query, conn)
            print(f"Data successfully read from table '{table_name}'")
            return data
        except Exception as e:
            print(f"Error reading data from table '{table_name}': {e}")
            return pd.DataFrame()  # Return an empty DataFrame on failure
        finally:
            # Close the connection
            conn.close()

    def validate_and_insert(self, table_name: str, data: pd.DataFrame, schema: pa.DataFrameSchema):
        """
        Validate the data and insert it into the database, appending to the existing data.
        :param table_name: Name of the table.
        :param data: DataFrame to validate and insert.
        :param schema: Pandera SchemaModel for validation.
        """
        # Validate the data using the schema
        validated_data = schema.validate(data)
        validated_coerced_data = schema.coerce_dtype(validated_data)
        self.write_data(table_name, validated_coerced_data)
       
    def _execute_query(self, query: str):
        '''
        Execute an SQL query on the database.
        :param query: SQL query to execute.
        :return: DataFrame containing the query result.
        '''
        conn = self._connect()
        cursor = conn.cursor()
        try:
            cursor.execute(query)
            result = cursor.fetchall()
            return result
        except Exception as e:
            result = ['Error executing query: ' + str(e)]
            return result

        finally:
            conn.close()
