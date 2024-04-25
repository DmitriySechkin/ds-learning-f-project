import os
import sqlite3
from abc import ABCMeta, abstractmethod, ABC

import pandas as pd


class DataSourcePdAdapter:
    __metaclass__ = ABCMeta

    def __init__(self, directory_path, file_name):
        self._directory_path = directory_path
        self._file_name = file_name
        self._extension = ''
        self._check_filename()

    def _get_full_path(self):
        path = os.path.join(self._directory_path, f'{self._file_name}.{self._extension}')
        return path

    def _check_directory_path(self):
        if self._directory_path == '':
            return
        if not os.path.exists(self._directory_path):
            os.mkdir(self._directory_path)

        if not os.path.exists(self._directory_path):
            raise Exception(f'Ошибка при создании каталога {self._directory_path}')

    def _check_filename(self):
        if self._file_name == '':
            raise Exception(f'file_name не может быть пустой строкой!')

    @abstractmethod
    def write(self, df):
        pass

    @abstractmethod
    def _read(self):
        pass


class CsvPdAdapter(DataSourcePdAdapter):

    def __init__(self, directory_path, file_name, sep=','):
        super().__init__(directory_path, file_name)
        self._sep = sep
        self._extension = 'csv'

    def write(self, df):
        df.to_csv(self._get_full_path(), index=False, sep=self._sep)

    def read(self, n_rows=None, chunksize=None):
        path = self._get_full_path()
        print(path)
        return pd.read_csv(path, sep=self._sep, chunksize=chunksize, nrows=n_rows, engine='python')


class SqlLitePdAdapter(DataSourcePdAdapter):
    def __init__(self, directory_path, file_name, table_name):
        super().__init__(directory_path, file_name)
        self._table = table_name
        self._operation_type = 'append'
        self._extension = 'db'

        self._create_table()

    def _create_table(self):
        with sqlite3.connect(self._get_full_path()) as con:
            con.execute(f"""
                CREATE TABLE if not exists {self._table} (
                    id INTEGER PRIMARY KEY,
                    name VARCHAR(100),
                    description TEXT,
                    branded_description TEXT,
                    key_skills VARCHAR(200),
                    professional_roles VARCHAR(200)

                );
            """)

    def write(self, df):
        self._check_directory_path()
        with sqlite3.connect(self._get_full_path()) as con:
            df.to_sql(self._table, con, index=False, if_exists="replace")

    def write_row(self, vacancy):
        self._check_directory_path()

        with sqlite3.connect(self._get_full_path()) as connection:
            cursor = connection.cursor()

            try:
                cursor.execute(
                    f"""
                      INSERT INTO {self._table}
                      (
                        id, name, description, branded_description, key_skills, professional_roles
                      )
                      VALUES (?, ?, ?, ?, ?, ?)
                      """, (vacancy.id, vacancy.name, vacancy.description, vacancy.branded_description,
                            vacancy.key_skills, vacancy.professional_roles)
                )
            except Exception as ex:
                print(f'vacancy - {str(vacancy)}')
                print(f'ошибка при записи в БД - {ex}')

                raise ex

    def _get_sql(self, target_columns, query_params_str):
        if target_columns:
            target_columns_srt = ', '.join(target_columns)
        else:
            target_columns_srt = '*'

        return f'select {target_columns_srt} from {self._table} {query_params_str}'

    def read_all(self, chunksize=None):
        query = f'select * from {self._table}'
        with sqlite3.connect(self._get_full_path()) as con:
            con.text_factory = str
            con.text_factory = lambda b: b.decode(errors='ignore')
            return pd.read_sql(query, con, chunksize=chunksize)

    def read_df(self, query_params_str='', target_columns=None):
        query = self._get_sql(target_columns, query_params_str)

        with sqlite3.connect(self._get_full_path()) as con:
            return pd.read_sql(query, con)

    def read_fetchall(self, query_params_str='', target_columns=None):
        query = self._get_sql(target_columns, query_params_str)

        with sqlite3.connect(self._get_full_path()) as con:
            try:
                cursor = con.cursor()
                cursor.execute(query)
                return cursor.fetchall()

            except Exception as ex:
                print(f'ошибка при обращении к БД - {ex}')
