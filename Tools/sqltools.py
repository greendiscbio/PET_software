import pymysql
import re
import pandas

class MySQLManager:
    """
    Class that allows manage MySQL database basic operations such as load data from a pandas dataframe.
    #Object handling methods
        close_connection(): closes the database connection.
    #Database handling methods
        show_databases(): shows the available databases.
        create_database(): creates a new database.
        use_database(): uses a database.
        current_database() shows the current database.
        drop_database(): drops a database.
    #Tables handling methods
        show_tables(): shows the available tables.
        create_table(): creates a new table from pandas dataframe.
        insert_into_table(): inserts data manually.
        get_table(): displays a table.
        get_table_description(): displays a table description.
        drop_table(): drops a table.
    #Query handling methods
        execute(): performs a SQL query against the selected database.
    """

    #################################################################
    #PRE-REQUISITES
    #################################################################

    # Data types definition

    INTEGER_TYPE = 'INT DEFAULT NULL'
    FLOAT_TYPE = 'FLOAT(10,2) DEFAULT NULL'
    DATE_TYPE = 'DATE'
    STRING_TYPE = 'CHAR(10)'
    PRIMARY_KEY = "INT NOT NULL PRIMARY KEY"

    #################################################################
    #PUBLIC METHODS
    #################################################################

    #Object handling methods

    def __init__(self, host: str, user: str, password: str):
        """
        Function that initializes a connection.
        :param host: (String) Host ('localhost' or ip).
        :param user: (String) Use name.
        :param password: (String) Database password.
        """
        try:
            self.host = host
            self.user = user
            self.connection = pymysql.connect(host=host,
                                              user=user,
                                              password=password,
                                              charset='utf8mb4',
                                              cursorclass=pymysql.cursors.DictCursor)
        except:
            raise TypeError("Incorrect data, make sure that the database exists"
                            "and the username and password are correct")

    def __repr__(self):
        return "<MySQLManager database=%s host=%s user=%s>" % (self.current_database, self.host, self.user)

    def __str__(self):
        return "<MySQLManager database=%s host=%s user=%s>" % (self.current_database, self.host, self.user)

    def close_connection(self):
        """
        Function that closes the database connection.
        :return: None.
        """
        self.connection.close()

    #Database handling methods

    def show_databases(self):
        """
        Function that shows the different existing databases.
        :return: None.
        """
        with self.connection.cursor() as cursor:
            cursor.execute("SHOW DATABASES")
            for database in cursor.fetchall():
                print(database['Database'])

    def create_database(self, database: str):
        """
        Function that creates a new database given a name. If the new database already exists, it displays a message.
        :param database: (String) New database name.
        :return: None.
        """
        if not self.__exists(param=database, element="DATABASES"):
            with self.connection.cursor() as cursor:
                # If the database doesn't exist create it
                cursor.execute("CREATE DATABASE %s" % database)
            self.connection.commit()
        else:
            print("Database %s already exists, use drop_database() to delete it" % database)

    def use_database(self, database: str):
        """
        Function that allows switching between databases.
        :param database: (String) Database to be used.
        :return: None.
        """
        if self.__exists(param=database, element="DATABASES"):
            with self.connection.cursor() as cursor:
                cursor.execute("USE %s" % database)
            self.connection.commit()
        else:
            print("Database %s doesn't exists use create_database()"
                  " to create new databases" % database)

    @property
    def current_database(self):
        """
        Function that returns the current database name.
        :return: (String) Database name.
        """
        with self.connection.cursor() as cursor:
            cursor.execute("SELECT DATABASE()")
            return cursor.fetchone()['DATABASE()']

    def drop_database(self, database: str):
        """
        Function that deletes the database.
        :param database: (String) Database name.
        :return: None.
        """
        if self.__exists(param=database, element="DATABASES"):
            with self.connection.cursor() as cursor:
                cursor.execute("DROP DATABASE %s" % database)
            self.connection.commit()
        else:
            print("Database %s doesn't exist" % database)

    #Table handling methods

    def show_tables(self):
        """
        Function that shows the current database tables.
        :return: None.
        """
        with self.connection.cursor() as cursor:
            if self.current_database is not None:
                cursor.execute("SHOW TABLES")
                for table in cursor.fetchall():
                    print(list(table.values())[0])
            else:
                print("To display the tables you must selected a database using use_databse()")

    def create_table(self, df: pandas.DataFrame, primary_key: str, table: str):
        """
        Function that creates a new table from a given pandas dataframe and inserts the dataframe values into.
        :param df: (Pandas DataFrame) Pandas DataFrame with data.
        :param primary_key: (String) Column name within the dataframe that will act as primary key.
        :param table: (String) Table name.
        :return: None.
        """
        try:
            # Add primary key
            table_definition = {primary_key: self.PRIMARY_KEY}
            # Check column type
            for col in df.columns:
                if col != primary_key:
                    # Check data type
                    table_definition[col] = self.__check_type(df[col])
            # Create table
            self.__create_table(table, table_definition)
            print('Table %s created successfully' % table)
            # Load data from pandas dataframe
            self.__load_data_from_pd(df, table, table_definition)
            print('Data correctly inserted in the %s table' % table)
        except:
            print("\nSomething wrong, check the arguments arguments:\n\n\tdf = "
                  "Pandas dataframe\n\tprimary_key = Unique primary key\n\ttable"
                  " = Table name")

    def insert_into_table(self, table: str, values: dict):
        """
        Function that inserts the values received such as dictionary in the indicated table. It is used to insert data into the database manually.
        :param table: (String) Table name.
        :param values: (Dictionary) Dictionary with table columns as keys and params like values.
        :return: None.
        """
        try:
            if self.__exists(param=table, element="TABLES"):
                with self.connection.cursor() as cursor:
                    columns = list(values.keys())
                    row_values = list(values.values())
                    sql = "INSERT INTO %s(%s) VALUES(%s);" \
                          % (table, ', '.join(columns), ', '.join(row_values))
                    cursor.execute(sql)
                self.connection.commit()
            else:
                print("Table %s doesn't exists" % table)
        except:
            print("Impossible insert data, check the format of the table "
                  "using get_table_description or the data format")

    def get_table(self, table: str):
        """
        Function that displays a table given a name.
        :param table: (String) Table name.
        :return: Pandas dataframe.
        """
        if self.__exists(param=table, element="TABLES"):
            sql = "SELECT * FROM %s" % (table)
            df = pandas.DataFrame(self.execute(sql))
            return df
        else:
            print("Table doesn't exists")

    def get_table_description(self, table: str):
        """
        Function that displays a table definition given a name.
        :param table: (String) Table name.
        :return: None.
        """
        if self.__exists(param=table, element="TABLES"):
            with self.connection.cursor() as cursor:
                cursor.execute("DESCRIBE %s" % table)
                for description in cursor.fetchall():
                    values = list(description.values())
                    print("Table: %s\tType: %s\tNull: %s\tKey: %s"
                          "\tDefault: %s\tExtra: %s\n" % tuple(values))
        else:
            print("Table doesn't exists")

    def drop_table(self, table: str):
        """
        Function that deletes a table given a name.
        :param table: (String) Table name.
        :return: None.
        """
        if self.__exists(param=table, element="TABLES"):
            with self.connection.cursor() as cursor:
                cursor.execute("DROP TABLE %s" % table)
            self.connection.commit()
        else:
            print("Table %s doesn't exist" % table)

    #Query handling methods

    def execute(self, sql: str, *, without=None):
        """
        Function that allows to execute MySQL queries.
        :param sql: (String) MySQL query.
        :param without: (String) Column to be excluded (usually id).
        :return: None.
        """
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(sql)
                results = cursor.fetchall()
                # Remove column (usually id) from results
                if without is not None:
                    new_results = []
                    for dictionary in results:
                        del dictionary[without]
                        new_results.append(dictionary)
                    return new_results
                else:
                    return results
        except:
            print("Wrong query")

    #################################################################
    #PRIVATE METHODS
    #################################################################

    def __exists(self, param, element):
        """
        Function that checks if a database or table exists.
        :param param: (String) Database or table name.
        :param element: (String) DATABASES / TABLES.
        :return: (Bool) True if the database / table exists.
        """
        with self.connection.cursor() as cursor:
            cursor.execute("SHOW %s" % element)
            # Check if the database already exists
            for dict_ in cursor.fetchall():
                if list(dict_.values())[0] == param:
                    return True
        return False

    def __create_table(self, table, table_definition):
        """
        Function that creates a new table in the database from a definition given by a dictionary.
        If the table already exists this method will replaced it with the new one.
        :param table: (String) Table name.
        :param table_definition: (Dictionary) Table names as keys and MySQL definition such as value.
        :return: None.
        """
        if self.current_database is not None:
            if self.__exists(param=table, element="TABLES"):
                print("The table %s has been replaced" % table)
            with self.connection.cursor() as cursor:
                cursor.execute("DROP TABLE IF EXISTS %s" % table)
                sql = self.__build_create_table(table, table_definition)
                cursor.execute(sql)
            self.connection.commit()
        else:
            raise Exception("Before creating tables you need to select a database")

    def __build_create_table(self, table, table_definition):
        """
        Function that builds the SQL statement to create a new table.
        :param table: (String) Table name.
        :param table_definition: (Dictionary) Dictionary with keys as table fields.
        and field definitions such as values.
        :return: (String) MySQL query.
        """
        sql = "CREATE TABLE IF NOT EXISTS %s(" % table
        for column, definition in table_definition.items():
            sql += "%s %s, " % (column, definition)
        sql = sql.rstrip(', ') + ')' # to remove the last ', '
        return sql

    def __check_type(self, df):
        """
        Function that checks the data type.
        :param df: (Pandas DataFrame).
        :return: (String) Data type.
        """
        for value in df.values:
            if value != 'NULL':
                # Numpy array -> string type
                value = str(value)
                if re.match(r'\d+/\d+/\d+', value):
                    return self.DATE_TYPE
                elif re.match(r'[^a-zA-Z]', value):
                    return self.FLOAT_TYPE
                else:
                    return self.STRING_TYPE

    def __load_data_from_pd(self, df, table, table_definition):
        """
        Function that inserts data into the database from a pandas dataframe given a table name and table definition.
        :param df: (Pandas DataFrame).
        :param table: (String) Table name.
        :param table_definition: (Dictionary) See function __create_table().
        :return: None.
        """
        try:
            columns = list(table_definition.keys())
            sql = "INSERT INTO %s(%s) VALUES(" % (table, ', '.join(columns))
            with self.connection.cursor() as cursor:
                for n in range(df.shape[0]):
                    sql_ = sql
                    row_values = df[columns].iloc[n, :].values.tolist()
                    # Cast to string allowing the use of join()
                    row_values = [str(val) for val in row_values]
                    # Check if it is a date and format it according to mysql standards
                    row_values = self.__format_date(row_values)
                    sql_ += "%s);" % ', '.join(row_values)
                    # Insert into table
                    cursor.execute(sql_)
                cursor.execute("SELECT * FROM %s" % table)
            print("Total rows in table %s: %d" % (table, len(cursor.fetchall())))
            self.connection.commit()
        except:
            raise Exception("Upps!! Something wrong check the data")

    def __format_date(self, values):
        """
        Function that formats the date from month / day / year to 'year-month-day'
        :param values: (String). 
        :return: (Lists) Dates formatted or the same value if values not include a date.
        """
        new_values = []
        for value in values:
            if re.match(r'\d+/\d+/\d+', value):
                month, day, year = re.findall(r'(\d+)/(\d+)/(\d+)', value)[0]
                value = "'%s-%s-%s'" % (year, month, day)
            new_values.append(value)
        return new_values


class NeuroDBExplorer(MySQLManager):
    """
    Class designed to obtain specific data from neurological database. It returns pandas dataframes.
    #Methods
        get_tables(): return tables.
        get_pet(): returns a dataframe with PET data.
        get_pet with_threshold): transform quantiative data into qualitative data and return the new dataframe.
    """

    #################################################################
    # PRE-REQUISITES
    #################################################################

    ATLAS = ['brodmann', 'aal']
    DATA_TYPE = ['quantitative', 'qualitative']

    #################################################################
    # PUBLIC METHODS
    #################################################################

    def __init__(self, localhost: str, user: str, password: str):
        super().__init__(localhost, user, password)

    def __repr__(self):
        return "<NeuDB database=%s host=%s user=%s>" % (self.current_database, self.host, self.user)

    def __str__(self):
        return "<NeuDB database=%s host=%s user=%s>" % (self.current_database, self.host, self.user)

    def get_tables(self, *args: str, include_diagnostic=True) -> pandas.DataFrame:
        """
        Function that allows to obtain dataframe of the tables provided as arguments, together with the diagnostic column.
        This method automatically excludes de 'id' column.
        :param args: (String) Table names.
        :params include_diagnostic: (Boolean) Whether to include or not the diagnostic column.
        :return: Pandas dataframe.
        """
        select_tables = ', '.join([w + '.* ' for w in args])
        sql = "SELECT diagnostic.diagnostic, %s FROM diagnostic " % (select_tables)
        for arg in args:
            sql += "JOIN %s ON diagnostic.id = %s.id " % (arg, arg)
        sql += ';'
        df = pandas.DataFrame(self.execute(sql, without='id'))
        # Remove the id column from tables
        for table in args[1:]:
            df = df.drop(labels=(table + '.id'), axis=1)
        if include_diagnostic==False:
            df = df.drop('diagnostic',axis=1)
        return df
    
    def get_pet(self, atlas: str, data_type: str, include_diagnostic=True) -> pandas.DataFrame:
        """
        Function that allows to obtain a dataframe with PET data.
        :param atlas: (String) Brodmann / AAL.
        :param data_type: (String) Quantitative / Qualitative.
        :params include_diagnostic: (Boolean) Whether to include or not the diagnostic column.
        :return: Pandas dataframe / None.
        """
        atlas = atlas.lower()
        data_type = data_type.lower()
        if atlas in self.ATLAS and data_type in self.DATA_TYPE:
            sql = f"""SELECT diagnostic.diagnostic, {atlas}_{data_type}.* 
            FROM diagnostic JOIN {atlas}_{data_type} ON 
            diagnostic.id = {atlas}_{data_type}.id;"""
            df = pandas.DataFrame(self.execute(sql, without='id'))
            if include_diagnostic==False:
                df = df.drop('diagnostic',axis=1)
            return df
        else:
            print("Incorrect query")
            return None

    def get_pet_with_threshold(self, *atlas: str, threshold=None, include_diagnostic=True) -> pandas.DataFrame:
        """
        Function that transforms quantitative data into qualitative data based on a threshold
        set by the user.
        :param atlas: (String) Atlas (Brodmann / AAL)
        :param threshold: (Int) Threshold
        :params include_diagnostic: (Boolean) Whether to include or not the diagnostic column.
        :return: Pandas dataframe.
        """
        if threshold is not None:
            tables = ["%s_quantitative" % at for at in atlas if at.lower() in self.ATLAS]
            df = self.get_tables(*tables,include_diagnostic=include_diagnostic)
            if include_diagnostic==True:
                for col in df.columns.values[1:]:
                    df.loc[df[col] < threshold, col] = 0
                    df.loc[df[col] >= threshold, col] = 1
            else:
                 for col in df.columns.values:
                    df.loc[df[col] < threshold, col] = 0
                    df.loc[df[col] >= threshold, col] = 1               
            return df
        else:
            print("Threshold not provided")
            return None