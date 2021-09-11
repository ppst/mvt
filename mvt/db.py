#!/usr/bin/python3

import sqlite3


__author__     = "Philip Stegmaier"
__contact__    = "https://github.com/ppst/mvt/issues"
__copyright__  = "Copyright (c) 2021, Philip Stegmaier"
__license__    = "https://en.wikipedia.org/wiki/MIT_License"
__maintainer__ = "Philip Stegmaier"
__version__    = "0.1.0"


class Database:
    
    connection = None
    
    
    def get_labels(self):
        """Returns data point indexes and label values.

        Returns
        -------
        list
            An array of arrays. The inner arrays contain `[label_num, label_value]`.
            The label values are the class labels 0 or 1.
        """
        print("Getting labels")
        labels = [[],[]]
        if self.connection == None:
            return pairs
        cur = self.connection.cursor()
        for row in cur.execute("SELECT label_num, label_value FROM label"):
            labels[row[1]].append(row)
        return labels
    
    
    
    def get_feature_vector_nums(self, labelNum):
        """Returns the feature vector indexes (`feature_vector_num`) of a data point
        identified by `label_num`.

        Parameters
        ----------
        labelNum : int
            The `label_num` of the data point.
            
        Returns
        -------
        list
            The list of feature vector indexes.
        """
        vnums = []
        if self.connection == None:
            return vnums
        cur = self.connection.cursor()
        for row in cur.execute("SELECT feature_vector_num FROM feature_vector WHERE label_num=%i" % labelNum):
            vnums.append(row[0])
        return vnums
    
    
    
    def get_feature_vectors(self, labelNum):
        """Returns feature vectors of data point identified by `label_num`

        Parameters
        ----------
        labelNum : int
            The `label_num` of the data point.
            
        Returns
        -------
        list
            The list of feature vectors.
        """
        if self.connection == None:
            return []
        cur = self.connection.cursor()
        vectors = []
        for v in cur.execute("SELECT * FROM feature_vector WHERE label_num=%i" % labelNum):
            vectors.append(v)
        return vectors
        
    
    
    def get_feature_vector(self, labelNum, vectorNum):
        """Returns the feature vector identified by `label_num` and `feature_vector_num`.

        Parameters
        ----------
        labelNum : int
            The `label_num` of the data point.
        
        vectorNum: int
            The `feature_vector_num` of the feature vector.
            
        Returns
        -------
        list
            The list of database values.
        
        Raises
        ------
        ValueError
            If zero or more than one feature vector is returned.
        """
        if self.connection == None:
            return []
        cur = self.connection.cursor()
        cur.execute("SELECT * FROM feature_vector WHERE label_num=%i AND feature_vector_num=%i" % (labelNum, vectorNum))
        result = cur.fetchall()
        if result == None or len(result) == 0:
            raise ValueError("Error: could not find vector %i of label %i" % (vectorNum, labelNum))
        elif len(result) > 1:
            raise ValueError("Error: more than one row matching vector %i and label %i: %i" % (vectorNum, labelNum, len(result)))
        return result[0]
            
    
    
    def get_feature_columns(self, vectorNo = 1):
        """Returns column names of `feature_vector` table

        Parameters
        ----------
        vectorNo : int, optional
            A valid primary key to select a feature vector.
            
        Returns
        -------
        list
            List of column names.
        """
        if self.connection == None:
            return []
        cur = self.connection.cursor()
        cur.execute("SELECT * FROM feature_vector WHERE feature_vector_no=%i" % vectorNo)
        return list(map(lambda x: x[0], cur.description))
        
    
    
    def connect(self, dbfile):
        """Connects to SQLite database file.
        
        Parameters
        ----------
        dbfile : str
            SQLite database file path.
        """
        self.connection = sqlite3.connect(dbfile)
    
    
    
    def disconnect(self):
        """Disconnects from database.
        """
        if not self.connection == None:
            self.connection.close()
            
            
            
    def close(self):
        """Synonym for `disconnect`
        """
        self.disconnect()
        
    
