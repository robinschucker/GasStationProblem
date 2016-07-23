''' Author: Robin Schucker (schucker.robin[at]gmail.com)
    -----------------------------------------------------
    Simple module that parses .csv file into numpy arrays
'''

import numpy as np
import warnings

class CSVData(object):

    ''' CSVData object is a class that is used to store and parse the data
    coming from a .csv file. The object is created by calling the constructor
    with an open file handle. This class handles two data types, strings and
    floats and each type is stored in its own contiguous block of memory.
    
    Input:
        Any .csv file that has text and numbers in it. Fields must be separated
        by a delimiter (usually ',') and strings that have the delimiters or
        line brakes in them must be wrapped by double quotes.
        E.g:
            first field, "second, field\n", third field, 4.0
            
        If there is an uneven number of quotes in the document, an error is
        generated.
        If there are missing fields, or any type of anomaly (text when number
        is expected). The program will only generate a warning and continue
        reading. Missing/unreadable numeric fields will be filled with NaN
    
    Usage:
        input_file = open('test.csv','r')
        data = CSVData(input_file)
        input_file.close()
    
    Data Stored:
        - Floats are stored in a traditional numpy array for efficiency
        access via: data.number_data
        
        - Strings are stored in a numpy array of dtype = object to be able to 
        store strings of all lengths.
        access via: data.text_data
        
        /!\ Column indexing of numpy array is not the same as in the csv file
        data.number_data[i,0] corresponds to the ith row and first column of
        numerical data in the .csv file.
        
        to use the original indexing use the getter/setter methods:
        data.get_data(i,j) returns the element at ith row, jth column of .csv
        and can be of type string or numeric.
    
    Header:
        If first lines of csv contain header information, calling the
        the constructor with n_header > 0  will store the first n_header
        lines of the file as unparsed data.
    
    Delimiter:
        The standard delimiter is a comma (i.e ','), but the CSVData 
        constructor can also be invoked with a custom delimiter
        For e.g.:
            data = CSVData(input_file, delimiter = ';') 
            would split the data on ';'
        
    Formatting:
        data.get_types() will return the format of the data.
        That format is either determined manually when calling the constructor
        by passing in a numpy array of 0s and 1s.
        For example:
            format = np.array([1,1,0,0,0]) specifies the format for a csv that
            has 5 columns, the first two being strings and rest numeric.
            
        If no format is passed in, the program will use the first line being
        read to set the format.
        
    Load selected parts of .csv:
        Loading only parts of the csv file can be done by passing in the file
        handle at the position to start loading.
        The program will read the file until the end, but it is possible to
        set the number of lines (actual lines in the text file, not number of
        rows if some line breaks are encapsulated in quotes) to read by
        specifying the num_lines argument.
        
    Assumptions:
        - Input file is a valid .csv file (plain text)
        - User is responsible for using indexes that are not out of bounds
        in the setter/getters
        - file handle must be closed by the user when done with calling the
        constructor
    
    Warnings:
        - Warnings on all malformed inputs of the csv file
        
    Errors:
        - Only if there is an uneven number of quotes
    '''
    
    NUMBER_IDENTIFIER = 0
    STRING_IDENTIFIER = 1
    
    ###########################################################################
    ######                    Public Methods                             ######
    ###########################################################################
    ''' Class constructor:
        usage: 
            input_file = open('test.csv','r')
            data = CSVData(input_file)
            input_file.close()
        
        params:
            input_file: open file handle to .csv being read
            n_header: specify number of header lines that won't be parsed
            delimiter: char to split the data on
            format: numpy array of 0s and 1s to specify type of fields manually
                e.g: np.array([1,1,0]) (String, String, numeric)
            num_lines: number of (actual text file) lines to read from file
            
        accessible fields:
            data.text_data: numpy array of dtype = object that stores the
                string fields
            data.number_data: numpy array of floats that stores the numeric
                fields
            to get number of numeric columns use
                sum(data.format == data.NUMBER_IDENTIFIER)
                (or STRING_IDENTIFIER for text columns)
            
            data.format: format numpy array
            data.header: list of raw header lines that were not parsed
            data.num_cols: number of columns stored (numeric plus text)
            data.num_rows: number of rows
            data.delimiter: delimiter the data was split on
            data.lookup_table: to relate column index from csv file to column
                index inside text_data and number_data
    '''
    def __init__(self, input_file, n_header = 0, delimiter = ',', 
                 format = None, num_lines = None):
        self.text_data = None
        self.number_data = None
        self.format = format
        if (self.format is not None):
            self.num_cols = self.format.size
        self.lookup_table = None
        self.delimiter = delimiter
        self.input_file = input_file
        self.num_lines = num_lines
        if (n_header == 0):
            self.header = None
        else:
            self.header = []
            for i in range(n_header):
                self.header.append(self.input_file.readline())
        self.fill()
        
    ''' Method that returns a list of "types" being used for our data
        usage: types = data.get_types()
            now types is ['String','String','Numeric']
    '''    
    def get_types(self):
        result = []
        for x in np.nditer(self.format):
            if (x == self.STRING_IDENTIFIER):
                result.append('String')
            elif (x == self.NUMBER_IDENTIFIER):
                result.append('Numeric')
        return result
    
    ''' Method that returns the content of the stored data at row ridx and
        column (in the original csv file) cidx
        usage: value = data.get_data(i,j)
        assumes: 0 <= i < data.num_rows
                 0 <= j < data.num_cols
    '''
    def get_data(self, ridx, cidx):
        if (self.format[cidx] == self.STRING_IDENTIFIER):
            return self.text_data[ridx][self.lookup_table[cidx]]
        elif (self.format[cidx] == self.NUMBER_IDENTIFIER):
            return self.number_data[ridx][self.lookup_table[cidx]]
        
    ''' Method that changes the content of the stored data at row ridx and
        column (in the original csv file) cidx to value
        usage: data.get_data(i,j,value)
        assumes: 0 <= i < data.num_rows
                 0 <= j < data.num_cols
                 value is of type string if stored in text_data
                 but can be either numeric or string if stored in number_data
                 (if parsing fails, value will become NaN)
    '''
    def set_data(self, ridx, cidx, value):
        ccidx = self.lookup_table[cidx]
        if (self.format[cidx] == self.STRING_IDENTIFIER):
            self.text_data[ridx][ccidx] = value
        elif (self.format[cidx] == self.NUMBER_IDENTIFIER):
            self.number_data[ridx][ccidx] = parse_to_float(value, ridx, cidx)
        
    
    ###########################################################################
    ######                   Private Methods                             ######
    ###########################################################################
    
    # Helper method that creates the lookup table
    def create_lookup_table(self):
        tidx = 0
        nidx = 0
        idx = 0
        self.lookup_table = np.empty_like(self.format)
        for f in np.nditer(self.format):
            if (self.format[idx] == self.STRING_IDENTIFIER):
                self.lookup_table[idx] = tidx
                tidx += 1
            elif (self.format[idx] == self.NUMBER_IDENTIFIER):
                self.lookup_table[idx] = nidx
                nidx += 1
            idx += 1
    
    
    # Helper method that goes through the text file line by line and parses
    # the values inside the internal data structure
    def fill(self):
        # count number of lines to know how much memory to allocate
        if (self.num_lines is None):
            pos = self.input_file.tell()
            self.num_lines = sum(1 for line in self.input_file)
            self.input_file.seek(pos) #return to original position
     
        # define format from first line of file if not inputed manually 
        if (self.format is None): 
            pos = self.input_file.tell()
            _, first_row, _ = self.readrow(0)
            self.line_to_format(first_row)
            self.input_file.seek(pos)
            self.num_cols = self.format.size
        
        # create hash table for internal memory indexing
        self.create_lookup_table()
     
        # allocate memory for data
        self.text_data = np.empty((self.num_lines, 
                sum(self.format == self.STRING_IDENTIFIER)), dtype = object)
        self.number_data = np.empty((self.num_lines, 
                sum(self.format == self.NUMBER_IDENTIFIER)))
    
        # fill data
        print 'Filling data...'
        ridx = 0
        n_line = 0
        while True:
            isend, line, n_line = self.readrow(n_line)
            if (isend):
                break
            self.fill_line(line, ridx)
            ridx += 1
        self.num_rows = ridx
        
        # free extra allocated memory
        self.number_data = self.number_data[0:ridx,:]
        self.text_data = self.text_data[0:ridx,:]
    
    
    # function that returns a line forming an entire row (so ignores lines
    # brakes that are inside of quotes
    # returns isend, line, count where:
    #   isend: True if end of file
    #   line: the string representing an entire row
    #   n_line: lines read in file
    def readrow(self, n_line):
        if (n_line >= self.num_lines):
            return (True, None, None)
        line = self.input_file.readline()
        if not line:
            return (True, None, None)
        n_line += 1
        counter = 0
        while ((line.count('"') % 2) != 0):    
            next_line = self.input_file.readline()
            counter += 1
            if not next_line or (n_line + counter) > self.num_lines:
                raise NameError('At line {} of file, opened quotation '
                                'that is never closed.'.format(n_line))
            line = line + next_line
        return (False, line, n_line + counter)
        
        
    # helper function that actually parses the data given a line of text
    def fill_line(self, line, ridx):
        items = self.split_on_delim(line, ridx)
        ncidx = 0
        tcidx = 0
        j = 0
        for item in items:
            if (j < self.num_cols):
                if (self.format[j] == self.NUMBER_IDENTIFIER):
                    self.number_data[ridx,ncidx] = parse_to_float(item, ridx,j)
                    ncidx += 1
                elif (self.format[j] == self.STRING_IDENTIFIER):
                    self.text_data[ridx,tcidx] = item
                    tcidx += 1
                j += 1
            else:
                str = ('Input data larger than planned num_cols at row. ' 
                       '{} Ignoring excess data.'.format(ridx + 1))
                warnings.warn(str)
                break
        if (j != self.num_cols):
            # fill missing data with nan or ''
            for jj in range(j, self.num_cols):
                if (self.format[jj] == self.NUMBER_IDENTIFIER):
                    self.number_data[ridx,ncidx] = np.NAN
                    ncidx += 1
                elif (self.format[jj] == self.STRING_IDENTIFIER):
                    self.text_data[ridx,tcidx] = ''
                    tcidx += 1   
            str = 'Input data is missing field(s) at row {}.'.format(ridx + 1)
            warnings.warn(str)
        
    # Helper that splits a line into fields. Does not consider the delimiter
    # inside quotes
    def split_on_delim(self, str, ridx):
        items = []
        match = 0
        in_quotes = False #bool that is true if inside of quotes
        for i in range(len(str)):
            if (str[i] == self.delimiter and not in_quotes):
                items.append(str[match:i])
                match = i + 1
            if (str[i] == '"'):
                in_quotes = not in_quotes
        items.append(str[match:])
        return items
             
    # Defines format from a line of data 
    def line_to_format(self, line):
        items = self.split_on_delim(line, 0)
        format = []
        for item in items:
            if (isfloat(item)):
                format.append(self.NUMBER_IDENTIFIER)
            else:
                format.append(self.STRING_IDENTIFIER)
        self.format = np.array(format)

    
# Test if string is float    
def isfloat(item):
    try:
        float(item)
        return True
    except ValueError:
        return False
   
        
# parse float if possible, return NaN otherwise   
def parse_to_float(item, ridx, cidx):
    try:
        return float(item)
    except ValueError:
        str = 'Parsing of float unsuccesful at row {0}, col {1}.' \
                    .format(ridx, cidx)
        warnings.warn(str)
        return np.NAN
        