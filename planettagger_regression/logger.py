from __future__ import division, print_function
import os
import sys

class Logger(object):

    def __init__(self, log_path, on=True):
        """
        Initialize the logger object.

        Parameters
        ---------
        log_path : str
            the path to the log file
        on : bool
            log or not
        """
        self.log_path = log_path
        self.on = on

        if self.on:
            while os.path.isfile(self.log_path):
                self.log_path += '+'

    def log(self, string, newline=True):
        """
        Write to log file.

        Parameters
        ---------
        string : str
            the string to write to the logfile
        newline : bool
            whether to write a new line after the string
        """
        if self.on:
            with open(self.log_path, 'a') as logf:
                logf.write(string)
                if newline: 
                    logf.write('\n')

            sys.stdout.write(string)
            if newline: 
                sys.stdout.write('\n')
            sys.stdout.flush()
