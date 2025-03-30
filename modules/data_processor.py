import re

class process():
    def __init__(self, config):
        self.config = config

    def pre_processing(self, string : str):

        # Reverse the series
        if self.config["pre_processing"]["reverse_series"]:
            string = str(self.config["pre_processing"]["column_delimiter"]).join([row[::-1] for row in string.split(self.config["pre_processing"]["column_delimiter"])])
            

        if self.config["pre_processing"]["replace_column_delimiter"] :
            return string.replace(" ", self.config["pre_processing"]["replace_column_delimiter"] )
        else:
            return string
        
    def post_processing(self, string : str):

        # Replace the column delimiter
        if self.config["pre_processing"]["replace_column_delimiter"] :
            string = string.replace(self.config["pre_processing"]["replace_column_delimiter"], " ")

        # Reverse the series
        if self.config["pre_processing"]["reverse_series"]:
            string = str(self.config["pre_processing"]["column_delimiter"]).join([row[::-1] for row in string.split(self.config["pre_processing"]["column_delimiter"])])

        return string


if __name__ == "__main__":
    # Option 1: Replace the column delimiter
    config = {"pre_processing": {"replace_column_delimiter": "+", "reverse_series": True, "column_delimiter": " "}}
    process_object = process(config)
    string = " 189 23456 1452 1234567890 "
    print(process_object.pre_processing(string))
    print(process_object.post_processing(process_object.pre_processing(string)))

    # Option 2: No reverse the series
    config = {"pre_processing": {"replace_column_delimiter": "+", "reverse_series": False, "column_delimiter": " "}}
    process_object = process(config)
    print(process_object.pre_processing(string))
    print(process_object.post_processing(process_object.pre_processing(string)))
