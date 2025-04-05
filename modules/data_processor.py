import re

def convert_from_base10(num, to_base):
    if to_base >= 2 and to_base <= 16:
        digits = "0123456789ABCDEF"
        result = ""
        while num > 0:
            remainder = num % to_base
            result = digits[remainder] + result
            num //= to_base
        return result
    else:
        raise ValueError("Invalid base")

def convert_to_base10(num_str, from_base):
    """
    Convert a number string from given base to base 10.
    
    Args:
        num_str (str): Number string to convert
        base (int): Base of the input number (2-16)
        
    Returns:
        int: Number in base 10
    """
    if from_base >= 2 and from_base <= 16:
        digits = "0123456789ABCDEF"
        result = 0
        for i, digit in enumerate(num_str[::-1]):
            value = digits.index(digit)
            if value >= from_base:
                raise ValueError(f"Invalid digit {digit} for base {from_base}")
            result += value * (from_base ** i)
        return result
    else:
        raise ValueError("Invalid base")

class process():
    def __init__(self, config, base=None):
        self.config = config
        self.base = base

    # Assume the base here is to_base
    def pre_processing(self, string : str):

        # Convert the base to decimal
        if self.base :
            string = str(self.config["pre_processing"]["column_delimiter"]).join(convert_from_base10(int(row), to_base=self.base) for row in string.split(self.config["pre_processing"]["column_delimiter"]))

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

        # Convert the base to decimal
        if self.base :
            string = str(self.config["pre_processing"]["column_delimiter"]).join(str(convert_to_base10(row, from_base=self.base)) for row in string.split(self.config["pre_processing"]["column_delimiter"]))

        return string


if __name__ == "__main__":
    # Option 1: Replace the column delimiter
    config = {"pre_processing": {"replace_column_delimiter": "+", "reverse_series": True, "column_delimiter": " "}}
    process_object = process(config, base = 2)
    string = "11 21 32"
    print(process_object.pre_processing(string))
    print(process_object.post_processing(process_object.pre_processing(string)))

    # Option 2: No reverse the series
    config = {"pre_processing": {"replace_column_delimiter": "+", "reverse_series": False, "column_delimiter": " "}}
    process_object = process(config, base = 16)
    print(process_object.pre_processing(string))
    print(process_object.post_processing(process_object.pre_processing(string)))
