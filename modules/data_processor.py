import re

class process():
    def __init__(self, config):
        self.config = config

    def pre_processing(self, string : str):
        if self.config["pre_processing"]["token_delimiter_type"] == "chain_of_thought":
            previous_token = "0"
            processed_string = ""
            for next_token in string.split(" "):
                filler = self.chain_of_thought(previous_token, next_token)
                processed_string += next_token + filler
                previous_token = next_token
            return processed_string
        elif self.config["pre_processing"]["token_delimiter_type"] != "blank":
            return string.replace(" ", self.config["pre_processing"]["token_delimiter_type"])
        else:
            return string
        
    def post_processing(self, string : str):
        if self.config["pre_processing"]["token_delimiter_type"] == "chain_of_thought":
            return re.sub(r'<[^>]*>', ' ', string)
        elif self.config["pre_processing"]["token_delimiter_type"] != "blank":
            return string.replace(self.config["pre_processing"]["token_delimiter_type"], " ")
        else:
            return string

    def chain_of_thought(self, num1 : str, num2 : str):
        """
        Process the data using chain of thought approach
        """
        new_token = "<"
        carry = 0
        previous_sum = None

        # Zero pad num1 and num2 to the same length
        num1 = num1.zfill(len(num2))
        num2 = num2.zfill(len(num1))

        for x, y in zip(num1[::-1], num2[::-1]):
            if previous_sum == None:
                new_token += "C" + x + y + "-"
            else:
                if carry == 0:
                    new_token += str(previous_sum) + "C" + x + y + "-"
                else:
                    new_token += str(carry) + str(previous_sum) + "C" + x + y + "-"
            previous_sum = int(x) + int(y) + carry
            if previous_sum > 9:
                carry = 1
                previous_sum = previous_sum - 10
            else:
                carry = 0
        new_token += str(previous_sum) + "C"
        new_token += ">"
        return new_token



if __name__ == "__main__":
    # Simple addition
    config = {"pre_processing": {"token_delimiter_type": "+"}}
    process_object = process(config)
    string = "189 23456 1452 1234567890 "
    print(process_object.pre_processing(string))
    print(process_object.post_processing(process_object.pre_processing(string)))

    # Chain of thought
    config = {"pre_processing": {"token_delimiter_type": "chain_of_thought"}}
    process_object = process(config)
    print(process_object.pre_processing(string))
    print(process_object.post_processing(process_object.pre_processing(string)))
