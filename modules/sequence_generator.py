
import random
from tqdm import tqdm
import math

def generate_sum(min_value, max_value,retrieve_percent=1, max_length=None):
    
    """
    Generates a sequence using a custom recurrence relation. 
    The sequence is a list of lists, where the first element is the min_value, the second element is the max_value, and the third element is the sequence of numbers.
    
    Parameters:
      min_value (int): The minimum value of the sequence.
      max_value (int): The maximum value of the sequence.
      retrieve_percent (float): The percentage of combinations to retrieve.
    
    Returns:
      list: The generated sequence of numbers.
    """
    seq = []
    # retrieve all possible combinations
    if retrieve_percent == 1:
      for i in tqdm(range(min_value, max_value+1)):
          for j in tqdm(range(min_value, max_value+1)):
              if max_length is None or len(str(i)) + len(str(j)) <= max_length:
                  k = i+j
                  seq.append([i,j,k])      

    # retrieve a random sample of combinations
    else:
      sample_rate = int((retrieve_percent*(max_value-min_value+1)))
      for i in tqdm(random.sample(range(min_value, max_value+1), sample_rate)):
          for j in tqdm(random.sample(range(min_value, max_value+1), sample_rate)):
              if max_length is None or len(str(i)) + len(str(j)) <= max_length:
                  k = i+j
                  seq.append([i,j,k]) 

    # shuffle the combinations
    random.shuffle(seq)

    return seq




if __name__ == "__main__":
    # Example usage:
  my_sequence = generate_sum(0, 9, 0.5)
  print(my_sequence)  

  my_sequence = generate_sum(11, 19, 1)
  print(my_sequence)  

  for row in my_sequence:
      print(" ".join(str(x) for x in row))
