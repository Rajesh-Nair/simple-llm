
import random

def generate_sum(min_value, max_value,retrieve_percent=1):
    
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
      for i in range(min_value, max_value+1):
          for j in range(min_value, max_value+1):
              seq.append([i,j,i+j])      

    # retrieve a random sample of combinations
    else:
      for i in random.sample(range(min_value, max_value+1), int(retrieve_percent*(max_value-min_value+1))):
          for j in random.sample(range(min_value, max_value+1), int(retrieve_percent*(max_value-min_value+1)) ):
              seq.append([i,j,i+j]) 

    # shuffle the combinations
    random.shuffle(seq)

    return seq

def generate_series(initial, mask, n, min_length=0):
    """
    Generates a sequence using a custom recurrence relation.

    The recurrence relation is determined by the mask. At each step,
    the next number is computed by summing those of the last len(mask)
    numbers where the mask has a 1.
    
    Parameters:
      initial (list): List of initial numbers.
      mask (list): List of 0s and 1s indicating which of the previous
                   numbers to include in the sum.
      n (int): Total number of sequence elements to generate (including the initial ones).
      min_length (int): Minimum total string length of sequence when joined with spaces.

    Returns:
      list: The generated sequence of numbers.
    """

    seq = initial.copy()
    total_length = len(" ".join(str(x) for x in seq))
    
    while (len(seq) < n or total_length < min_length) :
        # Get the last len(mask) elements from the sequence
        if len(seq) < len(mask):
            # Pad with zeros if sequence is shorter than mask
            last_elements = [0] * (len(mask) - len(seq)) + seq
        else:
            last_elements = seq[-len(mask):]
        # Calculate the next number using the mask (only add numbers where mask is 1)
        next_val = sum(num for num, m in zip(last_elements, mask) if m == 1)
        seq.append(next_val)
        total_length += len(str(next_val)) + 1  # Add length of new number plus space
    return seq


if __name__ == "__main__":
    # Example usage:
  # Fibonacci-like sequence: initial numbers [1, 2] and mask [1, 1]
  my_sequence = generate_series([1], [1, 1], 10)
  print(my_sequence)  # Output: [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]


  my_sequence = generate_sum(0, 9, 0.5)
  print(my_sequence)  

  my_sequence = generate_sum(11, 19, 1)
  print(my_sequence)  

  for row in my_sequence:
      print(" ".join(str(x) for x in row))
