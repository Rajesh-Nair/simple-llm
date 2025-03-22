def generate_sequence(initial, mask, n):
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

    Returns:
      list: The generated sequence of numbers.
    """

    seq = initial.copy()
    while len(seq) < n :
        # Get the last len(mask) elements from the sequence
        if len(seq) < len(mask):
            # Pad with zeros if sequence is shorter than mask
            last_elements = [0] * (len(mask) - len(seq)) + seq
        else:
            last_elements = seq[-len(mask):]
        # Calculate the next number using the mask (only add numbers where mask is 1)
        next_val = sum(num for num, m in zip(last_elements, mask) if m == 1)
        seq.append(next_val)
    return seq


if __name__ == "__main__":
    # Example usage:
  # Fibonacci-like sequence: initial numbers [1, 2] and mask [1, 1]
  my_sequence = generate_sequence([1], [1, 1], 10)
  print(my_sequence)  # Output: [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
