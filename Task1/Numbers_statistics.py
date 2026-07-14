# ============================================================
# Homework: Numbers Statistics
# Purpose: Read numbers from the user, store them in a List
#          and a Dictionary (number: frequency), then compute
#          the sum, average, maximum, and minimum.
# ============================================================

# ---- Step 1: Take user input ----
# The user types numbers separated by spaces, e.g. "5 3 8 3 10"
raw_input_text = input("Enter numbers separated by spaces: ")

# ---- Step 2: Store the numbers in a List ----
# split() breaks the text into pieces, and float() converts
# each piece into a number. We keep them all in a list.
numbers_list = []
for piece in raw_input_text.split():
    numbers_list.append(float(piece))

print("\nList of numbers:", numbers_list)

# ---- Step 3: Store the numbers in a Dictionary ----
# The dictionary maps each number to how many times it appears
# (its frequency). Example: [5, 3, 3] -> {5.0: 1, 3.0: 2}
frequency_dict = {}
for number in numbers_list:
    if number in frequency_dict:
        frequency_dict[number] += 1   # seen before -> add 1
    else:
        frequency_dict[number] = 1    # first time -> start at 1

print("Dictionary (number: frequency):", frequency_dict)

# ---- Step 4: Basic statistical operations ----
# sum()  -> adds all the numbers together
# len()  -> counts how many numbers there are (for the average)
# max()  -> the largest value
# min()  -> the smallest value
total = sum(numbers_list)
average = total / len(numbers_list)
maximum = max(numbers_list)
minimum = min(numbers_list)

# ---- Step 5: Display the results ----
print("\n----- Statistics -----")
print("Count of numbers:", len(numbers_list))
print("Sum:", total)
print("Average:", average)
print("Maximum:", maximum)
print("Minimum:", minimum)
