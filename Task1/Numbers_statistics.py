# ============================================================
# Homework: Numbers Statistics
# Purpose: Read numbers from the user, store them in a List
#          and a Dictionary (number: frequency), then compute
#          the sum, average, maximum, and minimum.
# ============================================================
# ============================================================
# Homework: Numbers Statistics
# Purpose: Read numbers from the user, store them in a List
#          and a Dictionary (number: frequency), then compute
#          the sum, average, maximum, and minimum.
# ============================================================
import re

# ---- Step 1: Take user input ----
raw_input_text = input("Enter numbers separated by spaces or commas: ")

# ---- Step 2: Store the numbers in a List ----
numbers_list = []
skipped = []

for piece in re.split(r"[\s,;]+", raw_input_text.strip()):
    if not piece:       
        continue
    try:
        numbers_list.append(float(piece))
    except ValueError:    
        skipped.append(piece)

if skipped:
    
    print("Ignored (not numbers):", skipped)

if not numbers_list:
    print("No valid numbers were entered.")
    raise SystemExit

print("\nList of numbers:", numbers_list)

# ---- Step 3: Dictionary ----
frequency_dict = {}
for number in numbers_list:
    frequency_dict[number] = frequency_dict.get(number, 0) + 1

print("Dictionary (number: frequency):", frequency_dict)

# ---- Step 4: Statistics ----
total = sum(numbers_list)
average = total / len(numbers_list)
maximum = max(numbers_list)
minimum = min(numbers_list)

# ---- Step 5: Display ----
print("\n----- Statistics -----")
print("Count of numbers:", len(numbers_list))
print("Sum:", total)
print("Average:", average)
print("Maximum:", maximum)
print("Minimum:", minimum)
