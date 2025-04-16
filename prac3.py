''' Spelling Correction in IR Systems 
● Develop a spelling correction module using edit distance algorithms. 
● Integrate the spelling correction module into an information retrieval system. '''

def editDistance(str1, str2, m, n):
    if m == 0:
        return n

    if n == 0:
        return m

    if str1[m-1] == str2[n-1]:
        return editDistance(str1, str2, m-1, n-1)

    return 1 + min(
        editDistance(str1, str2, m, n-1),    # Insert
        editDistance(str1, str2, m-1, n),    # Remove
        editDistance(str1, str2, m-1, n-1)   # Replace
    )

# Test case
str1 = "sunday"
str2 = "saturday"
print("Edit Distance is:", editDistance(str1, str2, len(str1), len(str2)))
