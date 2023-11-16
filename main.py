import string
from functools import reduce
from typing import List
from itertools import chain, combinations
from collections import defaultdict


def merge_string(word1, word2):
    output = ''
    if len(word1) == len(word2):
        for idx in range(len(word1)):
            output += word1[idx]
            output += word2[idx]
        return output
    elif len(word1) > len(word2):
        for idx in range(len(word2)):
            output += word1[idx]
            output += word2[idx]
        output += word1[len(word2):len(word1)]
        return output
    else:
        for idx in range(len(word1)):
            output += word1[idx]
            output += word2[idx]
        output += word2[len(word1):len(word2)]
        return output


def problem1431(candies, extra_candies):
    """ Kids with the greatest number of candies """

    output = []
    max_candies = max(candies)
    for candy in candies:
        if candy + extra_candies >= max_candies:
            output.append(True)
        else:
            output.append(False)

    return output


def problem345(s: str) -> str:
    """ Reverse vowels of a string """
    vowels = 'aeiouAEIOU'
    s_vowels = []
    for character in s:
        if character in vowels:
            s_vowels.append(character)
    output = ''
    counter = 0
    for character in s:
        if counter <= len(s_vowels) and character in vowels:
            output += s_vowels[::-1][counter]
            counter += 1
        else:
            output += character

    return output


def problem151(s: str) -> str:
    """ Reverse words in a string """
    return ' '.join(s.split()[::-1])


def problem283(nums):
    """ Move zeroes """
    counter = 0
    for ind in range(len(nums)):
        if nums[ind - counter] == 0:
            zero = nums.pop(ind - counter)
            counter += 1
            nums.append(zero)
    return nums


def problem392(s: str, t: str) -> bool:
    """ Is subsequence """
    for c in s:
        idx = t.find(c)
        if idx == -1:
            return False
        else:
            t = t[idx + 1:]
    return True


def problem643(nums, k):
    """ Maximum average subarray 1 """
    seq_sum = max_sum = sum(nums[:k])
    for idx in range(k, len(nums)):
        seq_sum += nums[idx] - nums[idx - k]
        max_sum = max(max_sum, seq_sum)
    return max_sum / k


def problem724(nums):
    """ Find pivot index """
    left_sum, right_sum = 0, sum(nums)
    for idx in range(len(nums)):
        right_sum -= nums[idx]
        if left_sum == right_sum:
            return idx
        left_sum += nums[idx]
    return -1


def problem202(n):
    already_numbers = []
    while n != 1:
        digits_sum = 0
        while n > 0:
            digit = n % 10
            digits_sum += digit ** 2
            n //= 10

        if digits_sum in already_numbers:
            return False

        already_numbers.append(digits_sum)

        n = digits_sum
    return True


def problem1232(coordinates):
    x1, y1 = coordinates[0][0], coordinates[0][1]
    x2, y2 = coordinates[1][0], coordinates[1][1]

    for idx in range(2, len(coordinates)):
        x3, y3 = coordinates[idx][0], coordinates[idx][1]
        if (x3 - x1) * (y2 - y1) != (y3 - y1) * (x2 - x1):
            return False
    return True


def problem1588(arr):
    output_sum = sum(arr)
    indexes = []

    if len(arr) <= 3:
        return output_sum

    counter = 0
    size = 3
    while len(arr) - size >= 3:
        for i in range(size):
            if len(arr) - counter >= 3:
                indexes.append(idx + counter)
                counter += 1
            size += 2
    return


def problem1732(gain):
    """ Find the highest altitude """
    points = [0]
    for elem in gain:
        points.append(points[-1] + elem)
    return max(points)


def problem1732_v2(gain):
    """ Find the highest altitude """
    max_height = 0
    previous_elem = 0
    for elem in gain:
        previous_elem += elem
        if previous_elem > max_height:
            max_height = previous_elem
    return max_height


def problem2215(nums1, nums2):
    nums1_diff = []
    nums2_diff = []
    for elem in nums1:
        if elem not in nums2 and elem not in nums1_diff:
            nums1_diff.append(elem)
    for elem in nums2:
        if elem not in nums1 and elem not in nums2_diff:
            nums2_diff.append(elem)
    return [nums1_diff, nums2_diff]


def problem1207(arr):
    counter = []
    for elem in set(arr):
        if arr.count(elem) not in counter:
            counter.append(arr.count(elem))
        else:
            return False
    return True


def problem238(nums):
    return [reduce(lambda a, b: a * b, nums[idx + 1: len(nums)] + nums[0: idx]) for idx in range(len(nums))]
    # answer = []
    # for idx in range(len(nums)):
    #     answer.append(reduce(lambda a, b: a * b, nums[idx + 1: len(nums)] + nums[0: idx]))
    # print(answer)


def problem1071(str1, str2):
    """ Greatest common divisor of string """
    if len(str1) > len(str2):
        counter = len(str1) // len(str2)
        flag = True
    else:
        counter = len(str2) // len(str1)
        flag = False


def problem334(nums):
    """ Increasing triplet subsequence """
    first = second = float('inf')

    for elem in nums:
        if elem <= first:
            first = elem
        elif elem <= second:
            second = elem
        elif elem > second:
            return True
    return False


def problem443(chars):
    output = []
    # unique_chars = set(chars)
    chars_count = {}
    for char in chars:
        if char not in chars_count.keys():
            chars_count[char] = chars.count(char)

    for key, value in chars_count.items():
        if value < 10:
            output.append(key)
            output.append(str(value))
        else:
            output.append(key)
            for idx in range(len(str(value))):
                output.append(str(value)[idx])
    chars = output

    return len(output)
    # output = 0
    # for char in set(chars):
    #     if chars.count(char) == 1:
    #         output += 1
    #     else:
    #         output += 1
    #         output += len(str(chars.count(char)))
    # return output


def problem118(num_rows):
    output = [[1]]
    idx = 0
    while num_rows > 1:
        if len(output) == 1:
            output.append([1, 1])
            idx += 1
            num_rows -= 1
        if num_rows == 1:
            break
        row = [1]
        for i in range(1, len(output[idx])):
            row.append(output[idx][i-1] + output[idx][i])
        row.append(1)
        output.append(row)
        idx += 1
        num_rows -= 1
    return output


def problem506(score: List[int]):

    result = []
    best_scores = {
        1: 'Gold Medal',
        2: 'Silver Medal',
        3: 'Bronze Medal',
    }
    sorted_scores = sorted(score, reverse=True)
    for elem in score:
        elem_idx = sorted_scores.index(elem) + 1
        if elem_idx in best_scores:
            result.append(best_scores[elem_idx])
        else:
            result.append(str(elem_idx))

    return result


def problem1647(s):
    frequency_list = []
    counter = 0
    for char in set(s):
        char_frequency = s.count(char)
        while char_frequency in frequency_list:
            char_frequency -= 1
            counter += 1
        else:
            frequency_list.append(char_frequency)
        if 0 in frequency_list:
            frequency_list.remove(0)
    return counter


def problem1800(nums):
    max_sum = current_sum = 0
    if len(nums) == 1:
        return nums[0]
    for idx in range(0, len(nums) - 1):
        if current_sum == 0:
            current_sum = nums[idx]
        if nums[idx + 1] > nums[idx]:
            current_sum += nums[idx + 1]
        else:
            if current_sum > max_sum:
                max_sum = current_sum
            current_sum = 0
    if max_sum == 0 or current_sum > max_sum:
        return current_sum
    else:
        return max_sum


def problem495(time, duration):
    poison_seconds = 0
    for i in range(len(time) - 1):
        if time[i + 1] - time[i] > duration:
            poison_seconds += duration
        else:
            poison_seconds += time[i + 1] - time[i]
    return poison_seconds + duration


def problem342(n):
    if n == 1:
        return True
    else:
        return n / 4 == 1 if n <= 4 else problem342(n / 4)


def problem119(row_index):
    if row_index == 0:
        return [1]
    output_row = [1, 1]

    for i in range(2, row_index + 1):
        curr_row = [output_row[0]]

        for j in range(1, len(output_row)):
            curr_row.append(output_row[j] + output_row[j - 1])

        curr_row.append(output_row[-1])
        output_row = curr_row

    return output_row


def problem779(n, k):

    flag = True
    all_row_len = 2 ** (n - 1)

    while all_row_len != 1:

        all_row_len //= 2
        if k > all_row_len:
            k -= all_row_len
            flag = not flag

    return 0 if flag else 1


def problem2785(s):
    vowels = 'AEIOUaeiou'
    output = ''
    vowels_in_str = sorted(list(filter(lambda letter: letter in vowels, s)))
    if not vowels_in_str:
        return s
    idx, counter = 0, 0
    while idx < len(s) and counter <= len(vowels_in_str):
        if s[idx] in vowels:
            output += vowels_in_str[counter]
            counter += 1
        else:
            output += s[idx]
        idx += 1
    return output


def problem1930(s):
    counter = 0
    letters = string.ascii_lowercase

    for letter in letters:
        first_index = s.find(letter)
        last_index = s.rfind(letter)
        if first_index != -1:
            counter += len(set(s[first_index + 1:last_index]))
    return counter


def problem1846(arr):
    if len(arr) == 1:
        return 1
    if 1 in arr and max(arr) <= len(arr) and sorted(arr)[-1] - sorted(arr)[-2] <= 1:
        return max(arr)
    elif 1 in arr and max(arr) <= len(arr) and sorted(arr)[-1] - sorted(arr)[-2] > 1:
        return sorted(arr)[-2] + 1
    elif max(arr) > len(arr):
        return len(arr)





if __name__ == '__main__':
    pass

