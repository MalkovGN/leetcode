import string
from functools import reduce
from typing import List, Optional, Tuple
from collections import Counter, defaultdict


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


def problem1980(nums):
    binary_str = ['0'] * len(nums)
    idx = len(nums) - 1

    while ''.join(binary_str) in nums:
        if binary_str[idx] == '0':
            binary_str[idx] = '1'
        else:
            binary_str[idx] = '0'
        if idx == 0:
            idx += 10
        else:
            idx -= 1

    return ''.join(binary_str)


def problem1877(nums):
    first_idx = 0
    last_idx = len(nums) - 1
    sorted_nums = sorted(nums)
    max_pair_sum = sorted_nums[first_idx] + sorted_nums[last_idx]

    while last_idx - first_idx >= 1:
        if sorted_nums[first_idx] + sorted_nums[last_idx] > max_pair_sum:
            max_pair_sum = sorted_nums[first_idx] + sorted_nums[last_idx]
        first_idx += 1
        last_idx -= 1

    return max_pair_sum


def problem1838(nums, k):
    counter = 0
    sorted_nums = sorted(nums)
    for idx in range(len(sorted_nums)):
        k += sorted_nums[idx]
        if sorted_nums[idx] * (idx - counter + 1) > k:
            k -= sorted_nums[counter]
            counter += 1

    return len(sorted_nums) - counter


def problem2391(garbage, travel):
    truck_idx = {
        'P': 0,
        'G': 0,
        'M': 0,
    }
    p_minutes = garbage[0].count('P')
    g_minutes = garbage[0].count('G')
    m_minutes = garbage[0].count('M')
    for idx in range(1, len(garbage)):
        if 'P' in garbage[idx]:
            p_minutes += sum(travel[truck_idx['P']:idx])
            p_minutes += garbage[idx].count('P')
            truck_idx['P'] = idx
        if 'M' in garbage[idx]:
            m_minutes += sum(travel[truck_idx['M']:idx])
            m_minutes += garbage[idx].count('M')
            truck_idx['M'] = idx
        if 'G' in garbage[idx]:
            g_minutes += sum(travel[truck_idx['G']:idx])
            g_minutes += garbage[idx].count('G')
            truck_idx['G'] = idx

    return p_minutes + m_minutes + g_minutes


def reverse_int(number):
    return int(str(number)[::-1])


def problem1814(nums):
    counter = 0
    for idx in range(len(nums)):
        nums[idx] = nums[idx] - reverse_int(nums[idx])

    c = Counter(nums)
    for value in c.values():
        counter += (value * (value - 1)) // 2
    return counter % (10 ** 9 + 7)


def problem1424(nums):
    elements_dict = {}
    output = []

    for idx, elem in enumerate(nums):
        for e_idx, e_elem in enumerate(elem):
            if idx + e_idx not in elements_dict.keys():
                elements_dict[idx + e_idx] = [e_elem]
            else:
                elements_dict[idx + e_idx].append(e_elem)

    for elem in elements_dict.values():
        output += elem[::-1]

    return output


def problem1630(nums, l, r):
    result = []
    for idx in range(len(l)):
        sub_nums = nums[l[idx]:r[idx] + 1]
        sub_nums.sort()
        d = sub_nums[1] - sub_nums[0]
        for i in range(len(sub_nums) - 1):
            if sub_nums[i] + d == sub_nums[i + 1] and i == len(sub_nums) - 2:
                result.append(True)
            elif sub_nums[i] + d == sub_nums[i + 1]:
                continue
            else:
                result.append(False)
                break

    return result


def problem1561(piles):
    return sum(sorted(piles)[len(piles) // 3::2])


def problem1464(nums: List[int]) -> int:
    return (sorted(nums, reverse=True)[0] - 1) * (sorted(nums, reverse=True)[1] - 1)


def problem1662(word1: List[str], word2: List[str]) -> bool:
    return ''.join(word1) == ''.join(word2)


def problem1266(points: List[List[int]]) -> int:
    seconds = 0

    for idx in range(len(points) - 1):
        seconds += max(abs(points[idx + 1][0] - points[idx][0]), abs(points[idx + 1][1] - points[idx][1]))

    return seconds


def problem1716(n: int) -> int:
    if n <= 7:
        return sum([elem for elem in range(1, n + 1)])
    else:
        weeks = n // 7
        return 28 * weeks + 7 * weeks + sum([elem for elem in range(weeks + 1, weeks + 1 + (n - n // 7 * 7))])


def problem1582(mat: List[List[int]]) -> int:
    count_special_idx = 0
    for row in mat:
        if sum(row) == 1:
            ones_idx = row.index(1)
            if sum([column[ones_idx] for column in mat]) == 1:
                count_special_idx += 1
    return count_special_idx


def problem169(nums: List[int]) -> int:
    return [elem for elem in set(nums) if nums.count(elem) > len(nums) // 2][0]


def problem1913(nums: List[int]) -> int:
    nums.sort()
    return nums[-1] * nums[-2] - nums[0] * nums[1]


def problem135(ratings: List[int]) -> int:
    candies = [1] * len(ratings)

    for idx in range(1, len(ratings)):
        if ratings[idx] > ratings[idx - 1]:
            candies[idx] = candies[idx - 1] + 1
        elif ratings[idx] == ratings[idx - 1]:
            candies[idx] = candies[idx - 1]

    for idx in range(len(ratings) - 2, -1, -1):
        if ratings[idx] > ratings[idx + 1]:
            candies[idx] = max(candies[idx], candies[idx + 1] + 1)
    return sum(candies)


def problem121(prices: List[int]) -> int:
    profit = 0
    buy = prices[0]
    for cost in prices[1:]:
        if cost > buy:
            profit = max(profit, cost - buy)
        else:
            buy = cost
    return profit


def problem2706(prices: List[int], money: int) -> int:
    prices.sort()
    return money - prices[0] - prices[1] if money >= prices[0] + prices[1] else money


def problem383(ransom_note: str, magazine: str) -> bool:
    for letter in ransom_note:
        if letter in magazine:
            idx = magazine.index(letter)
            magazine = magazine[:idx] + magazine[idx + 1:]
        else:
            return False
    return True


def problem1637(points: List[List[int]]) -> int:
    points = sorted(points, key=lambda x: x[0])
    return max([points[idx + 1][0] - points[idx][0] for idx in range(len(points) - 1)])


def problem1422(s: str) -> int:
    len_s: int = len(s)
    max_score: int = 0
    for idx in range(1, len_s):
        max_score = max(max_score, s[0:idx].count('0') + s[idx:len_s].count('1'))
    return max_score


def problem55(nums: List[int]) -> bool:
    next_idx = 0
    for i in range(len(nums)):
        if i > next_idx:
            return False
        next_idx = max(next_idx, i + nums[i])
    return True


def problem1496(path: str) -> bool:
    points: List[tuple[int, int]] = [(0, 0)]
    for letter in path:
        if letter == 'N':
            point: tuple[int, int] = (points[-1][0], points[-1][1] + 1)
        elif letter == 'S':
            point: tuple[int, int] = (points[-1][0], points[-1][1] - 1)
        elif letter == 'E':
            point: tuple[int, int] = (points[-1][0] + 1, points[-1][1])
        elif letter == 'W':
            point: tuple[int, int] = (points[-1][0] - 1, points[-1][1])

        if point in points:
            return True
        else:
            points.append(point)

    return False


def problem1758(s: str) -> int:
    count_0, count_1 = 0, 0

    for idx in range(len(s)):
        if idx % 2 == 0:
            count_0 += s[idx] != '0'
            count_1 += s[idx] != '1'
        else:
            count_0 += s[idx] != '1'
            count_1 += s[idx] != '0'

    return min(count_0, count_1)


def problem1578(colors: str, needed_time: List[int]) -> int:

    time_counter = 0
    curr_color_idx = []
    for idx in range(len(colors) - 1):
        if colors[idx] == colors[idx + 1]:
            curr_color_idx.append(idx)
            curr_color_idx.append(idx + 1)
        else:
            times = [needed_time[i] for i in set(curr_color_idx)] if curr_color_idx else [0]
            time_counter += sum(times) - max(times)
            curr_color_idx.clear()
    times = [needed_time[i] for i in set(curr_color_idx)] if curr_color_idx else [0]
    time_counter += sum(times) - max(times)
    return time_counter


def problem1897(words: List[str]) -> bool:
    return all([''.join(words).count(letter) % len(words) == 0 for letter in set(''.join(words))])


if __name__ == '__main__':
    print(problem1897(["abbab"]))
