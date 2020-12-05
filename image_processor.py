from matplotlib import pyplot as plt
import numpy as np

WHITE = [255,255,255]
BOUNDARY_BUFFER = 28

def tranformation(path):
    # helper
    def multipleOf(base, number):
        multiple = 0
        while multiple*base < number:
            multiple += 1
        return multiple*base

    def findLeftBound(start):
        found = -1
        for col in range(start, width):
            for row in range(height):
                if np.sum(image[row][col]) < 200:
                    return col
        return found

    def findRightBound(start):
        for col in range(start, width):
            found = True
            for row in range(height):
                if np.sum(image[row][col]) < 200:
                    found = False
            if found:
                return col
        return found

    def findUpperBound(start, left, right):
        found = -1
        for row in range(start, height):
            for col in range(left, right+1):
                if np.sum(image[row][col]) < 200:
                    return row
        return found

    def findLowerBound(start, left, right):
        for row in range(start, height):
            found = True
            for col in range(left, right+1):
                if np.sum(image[row][col]) < 200:
                    found = False
            if found:
                return row
        return found
    
    # load image into rgb matrix
    image = plt.imread(path)
    image = np.array(image)

    # declare variables
    height, width = np.shape(image)[0], np.shape(image)[1]
    print(height, width)

    # clean the image
    for i in range(height):
        for j in range(width):
            if np.sum(image[i][j]) > 400:
                image[i][j] = WHITE

    result = []
    num_digits = 10
    horizontal_start = 0

    for _ in range(num_digits):
        # find left&right bound
        left = findLeftBound(horizontal_start)
        horizontal_start = left
        right = findRightBound(horizontal_start)
        horizontal_start = right

        # find upper&lower bound
        vertical_start = 0
        upper = findUpperBound(vertical_start, left, right)
        vertical_start = upper
        lower = findLowerBound(vertical_start, left, right)

        digit_width, digit_height = right-left, lower-upper    

        side_length = max(digit_height, digit_width)
        adjust_side_length = multipleOf(28, side_length) + BOUNDARY_BUFFER*2
        left_buffer = (adjust_side_length-digit_width)//2
        upper_buffer = (adjust_side_length-digit_height)//2

        digit = [[WHITE for _ in range(adjust_side_length)] for _ in range(adjust_side_length)]

        for row in range(adjust_side_length):
            for col in range(adjust_side_length):
                if upper_buffer <= row < upper_buffer + digit_height: 
                    if left_buffer <= col < left_buffer + digit_width:
                        digit[row][col] = image[upper+row-upper_buffer][left+col-left_buffer]


        
        multiple = adjust_side_length//28
        scaled_digit = [[WHITE for _ in range(28)] for _ in range(28)]

        for row in range(28):
            for col in range(28):
                pixel_val = 255
                sub_left, sub_right = row*multiple, (row+1)*multiple
                sub_upper, sub_lower = col*multiple, (col+1)*multiple
                for sub_row in range(sub_left, sub_right):
                    for sub_col in range(sub_upper, sub_lower):
                        pixel_val = min(pixel_val, digit[sub_row][sub_col][0])
                        
                scaled_digit[row][col] = [pixel_val, pixel_val, pixel_val]

        result.append(scaled_digit)
    return result




    
    










