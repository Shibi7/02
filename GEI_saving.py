import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
new_folder = "C:/Users/KRS/PycharmProjects/pythonProject/GaitDatasetB-Silh"
data = "C:/Users/KRS/PycharmProjects/pythonProject/Extracted_Silhoutte01"
# data = "./E_S"

def CenterOfMass(img):
    height, width = img.shape  # Import height and width from frames
    X = []  # Sum of x-coordinates of each row
    Y = []  # Sum of y-coordinates of each column
    WoR = []  # Weights of rows
    WoC = []  # Weights of columns
    for i in range(height):
        Nx = []  # x-coordinates of silhouette pixels in a single row
        for j in range(width):
            if img[i, j] == 255:  # If the pixel is white
                Nx.append(j)  # Append its x-coordinate
        if len(Nx) != 0:  # If the row is not empty
            X.append(sum(Nx))  # Append the sum of the x-coordinates of the row
            WoR.append(len(Nx))  # Append the number of white pixels of the row
    CoMX = sum(X) // sum(WoR)  # The average of the x-coordinates is their center of mass on x
    for j in range(width):
        Ny = []  # y-coordinates of silhouette pixels in a single column
        for i in range(height):
            if img[i, j] == 255:  # If the pixel is white
                Ny.append(i)  # Append its y-coordinate
        if len(Ny) != 0:  # If the column is not empty
            Y.append(sum(Ny))  # Append the sum of the y-coordinates of the column
            WoC.append(len(Ny))  # Append the number of white pixels of the column
    CoMY = sum(Y) // sum(WoC)  # The average of the y-coordinates is their center of mass on y
    return CoMX, CoMY  # Return the coordinates of the center of mass of the silhouette


# Finding the corner coordinates of the rectangle surrounding the silhouette
def Border(img):
    height, width = img.shape  # Import height and width from frames
    indexl = []  # x-coordinates of left-most pixel of each row
    indexr = []  # x-coordinates of right-most pixel of each row
    for i in range(height):  # For each row
        for j in range(width):  # Start from the left-most pixel
            if img[i, j] != 0:  # If the pixel is not black
                indexl.append(j)  # Append its x-coordinate
                break  # Move to next row
        for j in reversed(range(width)):  # Start from the right-most pixel
            if img[i, j] != 0:  # If the pixel is not black
                indexr.append(j)  # Append its x-coordinate
                break  # Move to next row
    x = min(indexl)  # The left-most x-coordinate in all rows
    X = max(indexr)  # The right-most x-coordinate in all rows
    indext = []  # y-coordinates of top-most pixel of each column
    indexb = []  # y-coordinates of bottom-most pixel of each column
    for j in range(width):
        for i in range(height):  # Start from the top-most pixel
            if img[i, j] != 0:  # If the pixel is not black
                indext.append(i)  # Append its y-coordinate
                break  # Move to next column
        for i in reversed(range(height)):  # Start from the bottom-most pixel
            if img[i, j] != 0:  # If the pixel is not black
                indexb.append(i)  # Append its y-coordinate
                break  # Move to next column
    y = min(indext)  # The top-most y-coordinate in all columns
    Y = max(indexb)  # The bottom-most y-coordinate in all columns
    return x, X, y, Y


# Cropping the silhouette based on the results from Border function
def Crop(img, x, X, y, Y):
    Crop = img[y:Y, x:X]
    return Crop


# Aligning the silhouettes based on their center of mass and averaging them
def Add(old, new, center, black, i):  # Current GEI, new frame, center of mass, black background, frame index
    center = list(center)  # Width and height of silhouettes
    # print(center)
    height, width = old.shape  # Height and width of the frames
    # print(height,width)
    x = (width // 2) - center[0]  # left x-coordinate of the silhouette's frame
    y = (height // 2) - center[1]  # top y-coordinate of the silhouette's frame
    height, width = new.shape  # Width and height of a black background
    # print(height,width)
    X = (x + width)  # right x-coordinate of the silhouette's frame
    Y = (y + height)  # bottom y-coordinate of the silhouette's frame
    a = 1.0 / (i + 1)  # New silhouette weight
    b = 1.0 - a  # Current GEI weight
    new = cv2.addWeighted(new, a, old[y:Y, x:X], b, 0.0)  # Combine the new frame inside the silhouette
    old = cv2.addWeighted(black, a, old, b, 0.0)  # Combine the new frame outside the silhouette
    old[y:Y, x:X] = new  # GEI
    return old

def CreateGEt(Making_GET=False):
    if (not os.path.isdir(data)):
        os.mkdir(data)
    if(Making_GET):
        for person in os.listdir(new_folder):
            j = 1
            # os.mkdir(os.path.join(data, person))
            for nfol in os.listdir(os.path.join(new_folder, person)):
              # if nfol[0] == 'n' or 'b' or 'c':

                    for normalfol in os.listdir(os.path.join(new_folder, person, nfol)):
                        count = 0
                        total = np.zeros((240, 320), np.uint8)
                        GEI = total
                        for file in os.listdir(os.path.join(new_folder, person, nfol, normalfol)):
                            filename= os.path.join(new_folder, person, nfol, normalfol, file)
                            image = cv2.imread(filename)  # Read the frame
                            # plt.imshow(image)
                            # plt.show()
                            height, width, ch = image.shape  # Height and width of the frames
                            # print(height,width)
                            (thresh, image) = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Turn the frame to one channel

                            # plt.imshow(image)
                            # plt.show()
                            x, X, y, Y = Border(image)  # Find the borders of the silhouette
                            # print(x,X,y,Y)
                            cropped = Crop(image, x, X, y, Y)  # Crop the silhouette
                            # plt.imshow(cropped)
                            # plt.show()

                            center = CenterOfMass(cropped)  # Find the center of mass of the silhouette

                            GEI = Add(GEI, cropped, center, total, count)  # Find the GEI with the latest frame
                            count+=1  # Add 1 to index of frames
                        x, X, y, Y = Border(GEI)
                        GEI = Crop(GEI, x, X, y, Y)  # Crop the GEI
                        plt.imsave(fname=os.path.join(data,person,nfol,normalfol+".png"),arr=GEI,cmap="gray")  # Write the image
                        print('sffsf')
                        j = j + 1



    print ("The GET is stored in "+data+"\nYou are ready to go")



CreateGEt(Making_GET=True)
