import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import math
import time

def nonlocal_denoise(image, h=2,m=3,file_name="nonlocal.png"):
    startTime = time.time()
    
    # h is a weightparamter
    # m is the kernel size
    mp = round((m-1)/2)
    
    # Steps
    """
    - calculate an array of neighborhoods flattened for every pixel and centered at the origin pixel
    - for each pixel 
      - 1.
        - create an array of the differemce of pixels's vector and other pixel's value
        - take the euclidian distance squared of that vector
      - 2. find the normalizing constant with it
      - 3. plug it into final equation for the weight
      - 4. repeat and record each weight in a vector for each pixel
      - 5. apply the best weight to current pixel
    
    """

               
    # as defined by the report is detailed as
    # ||v(Ni-Nj)||_2^2 or expanded as \sum
    euclidean = lambda Ni, Nj :sum([Nij**2 for Nij in (Ni-Nj)])

    Z = lambda euclideans : sum([ math.e**(-1*(euc)/h**2) for euc in euclideans ])

    wij = lambda Zi, eucij : ((1/Zi)*math.e**(-1*((eucij)/h**2)))
    
    # make the paddedd image
    pImg = np.zeros((image.shape[0]+m-1,image.shape[1]+m-1))
    # set center of image
    pImg[mp:image.shape[0]+mp,mp:image.shape[1]+mp] = image
    # set left
    pImg[:mp,mp:image.shape[1]+mp] = image[image.shape[0]-mp:,:]
    # set right
    pImg[image.shape[0]+mp:,mp:image.shape[1]+mp] = image[:mp,:]
    # set top
    pImg[mp:image.shape[0]+mp,:mp] = image[:,image.shape[1]-mp:]
    # set bottom
    pImg[mp:image.shape[0]+mp,image.shape[1]+mp:] = image[:,:mp]

    newImg = np.copy(image)
    
    # Vector of neighborhoods
    N = [np.array(pImg[i-mp:i+mp,j-mp:j+mp])
         for i in range(mp,pImg.shape[0]-mp)
         for j in range(mp,pImg.shape[1]-mp)]

    fImg = image.flatten()
    l = len(fImg)
    #simple iteration calculatr for reducing operations done in the loop
    i0 = 0
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            print("{}/{}".format(i0,l))
            # create vector of euclidean distances for all j
            euclideans = [euclidean(N[i0],N[j0]) for j0 in range(len(N))]

            # calculate all Z(i)'s            
            Zi = Z(euclideans)

            # w(i,j)
            wi = [ wij(Zi, eucj) for eucj in euclideans ]

            
            newImg[i,j] = np.sum([ wi[j0]*fImg[j0] for j0 in range(len(fImg)) ])
            
            #increment 
            i0 += 1

    print("It took {} minutes.".format((time.time()-startTime)/60)
            
    fig = plt.figure()
    plt.title('Nonlocal means for denoising')
    plt.xticks([])
    plt.yticks([])

    plt.imshow(newImg)
    plt.show()
    fig.savefig(file_name)

    

if __name__ == '__main__':

    tools = cv.imread('tools_noisy.png', 0)
      
    nonlocal_denoise(tools,file_name='tools_nonlocal_denoise.png')

    
