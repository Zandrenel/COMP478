import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import math
import time, sys, os

def nonlocal_denoise(image, h=10,m=7,deviation=.04,search_space=21,file_name="nonlocal.png"):
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
    

    
    Z = lambda euclideans : sum([ math.e**(-1*(euc)/(h**2)) for euc in euclideans ])
    

    
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
    N = [np.array(pImg[i-mp:i+mp,j-mp:j+mp]).flatten()
         for i in range(mp,pImg.shape[0]-mp)
         for j in range(mp,pImg.shape[1]-mp)]

    fImg = image.flatten()
    l = len(fImg)
    indexlist = [ i for i in range(l) ]
    #simple iteration calculatr for reducing operations done in the loop
    i0 = 0

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):

            ctime = round(time.time()-startTime,2)
            if ctime/60 > 1:
                ctime = "{}:{}".format(round(ctime/60),round(ctime%60))


            sys.stdout.write("{}/{}, {}% {}\r".format(i0,l,math.floor(100*i0/l),ctime))
            sys.stdout.flush()
            


            # finds all pixel indexes of the search window
            swindow = []
            win = round(search_space/2)
            # bounds [lb, rb, ub, db]
            bounds = [i-win, i+win, j-win, j+win]
            if bounds[0] < 0:
                bounds[0] = 0
            if bounds[1] > image.shape[0]:
                bounds[1] = image.shape[0]
            if bounds[2] < 0:
                bounds[2] = 0
            if bounds[3] > image.shape[1]:
                bounds[3] = image.shape[0]
            for windex in range(bounds[2],bounds[3]):
                swindow += indexlist[(windex*image.shape[0]+bounds[0]):(windex*image.shape[0]+bounds[1])]
            
            
                
            # create vector of euclidean distances for all j
            euclideans = [euclidean(N[i0],N[j0])+(2*deviation**2) for j0 in swindow]

            
            # calculate all Z(i)'s            
            Zi = Z(euclideans)
            
            if Zi < 0:
                print(i0)
                print("swindow",swindow)
                if len(swindow) == 0:
                    print()
                    print(bounds)
                    print(indexlist[(windex*image.shape[0]+bounds[0]):(windex*image.shape[0]+bounds[1])])
                    print()
                print(N[swindow[i0]],N[swindow[0]])
                print(euclideans)
                
            # w(i,j)
            wi = [ wij(Zi, eucj) for eucj in euclideans ]
            
            
            newImg[i,j] = np.sum([ max(wi)*fImg[j0] for j0 in range(len(swindow)) ])
            #newImg[i,j] = np.sum([ wi[j0]*fImg[j0] for j0 in range(len(swindow)) ])
            
            #increment 
            i0 += 1
    print("It took {} minutes.".format(round(((time.time()-startTime)/60),2)))

    with open("out.txt",'w') as f:
        for line in newImg:
            f.write(str(line))

    fig = plt.figure()
    plt.title('Original')
    plt.xticks([])
    plt.yticks([])

    plt.imshow(image,cmap='gray', vmin = 0, vmax = 255)
    plt.show()
    fig.savefig("Original{}".format(image.shape)+file_name)

            
    fig = plt.figure()
    plt.title('Nonlocal means for denoising')
    plt.xticks([])
    plt.yticks([])

    plt.imshow(newImg,cmap='gray', vmin = 0, vmax = 255)
    plt.show()
    fig.savefig("{}".format(image.shape)+file_name)

    fig = plt.figure()
    plt.title('Nonlocal means for denoising')
    plt.xticks([])
    plt.yticks([])

    plt.imshow(image-newImg,cmap='gray', vmin = 0, vmax = 255)
    plt.show()
    fig.savefig("2"+"{}".format(image.shape)+file_name)

    
    fig = plt.figure()
    plt.title('Nonlocal means for denoising')
    plt.xticks([])
    plt.yticks([])

    plt.imshow(image+newImg,cmap='gray', vmin = 0, vmax = 255)
    plt.show()
    fig.savefig("2_inv"+"{}".format(image.shape)+file_name)

    
    fig = plt.figure()
    plt.title('Nonlocal means for denoising')
    plt.xticks([])
    plt.yticks([])

    plt.imshow(image+newImg,cmap='gray', vmin = 0, vmax = 255)
    plt.show()
    fig.savefig("3"+"{}".format(image.shape)+file_name)

    fig = plt.figure()
    plt.title('Nonlocal means for denoising')
    plt.xticks([])
    plt.yticks([])

    plt.imshow(newImg-image,cmap='gray', vmin = 0, vmax = 255)
    plt.show()
    fig.savefig("4"+"{}".format(image.shape)+file_name)

    

if __name__ == '__main__':

    tools = cv.imread('tools_noisy.png', 0)
    
    eats = cv.imread('eats.jpg', 0)
    eats = cv.resize(eats,(260,280))
    gauss = np.random.normal(0,1,eats.size)
    gauss = gauss.reshape(eats.shape[0],eats.shape[1]).astype('uint8')
    noisey_eats = eats + eats * gauss
    
    fish = cv.imread('playfish.png', 0)
    fish = cv.resize(fish,(180,260))
    gauss = np.random.normal(0,1,fish.size)
    gauss = gauss.reshape(fish.shape[0],fish.shape[1]).astype('uint8')
    noisey_fish = fish + fish * gauss

    stare = cv.imread('stare.png', 0)
    #stare = cv.resize(stare,(176,240))
    gauss = np.random.normal(0,1,stare.size)
    gauss = gauss.reshape(stare.shape[0],stare.shape[1]).astype('uint8')
    noisey_stare = stare + stare * gauss

    
    
    #nonlocal_denoise(tools,m=7,search_space=14,h=1,deviation=.6,file_name='tools_nonlocal_denoise.png')
    
    # with smaller frame
    #nonlocal_denoise(tools,m=7,search_space=14,deviation=29,file_name='tools_nl_lowdev.png')
    #nonlocal_denoise(tools,m=5,search_space=14,h=5,file_name='tools_nl_lowh.png')
    #nonlocal_denoise(noisey_eats,m=5,search_space=14,h=14,deviation=.6,file_name='eats_nl_lowh.png')
    #nonlocal_denoise(noisey_fish,m=5,search_space=14,h=14,deviation=.6,file_name='fish_nl_lowh.png')
    nonlocal_denoise(noisey_stare,m=5,search_space=29,h=14,deviation=.6,file_name='stare_nl.png')

