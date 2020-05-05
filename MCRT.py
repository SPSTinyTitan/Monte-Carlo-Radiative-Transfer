import cv2
import torch
import math
import time
import numpy as np

# Note: Run time scales proportionally with both the # of photons and the avg # of steps
# required for each photon (which changes as you adjust the region)
# For 2000 photons, outer radius of 20, inner radius of 7, bin size 10 - my PC does the
# calculations in ~2 mins

def draw(grid, side, max_counts):
    # Make a black image
    img = np.zeros((side, side, 3), np.uint8)
    for i in range(side):
        for j in range(side):
            # Pixel brightness scales with counts (wrt max counts in the grid), capping at 255
            img[i][j] = round(255 * (grid[i][j].item()/max_counts))
    img = cv2.resize(img, (300, 300))
    cv2.imshow("Ising Model", img)
    cv2.waitKey(0)

def step(posmatrix, mu_matrix, g, numphotons, exitradius, coreradius, flags, ctr, ctrmatrix):
    # Generate new directions for each photon
    tau_scatter = -torch.log(torch.ones(numphotons, 1) - torch.rand(numphotons, 1))
    # Todo - Ray tracing in spherical coordinates

    for i in range(numphotons):
        # step() only scatters a photon and moves it if it has flag==0
        if flags[i] == 0:
            temp = posmatrix.clone().detach()
            if g == 0:
                costheta = 1 - 2 * torch.rand(1)  # For isotropic scattering
            else:
                costheta = (1 / (2 * g)) * (1 + g ** 2 - ((1 - g ** 2) / (1 - g + 2 * g * torch.rand(1))) ** 2)  # For anisotropic scattering
            phi = 2 * math.pi * torch.rand(1)
            sintheta = torch.sqrt(1 - costheta ** 2)

            mu_matrix[i][0][0] = sintheta * torch.cos(phi)
            mu_matrix[i][0][1] = sintheta * torch.sin(phi)
            mu_matrix[i][0][2] = costheta

            # Only traverse the photon to its new position if its new position is NOT inside the core
            if ((temp[i][0][0] + mu_matrix[i][0][0] * tau_scatter[i][0])**2 + (temp[i][0][0] + mu_matrix[i][0][0] * tau_scatter[i][0]) ** 2 + (temp[i][0][0] + mu_matrix[i][0][0] * tau_scatter[i][0]) ** 2) >= coreradius:
                posmatrix[i][0][0] += mu_matrix[i][0][0] * tau_scatter[i][0]
                posmatrix[i][0][1] += mu_matrix[i][0][1] * tau_scatter[i][0]
                posmatrix[i][0][2] += mu_matrix[i][0][2] * tau_scatter[i][0]
            else:
                ctrmatrix[i] -= 1

            # Set a spherical boundary of the region of radius "exitradius"
            if posmatrix[i][0][0]**2 + posmatrix[i][0][1]**2 + posmatrix[i][0][2]**2 >= exitradius**2:
                # If posmatrix magnitude is outside radius, find pt it passed through on sphere's surface
                # and assign the pt to posmatrix
                # temp gives coords of point inside sphere, posmatrix gives coords of point outside
                # Math is done by parametrizing x,y,z (in terms of t) between pt1 and pt2, and solving for t
                A = (temp[i][0][0]-posmatrix[i][0][0])**2+(temp[i][0][1]-posmatrix[i][0][1])**2+(temp[i][0][2]-posmatrix[i][0][2])**2
                B = -2*(temp[i][0][0]*(temp[i][0][0]-posmatrix[i][0][0])+temp[i][0][1]*(temp[i][0][1]-posmatrix[i][0][1])+temp[i][0][2]*(temp[i][0][2]-posmatrix[i][0][2]))
                C = temp[i][0][0]**2+temp[i][0][1]**2+temp[i][0][2]**2-exitradius**2

                t1 = (-B + torch.sqrt(B ** 2 - 4 * A * C)) / (2 * A)
                t2 = (-B - torch.sqrt(B ** 2 - 4 * A * C)) / (2 * A)
                if t1 >= 0 and t1 <= 1:
                    posmatrix[i][0][0] = temp[i][0][0] * (1 - t1) + posmatrix[i][0][0] * t1
                    posmatrix[i][0][1] = temp[i][0][1] * (1 - t1) + posmatrix[i][0][1] * t1
                    posmatrix[i][0][2] = temp[i][0][2] * (1 - t1) + posmatrix[i][0][2] * t1
                else:
                    posmatrix[i][0][0] = temp[i][0][0] * (1 - t2) + posmatrix[i][0][0] * t2
                    posmatrix[i][0][1] = temp[i][0][1] * (1 - t2) + posmatrix[i][0][1] * t2
                    posmatrix[i][0][2] = temp[i][0][2] * (1 - t2) + posmatrix[i][0][2] * t2

                # i.e. if the photon leaves the region, stop scattering it
                flags[i] = 1
                ctrmatrix[i] += ctr

def main():
    numphotons = 2000
    # Todo: make density a radial function and add ray tracing
    rho = 1  # assuming uniform density for now
    g = 0  # phase function (values b/w 0 (isotropic) and 1 (forward-throwing scattering))
    # Note: based on how theta is defined, forward-throwing scattering would tend for photons to scatter in the z-direction
    exitradius = 20
    core_radius = 7

    # Define an image plane coincident with x = d + exitradius
    d = exitradius * 3 # relative to the sphere's equator
    bin_width = 10
    no_bins_per_side = 32

    # Tie into physics of cross section
    time.process_time()

    # Construct a grid of bins (photometers) to count the incident photons
    photometers = torch.zeros(no_bins_per_side, no_bins_per_side)

    ### Todo Force radial component of generated photons to be positive (if I have time)

    posmatrix = torch.zeros(numphotons, 1, 3)
    mu_matrix = torch.zeros(numphotons, 1, 3)
    flags = torch.zeros(numphotons)
    image_coords = torch.zeros(numphotons, 2) # yz-coords of photons as they hit CCD

    for i in range(numphotons):
        # Random emission from surface of radiative zone of radius rad_zone_radius
        costheta = 2 * torch.rand(1) - 1
        phi = 2 * math.pi * torch.rand(1)
        sintheta = torch.sqrt(1 - costheta ** 2)

        posmatrix[i][0][0] = core_radius * sintheta * torch.cos(phi)
        posmatrix[i][0][1] = core_radius * sintheta * torch.sin(phi)
        posmatrix[i][0][2] = core_radius * costheta

        # Radial emission
        mu_matrix[i][0][0] = sintheta * torch.cos(phi)
        mu_matrix[i][0][1] = sintheta * torch.sin(phi)
        mu_matrix[i][0][2] = costheta
        # where (mu_x)^2 + (mu_y)^2 + (mu_z)^2 = 1

    for i in range(numphotons):
        tau_scatter = -torch.log(1 - torch.rand(1))
        for j in range(3):
            posmatrix[i][0][j] += mu_matrix[i][0][j] * tau_scatter.item()
        if posmatrix[i][0][0] ** 2 + posmatrix[i][0][1] ** 2 + posmatrix[i][0][2] ** 2 >= exitradius ** 2:
            flags[i] = 1

    ones = torch.ones(numphotons)
    countermatrix = torch.zeros(numphotons)
    counter = 0

    # Until EVERY photon exits the region, invoke step
    while torch.equal(flags, ones) != True:
        counter += 1
        step(posmatrix, mu_matrix, g, numphotons, exitradius, core_radius, flags, counter, countermatrix)

    # print("\nFinal positions on sphere:")
    # for i in range(numphotons):
    #     print("Photon " + str(i+1) + ":",
    #           "(x,y,z) = (" + str(posmatrix[i][0][0].item()) + "," + str(posmatrix[i][0][1].item()) + "," + str(posmatrix[i][0][2].item()) + ")")

    # print("\nPositions on detector:")
    for i in range(numphotons):
        if mu_matrix[i][0][0] > 0:
            alpha = (d - posmatrix[i][0][0])/mu_matrix[i][0][0]
            image_coords[i][0] = posmatrix[i][0][1] + alpha * mu_matrix[i][0][1]
            image_coords[i][1] = posmatrix[i][0][2] + alpha * mu_matrix[i][0][2]
        #     print("Photon "+str(i+1)+":", "(y,z) = ("+str(image_coords[i][0].item())+","+str(image_coords[i][1].item())+")")
        # else:
        #     print("Photon "+str(i+1)+":", "Doesn't reach detector.")

    # print("\nMaximum number of steps:", counter)
    # print("Number of steps taken for each photon:")
    # for i in range(numphotons):
    #     print("Photon "+str(i+1)+":", int(countermatrix[i].item()))

    print("\nTotal steps (across all photons):", int(sum(countermatrix)))

    y = bin_width * no_bins_per_side / -2

    for j in range(no_bins_per_side):
        z = bin_width * no_bins_per_side / -2
        for k in range(no_bins_per_side):
            for i in range(numphotons):
                if y + bin_width > image_coords[i][0] > y:
                    if z + bin_width > image_coords[i][1] > z:
                        photometers[no_bins_per_side-k-1][j] += 1
            z += bin_width
        y += bin_width

    #print(photometers)
    print("Total counts:", sum(sum(photometers)))
    maxcounts = torch.max(photometers).item()

    stop = time.process_time()
    print("Time elapsed:", stop, "seconds")

    draw(photometers, no_bins_per_side, maxcounts)

main()
