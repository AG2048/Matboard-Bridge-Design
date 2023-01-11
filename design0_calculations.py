import numpy as np
import matplotlib.pyplot as plt
import simple_loads_analysis


"""
Table of contents:
1. Define key bridge dimensions
2. Merge key dimensions into every x-coord
3. Calculate cross-sectional properties (A, y-bar, I, Q)
4. Calculate Failure loads (SF or BM)
5. Display
"""

"""
Note - all "key changes" must include the beginning and end
"""
# Default length of bridge, may become longer if bridge is... 1270?
L = 1200
board_thickness = 1.27
"""
Top (width, thickness)
"""
top_board_key_changes = {
    0: (100, board_thickness),
    L: (100, board_thickness)
}
"""
additional thickness between side walls below top board (number_of_layers, )
width given by side walls, height given by side_walls_height
"""
top_board_additional_thickness_changes = {
    0: (0,),
    L: (0,)
}
"""
Sides - assume both sides are equal (height, thickness, gap_between_inner_walls_of_side_boards)
"""
side_boards_key_changes = {
    0: (75, board_thickness, 80-2*board_thickness),
    L: (75, board_thickness, 80-2*board_thickness)
}
"""
Bottom - assume lowest pos of bottom is at edge of the side board (width, thickness)
"""
bottom_boards_key_changes = {
    0: (80-2*board_thickness, board_thickness),
    L: (80-2*board_thickness, board_thickness)
}
"""
Flaps (thickness, length) length = folding inward outside of side wall
only enter one side, assume other side symmetrical
"""
flaps_key_changes = {
    0: (board_thickness, 5),
    L: (board_thickness, 5)
}
"""
Glue (total_width_of_glue_on_cross_section) - FOR FLAPS ONLY
"""
glue_key_changes = {
    0: (5+5+2*board_thickness,),
    L: (5+5+2*board_thickness,)
}
"""
Diaphragms (x_location) - assume covers whole side board (may overlap with bottom board but shouldn't affect calculation?)
must have one on each end.
"""
diaphragm_locations = [-15, 15, 400, 800, 1185, 1215]
diaphragm_flap_length = 5


"""
This section merges all input into lists usable by computer
Assume all dimensions changes uniformly across bridge, except for thickness
for thickness, at point of change, the larger one is used
"""
# one x-coord for every integer x pos
x_coords = np.linspace(0, L, L-0+1, dtype=int)
top_width = np.interp(x_coords, list(top_board_key_changes.keys()), [list(i) for i in zip(*top_board_key_changes.values())][0])
side_height = np.interp(x_coords, list(side_boards_key_changes.keys()), [list(i) for i in zip(*side_boards_key_changes.values())][0])
side_gap = np.interp(x_coords, list(side_boards_key_changes.keys()), [list(i) for i in zip(*side_boards_key_changes.values())][2])
bottom_width = np.interp(x_coords, list(bottom_boards_key_changes.keys()), [list(i) for i in zip(*bottom_boards_key_changes.values())][0])
flap_length = np.interp(x_coords, list(flaps_key_changes.keys()), [list(i) for i in zip(*flaps_key_changes.values())][1])
glue_width = np.interp(x_coords, list(glue_key_changes.keys()), [list(i) for i in zip(*glue_key_changes.values())][0])

# calculate thickness
top_thickness = [top_board_key_changes[0][1]]
top_additional_layers = [top_board_additional_thickness_changes[0][0]]
side_thickness = [side_boards_key_changes[0][1]]
bottom_thickness = [bottom_boards_key_changes[0][1]]
flap_thickness = [flaps_key_changes[0][0]]
# for any abrupt changes, current one becomes the larger one of current or new. but next one must be new
added_new_info = False
info_to_add = None
for x in x_coords[1:]:
    if x in top_board_key_changes:
        top_thickness.append(max(top_board_key_changes[x][1], top_thickness[-1]))
        added_new_info = True
        info_to_add = top_board_key_changes[x][1]
    else:
        if added_new_info:
            top_thickness.append(info_to_add)
            added_new_info = False
        else:
            top_thickness.append(top_thickness[-1])
added_new_info = False
info_to_add = None
for x in x_coords[1:]:
    if x in top_board_additional_thickness_changes:
        top_additional_layers.append(max(top_board_additional_thickness_changes[x][0], top_additional_layers[-1]))
        added_new_info = True
        info_to_add = top_board_additional_thickness_changes[x][0]
    else:
        if added_new_info:
            top_additional_layers.append(info_to_add)
            added_new_info = False
        else:
            top_additional_layers.append(top_additional_layers[-1])
added_new_info = False
info_to_add = None
for x in x_coords[1:]:
    if x in side_boards_key_changes:
        side_thickness.append(max(side_boards_key_changes[x][1], side_thickness[-1]))
        added_new_info = True
        info_to_add = side_boards_key_changes[x][1]
    else:
        if added_new_info:
            side_thickness.append(info_to_add)
            added_new_info = False
        else:
            side_thickness.append(side_thickness[-1])
added_new_info = False
info_to_add = None
for x in x_coords[1:]:
    if x in bottom_boards_key_changes:
        bottom_thickness.append(max(bottom_boards_key_changes[x][1], bottom_thickness[-1]))
        added_new_info = True
        info_to_add = bottom_boards_key_changes[x][1]
    else:
        if added_new_info:
            bottom_thickness.append(info_to_add)
            added_new_info = False
        else:
            bottom_thickness.append(bottom_thickness[-1])
added_new_info = False
info_to_add = None
for x in x_coords[1:]:
    if x in flaps_key_changes:
        flap_thickness.append(max(flaps_key_changes[x][0], flap_thickness[-1]))
        added_new_info = True
        info_to_add = flaps_key_changes[x][0]
    else:
        if added_new_info:
            flap_thickness.append(info_to_add)
            added_new_info = False
        else:
            flap_thickness.append(flap_thickness[-1])

top_thickness = np.array(top_thickness)
top_additional_layers = np.array(top_additional_layers)
side_thickness = np.array(side_thickness)
bottom_thickness = np.array(bottom_thickness)
flap_thickness = np.array(flap_thickness)

# Now every dimension is a numpy array of length L-0+1

# This stores for every x coord, how big of a "diaphragms free zone" it is in
gaps_between_diaphragms = np.zeros(len(x_coords))
last_diaphragm_index = 0
for diaphragm_index in range(1, len(diaphragm_locations)):
    last_diaphragm_location = diaphragm_locations[diaphragm_index-1]
    current_diaphragm_location = diaphragm_locations[diaphragm_index]
    distance = current_diaphragm_location - last_diaphragm_location
    for i in range(max(0, last_diaphragm_location), min(L+1, current_diaphragm_location+1)):
        gaps_between_diaphragms[i] = max(gaps_between_diaphragms[i], distance)


"""
Calculate Cross Sectional Areas
"""
areas = top_thickness * top_width + top_additional_layers*board_thickness*side_gap + 2 * side_thickness * side_height + bottom_thickness * bottom_width + 2 * flap_thickness * flap_length
print("cross_sec_areas", [i for i in areas])

"""
Calculate Centroidal Axis
"""
y_bars = (top_thickness * top_width * (side_height + top_thickness/2)
          + top_additional_layers*board_thickness * side_gap * (side_height-top_additional_layers*board_thickness/2)
          + 2 * side_thickness * side_height * (side_height/2)
          + bottom_thickness * bottom_width * (bottom_thickness/2)
          + 2 * flap_thickness * flap_length * (side_height - flap_thickness/2)
          ) / areas
print("y_bars", [i for i in y_bars])

# check if y-bar > side_height
if max(y_bars - (side_height - flap_thickness)) > 0:
    print("y_bar > side_height - flap_thickness, consider this issue. Failure loads may not be accurate")

"""
Calculate Moment of Inertia, I = I0 + Ad^2
"""
Is = top_width * top_thickness**3 / 12 + top_thickness * top_width * (side_height + top_thickness/2 - y_bars)**2 \
        + side_gap * (top_additional_layers*board_thickness)**3 / 12 + top_additional_layers*board_thickness * side_gap * (side_height-top_additional_layers*board_thickness/2 - y_bars)**2 \
        + 2 * side_thickness * side_height**3 / 12 + 2 * side_thickness * side_height * (side_height/2 - y_bars)**2 \
        + bottom_width * bottom_thickness**3 / 12 + bottom_thickness * bottom_width * (bottom_thickness/2 - y_bars)**2 \
        + 2 * flap_length * flap_thickness**3 / 12 + 2 * flap_thickness * flap_length * (side_height - flap_thickness/2 - y_bars)**2
print("Is", [i for i in Is])

"""
Calculate Shear Centre from bottom
"""
Q_centres = 2 * y_bars * side_thickness * y_bars/2 \
            + bottom_thickness * bottom_width * (y_bars - bottom_thickness/2) \
            + np.array([max(0, y_bars[x] - (side_height[x] - top_additional_layers[x]*board_thickness))**2 / 2 * side_gap[x] for x in x_coords])  # squared / 2 because it's h * h/2
print("Q_centres",  [i for i in Q_centres])

"""
Calculate Shear Glue from top
"""
Q_glues = top_thickness * top_width * (side_height + top_thickness/2 - y_bars)
print("Q_glues", [i for i in Q_glues])

"""
Calculate Shear Glue in every top_board additional layers
Actually, it's sufficient to find the one closest to centroid axis that's not same as glue flaps
This code is not necessary for design 0, but may be useful for future designs
"""
additional_layers_largest_Q = np.zeros(L-0+1)+0.000000000001  # set to small value to prevent divide by 0
for x in x_coords:
    if top_additional_layers[x] > 1:  # one layer shares Q with glue flaps
        num_additional_boards_closest_to_y_bar = min([i for i in range(top_additional_layers[x]+1)], key=lambda i: abs(side_height[x] - (i-1)*board_thickness - y_bars[x]))
        y_for_Q = side_height[x] - (num_additional_boards_closest_to_y_bar-1)*board_thickness
        # Get Q from bottom
        additional_layers_largest_Q[x] = 2 * y_for_Q * side_thickness[x] * (y_bars[x] - y_for_Q/2) \
                                         + bottom_thickness[x] * bottom_width[x] * (y_bars[x] - bottom_thickness[x]/2) \
                                         + (top_additional_layers[x] - num_additional_boards_closest_to_y_bar + 1) * board_thickness * side_gap[x] * (y_bars[x] - (y_for_Q-(top_additional_layers[x] - num_additional_boards_closest_to_y_bar + 1) * board_thickness/2))


"""
Failure Loads
"""
# Properties
E = 4000
sigma_tension = 30
sigma_compression = 6
tau_shear = 4
poisson_ratio = 0.2
glue_shear_strength = 2

# Tension yield (sigma = My/I -> M = sigma*I/y)
tension_yield_BMs = sigma_tension * Is / y_bars  # tension yielding always at bottom
print("tension_yield_BMs", [i for i in tension_yield_BMs])

# Compression yield (sigma = My/I -> M = sigma*I/(h-y))
compression_yield_BMs = sigma_compression * Is / (side_height+top_thickness - y_bars)  # compression yielding always at top
print("Compression yielding BM", [i for i in compression_yield_BMs])

# Centroid axis shear (tau = VQ/Ib -> V = tau*Ib/Q)
centroid_axis_shear_fail_Vs = tau_shear * Is * (2*side_thickness
                                                + np.array([side_gap[x] if y_bars[x] > (side_height[x] - top_additional_layers[x]*board_thickness) else 0 for x in x_coords])  # If centre board extends there, add it
                                                ) / Q_centres
print("Centroid axis shear fail V", [i for i in centroid_axis_shear_fail_Vs])
# Glue shear (tau = VQ/Ib -> V = tau*Ib/Q)
glue_shear_fail_Vs = glue_shear_strength * Is * (2*flap_length+2*side_thickness + (np.array([side_gap[x] if top_additional_layers[x] > 0 else 0 for x in x_coords]))) / Q_glues
print("Glue shear fail V", [i for i in glue_shear_fail_Vs])
# layer shear (tau = VQ/Ib -> V = tau*Ib/Q)
top_additional_layers_glue_fail_Vs = glue_shear_strength * Is * (2*side_thickness + np.array([side_gap[x] if top_additional_layers[x] > 0 else 999999 for x in x_coords])) / additional_layers_largest_Q  # additional_layers_largest_Q is TINY if no additional layers, b is huge if no additional layers -> to ensure it's not used as a fail case
print("additional_layers_glue_fail V", [i for i in top_additional_layers_glue_fail_Vs])
# Actual glue fail is minimum of at flap or at additional layers
limiting_glue_fail_Vs = np.minimum(glue_shear_fail_Vs, top_additional_layers_glue_fail_Vs)
print("limiting glue fail V", [i for i in limiting_glue_fail_Vs])

# top flange type 1 buckle, both side restrained (sigma = k pi^2 E / (12 (1-mu^2)) * (t/b)^2), k = 4. b is distance between midpoint of webs (gap + side_thickness)
top_flange_middle_buckle_fail_compressive_stress = (4 * np.pi**2 * E) / (12 * (1-poisson_ratio**2)) * ((top_thickness+top_additional_layers*board_thickness)/(side_gap+side_thickness))**2
print("top flange middle buckle stress", [i for i in top_flange_middle_buckle_fail_compressive_stress])
# top flange type 2 buckle, one side restrained (sigma = k pi^2 E / (12 (1-mu^2)) * (t/b)^2), k = 0.425. b is distance between webs to edge of top flange
# Only consider one side, cuz symmetrical
top_flange_edge_buckle_fail_compressive_stress = (0.425 * np.pi**2 * E) / (12 * (1-poisson_ratio**2)) * (top_thickness/((top_width - side_gap)/2 - side_thickness/2))**2
print("top flange edge buckle stress", [i for i in top_flange_edge_buckle_fail_compressive_stress])
# web type 3 buckle, 2 side restrained non-uniform stress. k = 6
# b is from centroidal axis to middle of glue flap
webs_buckle_fail_compressive_stress = (6 * np.pi**2 * E) / (12 * (1-poisson_ratio**2)) * (side_thickness/(side_height-flap_thickness/2-y_bars))**2
print("webs buckle stress", [i for i in webs_buckle_fail_compressive_stress])
# shear buckling of webs
webs_shear_buckling_stress = (5 * np.pi**2 * E) / (12 * (1-poisson_ratio**2)) * ((side_thickness/gaps_between_diaphragms)**2 + (side_thickness/(side_height-flap_thickness/2-bottom_thickness/2))**2)
print("webs shear buckling stress", [i for i in webs_shear_buckling_stress])

# Flexural buckling bending moments: sigma = My/I -> M = sigma*I/(h-y)
top_flange_middle_buckle_BM = top_flange_middle_buckle_fail_compressive_stress * Is / (side_height+top_thickness - y_bars)
top_flange_edge_buckle_BM = top_flange_edge_buckle_fail_compressive_stress * Is / (side_height+top_thickness - y_bars)
webs_buckle_BM = webs_buckle_fail_compressive_stress * Is / (side_height - y_bars)  # max flexural stress on TOP of side board
print("top flange middle buckle BM", [i for i in top_flange_middle_buckle_BM])
print("top flange edge buckle BM", [i for i in top_flange_edge_buckle_BM])
print("webs buckle BM", [i for i in webs_buckle_BM])

# Shear buckling SFs: tau = VQ/Ib -> V = tau*Ib/Q (use midpoint Q as it's the largest)
webs_shear_buckling_SF = webs_shear_buckling_stress * Is * (2*side_thickness + np.array([side_gap[x] if y_bars[x] > (side_height[x] - top_additional_layers[x]*board_thickness) else 0 for x in x_coords])) / Q_centres
print("webs shear buckling SF", [i for i in webs_shear_buckling_SF])


# PLOTTING


max_load_xs, max_load_sfs, max_load_bms = simple_loads_analysis.print_all_time_max_sf_bm(print_result=False, plot=False)

"""FOS"""
# FOS for SF
board_shear_fos = min(centroid_axis_shear_fail_Vs / max_load_sfs)
glue_failure_fos = min(limiting_glue_fail_Vs / max_load_sfs)
shear_buckling_fos = min(webs_shear_buckling_SF / max_load_sfs)
# FOS for BM
tension_fos = min(tension_yield_BMs / max_load_bms)
compression_fos = min(compression_yield_BMs / max_load_bms)
top_flange_middle_buckle_fos = min(top_flange_middle_buckle_BM / max_load_bms)
top_flange_edge_buckle_fos = min(top_flange_edge_buckle_BM / max_load_bms)
webs_buckle_fos = min(webs_buckle_BM / max_load_bms)

board_shear_fos_location = min(x_coords, key=lambda x: centroid_axis_shear_fail_Vs[x] / max_load_sfs[x])
glue_failure_fos_location = min(x_coords, key=lambda x: limiting_glue_fail_Vs[x] / max_load_sfs[x])
shear_buckling_fos_location = min(x_coords, key=lambda x: webs_shear_buckling_SF[x] / max_load_sfs[x])
tension_fos_location = min(x_coords, key=lambda x: tension_yield_BMs[x] / max_load_bms[x])
compression_fos_location = min(x_coords, key=lambda x: compression_yield_BMs[x] / max_load_bms[x])
top_flange_middle_buckle_fos_location = min(x_coords, key=lambda x: top_flange_middle_buckle_BM[x] / max_load_bms[x])
top_flange_edge_buckle_fos_location = min(x_coords, key=lambda x: top_flange_edge_buckle_BM[x] / max_load_bms[x])
webs_buckle_fos_location = min(x_coords, key=lambda x: webs_buckle_BM[x] / max_load_bms[x])

print("board shear FOS", board_shear_fos, "at", board_shear_fos_location)
print("glue failure FOS", glue_failure_fos, "at", glue_failure_fos_location)
print("shear buckling FOS", shear_buckling_fos, "at", shear_buckling_fos_location)
print("tension FOS", tension_fos, "at", tension_fos_location)
print("compression FOS", compression_fos, "at", compression_fos_location)
print("top flange middle buckle FOS", top_flange_middle_buckle_fos, "at", top_flange_middle_buckle_fos_location)
print("top flange edge buckle FOS", top_flange_edge_buckle_fos, "at", top_flange_edge_buckle_fos_location)
print("webs buckle FOS", webs_buckle_fos, "at", webs_buckle_fos_location)

print("Minimum FOS:", min(board_shear_fos, glue_failure_fos, shear_buckling_fos, tension_fos, compression_fos, top_flange_middle_buckle_fos, top_flange_edge_buckle_fos, webs_buckle_fos))
print("Max Expected Train Load (N):", 400*min(board_shear_fos, glue_failure_fos, shear_buckling_fos, tension_fos, compression_fos, top_flange_middle_buckle_fos, top_flange_edge_buckle_fos, webs_buckle_fos))

graph, ((sfd1, sfd2, sfd3), (bmd1, bmd2, bmd3)) = plt.subplots(2, 3, figsize=(25, 10))

# Plot the points and curves
sfd1.plot(x_coords, max_load_sfs, "r-", label="Max Shear Force Envelope")
sfd1.plot(x_coords, centroid_axis_shear_fail_Vs, "b-", label="Centroid Shear failure")
sfd1.set_ylim(bottom=-24)
sfd1.set_title("Centroid Shear Failure")
sfd1.set_xlabel("Length Along Bridge (mm)")
sfd1.set_ylabel("Shear Force (N)")
sfd1.axhline(y=0, color='k')

sfd2.plot(x_coords, max_load_sfs, "r-", label="Max Shear Force Envelope")
sfd2.plot(x_coords, limiting_glue_fail_Vs, "b-", label="Glue Failure")
sfd2.set_ylim(bottom=-24)
sfd2.set_title("Glue Failure")
sfd2.set_xlabel("Length Along Bridge (mm)")
sfd2.set_ylabel("Shear Force (N)")
sfd2.axhline(y=0, color='k')

sfd3.plot(x_coords, max_load_sfs, "r-", label="Max Shear Force Envelope")
sfd3.plot(x_coords, webs_shear_buckling_SF, "b-", label="Webs Shear Buckling")
sfd3.set_ylim(bottom=-24)
sfd3.set_title("Webs Shear Buckling")
sfd3.set_xlabel("Length Along Bridge (mm)")
sfd3.set_ylabel("Shear Force (N)")
sfd3.axhline(y=0, color='k')

bmd1.plot(x_coords, max_load_bms, "r-", label="Max Bending Moment Envelope")
bmd1.plot(x_coords, tension_yield_BMs, "b-", label="Tension Yield")
bmd1.plot(x_coords, compression_yield_BMs, "g-", label="Compression Yield")
bmd1.set_ylim(bottom=-120000)
bmd1.set_title("Flexural Stress Failure")
bmd1.set_xlabel("Length Along Bridge (mm)")
bmd1.set_ylabel("Bending Moment (Nmm)")
bmd1.axhline(y=0, color='k')

bmd2.plot(x_coords, max_load_bms, "r-", label="Max Bending Moment Envelope")
bmd2.plot(x_coords, top_flange_middle_buckle_BM, "b-", label="Top Flange Middle Buckle")
bmd2.plot(x_coords, top_flange_edge_buckle_BM, "g-", label="Top Flange Edge Buckle")
bmd2.set_ylim(bottom=-120000)
bmd2.set_title("Top Flange Buckling")
bmd2.set_xlabel("Length Along Bridge (mm)")
bmd2.set_ylabel("Bending Moment (Nmm)")
bmd2.axhline(y=0, color='k')

bmd3.plot(x_coords, max_load_bms, "r-", label="Max Bending Moment Envelope")
bmd3.plot(x_coords, webs_buckle_BM, "b-", label="Webs Buckle")
bmd3.set_ylim(bottom=-100000)
bmd3.set_title("Webs Buckling")
bmd3.set_xlabel("Length Along Bridge (mm)")
bmd3.set_ylabel("Bending Moment (Nmm)")
bmd3.axhline(y=0, color='k')

sfd1.grid()
sfd2.grid()
sfd3.grid()
bmd1.grid()
bmd2.grid()
bmd3.grid()
sfd1.legend(loc="center")
sfd2.legend(loc="center")
sfd3.legend(loc="center")
bmd1.legend(loc="upper right")
bmd2.legend(loc="upper right")
bmd3.legend(loc="upper right")

bmd1.invert_yaxis()
bmd2.invert_yaxis()
bmd3.invert_yaxis()

plt.suptitle("Bridge Design 0 Failure Graphs")
plt.savefig("Design 0.png")
plt.show()

"""Followings are just added pieces of the code to show the same process work done by hand calculations"""
print("Cross Sectional Area", [i for i in areas])
print("Second Moment of Area", [i for i in Is])
print("Q at centroid", [i for i in Q_centres])
print("Q at glue", [i for i in Q_glues])
print("Centroid Axis y", [i for i in y_bars])
print("Compressive Flexural Stress At Top", [i for i in max_load_bms*(side_height+top_thickness - y_bars)/Is])
print("\tMax", max([i for i in max_load_bms*(side_height+top_thickness - y_bars)/Is]))
print("Tensile Flexural Stress At Bottom", [i for i in max_load_bms*y_bars/Is])
print("\tMax", max([i for i in max_load_bms*y_bars/Is]))
print("Shear Stress At Centroid", [i for i in max_load_sfs*Q_centres/Is/(1.27*2)])
print("\tMax", max([i for i in max_load_sfs*Q_centres/Is/(1.27*2)]))
print("Shear Stress At Glue", [i for i in max_load_sfs*Q_glues/Is/(1.27*2+10)])
print("\tMax", max([i for i in max_load_sfs*Q_glues/Is/(1.27*2+10)]))
print("Top flange middle buckling stress", [i for i in top_flange_middle_buckle_fail_compressive_stress])
print("\tMin", min([i for i in top_flange_middle_buckle_fail_compressive_stress]))
print("Top flange edge buckling stress", [i for i in top_flange_edge_buckle_fail_compressive_stress])
print("\tMin", min([i for i in top_flange_edge_buckle_fail_compressive_stress]))
print("Webs buckling stress", [i for i in webs_buckle_fail_compressive_stress])
print("\tMin", min([i for i in webs_buckle_fail_compressive_stress]))
print("Compressive flexural stress on webs", [i for i in max_load_bms*(side_height - y_bars)/Is])
print("\tMax", max([i for i in max_load_bms*(side_height - y_bars)/Is]))
print("Shear Buckling Stress on Webs", [i for i in webs_shear_buckling_stress])
print("\tMin - near the edges: ", min([i for i in webs_shear_buckling_stress[:150]]))



