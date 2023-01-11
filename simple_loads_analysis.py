"""
This code only considers point load as a simplified model for the bridge project
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio


def slide_rule_round(x, s=3):
    # return x
    """Slide-rule precision"""
    if x == 0:
        return 0
    is_neg = 1 if x >= 0 else -1
    x *= is_neg
    first_digit = int(str(x).replace(".", "").lstrip("0")[0])
    if first_digit == 1:
        s = 4
    factor = 10 ** math.floor(np.log10(x))
    x = str(round((x / factor) * 10 ** (s-1)))
    len_x = len(x)
    num_zeros_to_add = int(np.log10((10 ** (1-s) * factor)))
    if num_zeros_to_add > 0:
        return int(x + "0"*num_zeros_to_add) * is_neg
    elif num_zeros_to_add < 0:
        x = x[:num_zeros_to_add] + "." + x[num_zeros_to_add:]
        return float(x + "0" * (s - len_x)) * is_neg
    if len_x < s:
        return float(x + "." + "0" * (s - len_x)) * is_neg
    return int(x) * is_neg


def find_support_forces(point_loads_dict, right_support_location=1200):
    """
    Given a dictionary of point loads, find the support forces
    Args:
        point_loads_dict: dictionary of point loads, with location as key and force as value
        right_support_location: int value of the location of the right support, default 1200 for this project

    Returns: upward support forces at left and right supports
    """
    sum_force = 0
    sum_moments_about_left = 0
    for location in point_loads_dict:
        sum_force += point_loads_dict[location]
        sum_moments_about_left += location * point_loads_dict[location]
    right_support = sum_moments_about_left / right_support_location
    left_support = sum_force - right_support
    return left_support, right_support

def produce_sf_and_bm_lists(point_loads_dict, left_support_force, right_support_force, right_support_location=1200):
    """
    Given a dictionary of point loads, find the shear force and bending moment lists
    Args:
        point_loads_dict: dictionary of point loads, with location as key and force as value, downward force is positive
        left_support_force: upward force at left support
        right_support_force: upward force at right support
        right_support_location: int value of the location of the right support, default 1200 for this project

    Returns: shear force and bending moment lists
    """
    locations_from_left = np.array([0])
    shear_forces = np.array([0])
    current_shear_force = left_support_force
    bending_moments = np.array([0])
    current_bending_moment = 0
    previous_location = 0

    for location in sorted(point_loads_dict):
        dxs = np.linspace(0, location-previous_location, location-previous_location+1)
        interval_shear_forces = np.repeat(current_shear_force, location-previous_location+1)
        interval_bending_moments = np.repeat(current_bending_moment, location-previous_location+1) + dxs * current_shear_force

        current_bending_moment += (location - previous_location) * current_shear_force
        current_shear_force -= point_loads_dict[location]

        locations_from_left = np.concatenate((locations_from_left, dxs+previous_location))
        shear_forces = np.concatenate((shear_forces, interval_shear_forces))
        bending_moments = np.concatenate((bending_moments, interval_bending_moments))
        previous_location = location

    dxs = np.linspace(0, right_support_location-previous_location, right_support_location-previous_location+1)
    interval_shear_forces = np.repeat(current_shear_force, right_support_location-previous_location+1)
    interval_bending_moments = np.repeat(current_bending_moment, right_support_location-previous_location+1) + dxs * current_shear_force

    locations_from_left = np.concatenate((locations_from_left, dxs + previous_location))
    shear_forces = np.concatenate((shear_forces, interval_shear_forces))
    bending_moments = np.concatenate((bending_moments, interval_bending_moments))

    locations_from_left = np.concatenate((locations_from_left, [right_support_location]))
    shear_forces = np.concatenate((shear_forces, [right_support_force+current_shear_force]))
    bending_moments = np.concatenate((bending_moments, [0]))

    return locations_from_left, shear_forces, bending_moments


def plot_sfd_and_bmd(locations_from_left, shear_forces, bending_moments, right_support_location=1200,
                     x=None, length_unit="mm", force_unit="N",
                     filename="SFD + BMD", print_results=True, show_graph=True, save_graph=False):
    """
    given x/y values, plot them, save the graph, and return the key values (max BM/SF)
    Also display the max SF and BM
    :param length_unit: unit of length - to be displayed on graph
    :param force_unit: unit of force - to be displayed on graph
    :param maximum_y_min_on_bmd: plot a point on this plot so the y-axis will always at least include this point
    :param maximum_y_min_on_sfd: same
    :param minimum_y_max_on_bmd: plot a point on this plot so the y-axis will always at least include this point
    :param minimum_y_max_on_sfd: same as above
    :param save_graph: will the graph be saved as a png?
    :param show_graph: does this function show the graph on console?
    :param print_results: does this function print the max/min BM/SF
    :param key_points_x:
    :param sfd_key_values:
    :param bmd_key_values:
    :param bmd_full_curve_x:
    :param bmd_full_curve_y:
    :param filename:
    :return: max and location of: +sf, -sf, +bm, -bm
    """

    # Find the max SF and BM
    pos_max_sf = np.amax(shear_forces)
    pos_max_sf_indexes = np.where(shear_forces == pos_max_sf)[0]
    pos_max_sf_pos = int(locations_from_left[pos_max_sf_indexes[0]]) if locations_from_left[pos_max_sf_indexes[0]]==locations_from_left[pos_max_sf_indexes[-1]] else f"{int(locations_from_left[pos_max_sf_indexes[0]])} or {int(locations_from_left[pos_max_sf_indexes[-1]])}"
    pos_max_sf = slide_rule_round(pos_max_sf)

    pos_max_bm = np.amax(bending_moments)
    pos_max_bm_indexes = np.where(bending_moments == pos_max_bm)[0]
    pos_max_bm_pos = int(locations_from_left[pos_max_bm_indexes[0]]) if locations_from_left[pos_max_bm_indexes[0]]==locations_from_left[pos_max_bm_indexes[-1]] else f"{int(locations_from_left[pos_max_bm_indexes[0]])} or {int(locations_from_left[pos_max_bm_indexes[-1]])}"
    pos_max_bm = slide_rule_round(pos_max_bm)

    neg_max_sf = np.amin(shear_forces)
    neg_max_sf_indexes = np.where(shear_forces == neg_max_sf)[0]
    neg_max_sf_pos = int(locations_from_left[neg_max_sf_indexes[0]]) if locations_from_left[neg_max_sf_indexes[0]]==locations_from_left[neg_max_sf_indexes[-1]] else f"{int(locations_from_left[neg_max_sf_indexes[0]])} or {int(locations_from_left[neg_max_sf_indexes[-1]])}"
    neg_max_sf = slide_rule_round(neg_max_sf)

    neg_max_bm = np.amin(bending_moments)
    neg_max_bm_indexes = np.where(bending_moments == neg_max_bm)[0]
    neg_max_bm_pos = int(locations_from_left[neg_max_bm_indexes[0]]) if locations_from_left[neg_max_bm_indexes[0]]==locations_from_left[neg_max_bm_indexes[-1]] else f"{int(locations_from_left[neg_max_bm_indexes[0]])} or {int(locations_from_left[neg_max_bm_indexes[-1]])}"
    neg_max_bm = slide_rule_round(neg_max_bm)

    if show_graph or save_graph:
        graph, (sfd, bmd) = plt.subplots(2, figsize=(10, 12))

        # Plot the points and curves
        sfd.plot(locations_from_left, shear_forces, 'b-')
        sfd.plot([0, right_support_location], [0, 0], 'k-')

        bmd.plot(locations_from_left, bending_moments, "r-", )
        bmd.plot([0, right_support_location], [0, 0], 'k-')
        bmd.invert_yaxis()

        # Generate graph titles
        graph.suptitle(f'SFD and BMD {f"for x={str(x)} {length_unit}" if x is not None else ""} (Andy Version 3.0)')
        sfd.set_title(
            f"SFD\n(+) max {pos_max_sf} {force_unit} at {pos_max_sf_pos} {length_unit}\n(-) max {neg_max_sf} {force_unit} at {neg_max_sf_pos} {length_unit}")
        sfd.set(xlabel=f'Distance From Left Support ({length_unit})', ylabel=f'Shear Force ({force_unit})')
        bmd.set_title(
            f"BMD\n(+) max {pos_max_bm} {force_unit + length_unit} at {pos_max_bm_pos} {length_unit}\n(-) max {neg_max_bm} {force_unit + length_unit} at {neg_max_bm_pos} {length_unit}")
        bmd.set(xlabel=f'Distance From Left Support ({length_unit})',
                ylabel=f'Bending Moment ({force_unit + length_unit})')
        graph.tight_layout()

        if show_graph:
            bmd.grid()
            sfd.grid()
            plt.show()
        if save_graph:
            graph.savefig(filename)


    if print_results:
        print(f"(+) Max SF {pos_max_sf} {force_unit} at {pos_max_sf_pos} {length_unit} from left support")
        print(f"(-) Max SF {neg_max_sf} {force_unit} at {neg_max_sf_pos} {length_unit} from left support")

        print(f"(+) Max BM {pos_max_bm} {force_unit+length_unit} at {pos_max_bm_pos} {length_unit} from left support")
        print(f"(-) Max BM {neg_max_bm} {force_unit+length_unit} at {neg_max_bm_pos} {length_unit} from left support")

    return pos_max_sf, pos_max_sf_pos, neg_max_sf, neg_max_sf_pos, pos_max_bm, pos_max_bm_pos, neg_max_bm, neg_max_bm_pos


def generate_train_data(x, plot=True, print_results=True):
    """
    Units are in mm and N
    Args:
        x:

    Returns:

    """
    full_train_load = 400
    load_pos = np.array([52, 228, 392, 568, 732, 908]) + x

    load_dict = {}
    for pos in load_pos:
        load_dict[pos] = full_train_load/len(load_pos)

    left_support_force, right_support_force = find_support_forces(load_dict)
    locations_from_left, shear_forces, bending_moments = produce_sf_and_bm_lists(load_dict, left_support_force, right_support_force)
    if plot:
        plot_sfd_and_bmd(locations_from_left, shear_forces, bending_moments, x=x, show_graph=plot, save_graph=False, print_results=print_results)
    return locations_from_left, shear_forces, bending_moments




def print_sf_bm_at_x(x, locations_from_left, shear_forces, bending_moments, length_unit="mm", force_unit="N"):
    """
    Units are in mm and N
    Args:
        x:
        locations_from_left:
        shear_forces:
        bending_moments:

    Returns:

    """
    # Find the index of the first location from left that is greater than x
    index = np.where(locations_from_left == x)[0]
    # Use the index to find the shear force and bending moment at x
    sf = shear_forces[index]
    bm = bending_moments[index]
    return_sf = []
    for s in sf:
        if s not in return_sf:
            return_sf.append(s)
    return_bm = []
    for b in bm:
        if b not in return_bm:
            return_bm.append(b)

    if len(return_sf) == 1:
        print(f"SF at {x} {length_unit} from left support: {slide_rule_round(return_sf[0])} {force_unit}")
        return_sf = slide_rule_round(return_sf[0])
    else:
        print(f"SFs at {x} {length_unit} from left support: ", end="")
        for i in range(len(return_sf)-1):
            print(f"{slide_rule_round(return_sf[i])} {force_unit}, ", end="")
            return_sf[i] = slide_rule_round(return_sf[i])
        print(f"{slide_rule_round(return_sf[i+1])} {force_unit}")
        return_sf[i+1] = slide_rule_round(return_sf[i+1])

    if len(return_bm) == 1:
        print(f"BM at {x} {length_unit} from left support: {slide_rule_round(return_bm[0])} {force_unit+length_unit}")
        return_bm = slide_rule_round(return_bm[0])
    else:
        print(f"BMs at {x} {length_unit} from left support: ", end="")
        for i in range(len(return_bm)-1):
            print(f"{slide_rule_round(return_bm[i])} {force_unit+length_unit}, ", end="")
            return_bm[i] = slide_rule_round(return_bm[i])
        print(f"{slide_rule_round(return_bm[i+1])} {force_unit+length_unit}")
        return_bm[i+1] = slide_rule_round(return_bm[i+1])

    return return_sf, return_bm


def print_all_time_max_sf_bm(print_result=True, plot=True):
    global_locations_from_left = np.linspace(0, 1200, 1201, dtype=int)
    global_shear_forces = np.zeros(1201)
    global_bending_moments = np.zeros(1201)

    for x in range(0, 241):
        temp_locations_from_left, temp_shear_forces, temp_bending_moments = generate_train_data(x, plot=False, print_results=False)
        temp_locations_from_left = np.array(temp_locations_from_left, dtype=int)
        for i in range(len(temp_locations_from_left)):
            location = temp_locations_from_left[i]  # location is 0-1200
            if abs(temp_shear_forces[i]) > abs(global_shear_forces[location]):
                global_shear_forces[location] = abs(temp_shear_forces[i])
            if abs(temp_bending_moments[i]) > abs(global_bending_moments[location]):
                global_bending_moments[location] = temp_bending_moments[i]
    max_bm = max(global_bending_moments)
    # max_bm_location = locations_from_left[bending_moments.index(max_bm)]
    max_sf = max(global_shear_forces)
    # max_sf_location = locations_from_left[shear_forces.index(max_sf)]
    if print_result:
        print(f"Max SF: {max_sf} N")
        print(f"Max BM: {max_bm} Nmm")
    if plot:
        plot_sfd_and_bmd(global_locations_from_left, global_shear_forces, global_bending_moments, show_graph=True, save_graph=True, print_results=True, filename="../Generate BMD and SFD/all_time_max_sfd_and_bmd.png")
    return global_locations_from_left, global_shear_forces, global_bending_moments

if __name__ == "__main__":
    print_all_time_max_sf_bm(print_result=False, plot=True)
    # generate_train_data(90)
    # train_start_x = 120
    # force_check_x = 52
    #
    # locations_from_left, shear_forces, bending_moments = generate_train_data(train_start_x)
    # print_sf_bm_at_x(force_check_x, locations_from_left, shear_forces, bending_moments)

