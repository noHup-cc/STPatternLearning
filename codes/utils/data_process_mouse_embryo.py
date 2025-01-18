#! /usr/bin/env python
import re
import math
import numpy as np
import pandas as pd
import random
import os
from os import path
import pickle
from configparser import ConfigParser
from typing import Dict, List, Tuple, Any, Union, Optional

# Constants
EMBRYO_DATA_FOLDER_EXTENSION = "_timeFused_blending"  # The extension of the folder that contains the klb file
EMBRYO_DATA_FILE_EXTENSION = "fusedStack.corrected.shifted.klb"  # The extension of the klb file

def get_embryo_data_path(embryo_name: str, config_path: Optional[str] = None) -> str:
    """Get the data folder path for a specific embryo."""
    cfg = ConfigParser()
    cfg.read(config_path)
    home_dir = cfg.get('embryo_data', 'embryo_data_dir')  # changed 2024/04/12
    embryo_dir = cfg.get('embryo', embryo_name)
    return os.path.join(home_dir, 'embryo_registration/data/idr0044/', embryo_dir)

def special_form(time_point: int) -> str:
    """Convert a time point number to a 6-digit string padded with leading zeros."""
    return str(time_point).zfill(6)

def get_embryo_slice_path(embryo_type: str, original_slice_time: int, original_slice_z: int, 
                         config_path: Optional[str] = None) -> str:
    """Get the path to a specific embryo slice image."""
    data_folder = get_embryo_data_path(embryo_type, config_path=config_path)
    data_folder = os.path.join(data_folder, '') if not data_folder.endswith('/') else data_folder
    
    process_folder = ''
    for fd in os.listdir(data_folder):
        if fd.endswith(EMBRYO_DATA_FOLDER_EXTENSION) and not fd.startswith('.'):
            process_folder = os.path.join(data_folder, fd)
            break
    
    time_str = special_form(original_slice_time)
    base_path = f"{process_folder.split('TM')[0]}TM{time_str}{EMBRYO_DATA_FOLDER_EXTENSION}"
    return os.path.join(base_path, f'klb_slice{time_str}', f'{time_str}z{original_slice_z}.jpg')

def manual_time_zgap_gradient(start_time: int, final_time: int, start_zgap: int, 
                            end_zgap: int, zgap_dict: Dict[str, Dict[int, int]], 
                            section_type: int, embr_type: str) -> Dict[str, Dict[int, int]]:
    """Calculate gradient for z-gap and populate dictionary with values."""
    gradient = (end_zgap - start_zgap) / (final_time - start_time)
    print('z gap gradient:', gradient)
    
    if gradient <= 0:
        print('manual_time_zgap_gradient() -> error 2')
        return zgap_dict
        
    def get_zgap_dict(section_number: str) -> Dict[str, Dict[int, int]]:
        zgap_dict[section_number] = {}
        cur_gap = start_zgap
        for time_p in range(start_time, final_time + 1):
            delta_time = time_p - start_time
            if math.floor(delta_time * gradient) != 0:
                cur_gap = start_zgap + math.floor(delta_time * gradient)
            zgap_dict[section_number][time_p] = cur_gap
        return zgap_dict
    
    section_map = {1: 'section1', 2: 'section2', 3: 'section3', 4: 'section4'}
    section_name = section_map.get(section_type)
    
    if not section_name:
        print('manual_time_zgap_gradient() -> error 1')
        return zgap_dict
        
    return get_zgap_dict(section_name)

def get_zslice_bound(embryo_type: str) -> Dict[int, List[int]]:
    """Get lower and upper bounds for z-slices across all time points."""
    def manual_time_gradient(start_time: int, start_time_bound: List[int],
                           final_time: int, final_time_bound: List[int]) -> Dict[int, List[int]]:
        bound_data_dict = {}
        increase_gradient = (final_time_bound[1] - start_time_bound[1]) / (final_time - start_time)
        decrease_gradient = (final_time_bound[0] - start_time_bound[0]) / (final_time - start_time)
        
        cur_l_bound, cur_u_bound = start_time_bound[0], start_time_bound[1]
        
        for time in range(start_time, final_time + 1):
            delta_time = time - start_time
            if math.floor(delta_time * increase_gradient) != 0:
                cur_u_bound = start_time_bound[1] + math.floor(delta_time * increase_gradient)
            if math.floor(delta_time * decrease_gradient) != 0:
                cur_l_bound = start_time_bound[0] + math.floor(delta_time * decrease_gradient)
            bound_data_dict[time] = [cur_l_bound, cur_u_bound]
        return bound_data_dict

    embryo_configs = {
        'A': {'times': [(5, [415, 535]), (100, [395, 565]), (200, [335, 625]),
                       (300, [235, 725]), (400, [155, 815]), (531, [85, 875])]},
        'B': {'times': [(0, [385, 525]), (100, [305, 595]), (200, [235, 665]),
                       (285, [165, 725])]},
        'C': {'times': [(0, [405, 555]), (100, [365, 595]), (200, [305, 655]),
                       (350, [195, 795])]},
        'D': {'times': [(0, [265, 455]), (100, [215, 505]), (200, [145, 565]),
                       (265, [105, 605])]}
    }
    
    if embryo_type not in embryo_configs:
        raise ValueError(f'Invalid embryo type: {embryo_type}')
    
    total_zslice_bound_dict = {}
    times = embryo_configs[embryo_type]['times']
    
    for i in range(len(times) - 1):
        start_time, start_bound = times[i]
        end_time, end_bound = times[i + 1]
        data_dict = manual_time_gradient(start_time, start_bound, end_time, end_bound)
        total_zslice_bound_dict.update(data_dict)
    
    return total_zslice_bound_dict

def get_z_gap_total_dict(embryo_type: str) -> Dict[str, Dict[int, int]]:
    """Get z gaps for each section across all time points in an embryo."""
    embryo_configs = {
        'A': {'start_time': 5, 'end_time': 531, 
              'gaps': [(1, 13, 37), (2, 25, 58), (3, 15, 81), (4, 10, 40)]},
        'B': {'start_time': 0, 'end_time': 285,
              'gaps': [(1, 12, 26), (2, 25, 46), (3, 15, 25), (4, 13, 23)]},
        'C': {'start_time': 0, 'end_time': 350,
              'gaps': [(1, 5, 16), (2, 10, 74), (3, 15, 31), (4, 5, 26)]},
        'D': {'start_time': 0, 'end_time': 265,
              'gaps': [(1, 13, 17), (2, 14, 24), (3, 13, 21), (4, 9, 17)]}
    }
    
    if embryo_type not in embryo_configs:
        print('get_z_gap_total_dict() -> error')
        return {}
    
    config = embryo_configs[embryo_type]
    total_zgap_dict = {}
    
    for section_num, start_gap, end_gap in config['gaps']:
        zgap_dict = manual_time_zgap_gradient(
            config['start_time'], config['end_time'], 
            start_gap, end_gap, {}, section_num, embryo_type
        )
        total_zgap_dict.update(zgap_dict)
    
    return total_zgap_dict

def section_division(data_zslice_bound: Dict[int, List[int]], time_point: int) -> Tuple[List[int], ...]:
    """Divide z slices into 4 sections at one time point."""
    lower, upper = data_zslice_bound[time_point]
    mid_point = math.floor((upper + lower) / 2)
    lower_mid = math.floor((mid_point + lower) / 2)
    upper_mid = math.floor((mid_point + upper) / 2)
    
    return (
        [lower, lower_mid],           # section_1
        [lower_mid, mid_point],       # section_2
        [mid_point, upper_mid],       # section_3
        [upper_mid, upper]            # section_4
    )

def get_zgap_coefficient(data_zslice_bound: Dict[int, List[int]], 
                        z_gap_dict: Dict[str, Dict[int, int]]) -> Tuple[List[float], ...]:
    """Get min and max z gap coefficients for each section."""
    coefficients = {f'section{i}': [] for i in range(1, 5)}
    
    for time in z_gap_dict['section1'].keys():
        sections = section_division(data_zslice_bound, time)
        
        for idx, section in enumerate(sections, 1):
            section_size = section[1] - section[0]
            z_gap = z_gap_dict[f'section{idx}'][time]
            coefficient = section_size / z_gap
            coefficients[f'section{idx}'].append(coefficient)
    
    return tuple([min(coeff), max(coeff)] for coeff in coefficients.values())

def iter_folder(folder: str, slice_dict: Dict[str, str]) -> Dict[str, str]:
    """Recursively iterate through folders to get time point and z slice numbers."""
    regex = re.compile(r'\d+')
    
    if os.path.isdir(folder):
        for file in os.listdir(folder):
            if not file.endswith("klb") and not file.startswith('.'):
                folder_file = os.path.join(folder, file)
                iter_folder(folder_file, slice_dict)
    else:
        tz = [int(x) for x in regex.findall(folder)]
        t, z = tz[-2], tz[-1]
        slice_dict[f't{t}z{z}'] = folder
    
    return slice_dict

def batch_klb_slicing(data_folder: str, slice_dict: Dict[str, str], 
                      folder_extension: str) -> Dict[str, str]:
    """Process multiple folders under the main data folder."""
    slice_final = {}
    data_folder = os.path.join(data_folder, '')
    
    for fd in os.listdir(data_folder):
        if fd.endswith(folder_extension):
            process_folder = os.path.join(data_folder, fd)
            tp = re.findall(r"\d+\.?\d*", process_folder)[-1]
            target_folder = os.path.join(process_folder, f"klb_slice{tp}")
            
            if os.path.exists(target_folder):
                slice_dict = iter_folder(target_folder, slice_dict)
                slice_final.update(slice_dict)
            else:
                print(f"batch_klb_slicing() -> error (slices folder at time point {tp} doesn't exist)")
    
    return slice_final

def get_z_slice_path_dict(data_folder: str, folder_extension: str) -> Dict[int, Dict[int, str]]:
    """Get dictionary of z slice paths organized by time and z value."""
    zslice_dict = batch_klb_slicing(data_folder, {}, folder_extension)
    regex = re.compile(r'\d+')
    
    # Extract time and z values
    time_points, z_vals = [], []
    for z_slice in zslice_dict:
        t, z = map(int, regex.findall(z_slice)[-2:])
        time_points.append(t)
        z_vals.append(z)
    
    # Organize by time and z value
    result = {}
    for time in range(min(time_points), max(time_points) + 1):
        z_val_dict = {
            z_val: zslice_dict[f't{time}z{z_val}']
            for z_val in range(min(z_vals), max(z_vals) + 1)
        }
        result[time] = z_val_dict
    
    return result

def divide_datasets(data_f: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Divide dataset into training, validation, and test sets."""
    training_data = data_f.sample(frac=0.7, random_state=25)
    remain_data = data_f.drop(training_data.index)
    validation_data = remain_data.sample(frac=1/3, random_state=25)
    test_data = remain_data.drop(validation_data.index)
    
    return training_data, validation_data, test_data

def get_features_of_embryo_slices(embryo_type: str, feature_type: Optional[str] = None,
                                 home_dir: Optional[str] = None,
                                 is_empty_slices_removed: bool = False) -> Tuple[List[Any], List[str]]:
    """Get features of embryo slices with metadata.
    
    Args:
        embryo_type: Type/ID of the embryo
        feature_type: Type of features to extract
        home_dir: Base directory path
        is_empty_slices_removed: Whether to exclude empty slices
    
    Returns:
        Tuple containing:
        - List of slice features with metadata
        - List of column headers
    """
    data_z_slices_bound_embryo = get_zslice_bound(embryo_type)
    
    # Load features
    features_path = os.path.join(
        home_dir or '', 'embryo_registration/data/samples/', 
        feature_type or '', f'embryo{embryo_type}',
        f'embryo{embryo_type}_{feature_type}_features.pickle'
    )
    total_slices_features_embryo = pickle.load(open(features_path, 'rb'))
    
    # Determine time and z bounds
    time_keys = list(data_z_slices_bound_embryo.keys() if is_empty_slices_removed 
                    else total_slices_features_embryo.keys())
    min_target_time, max_target_time = min(time_keys), max(time_keys)
    
    first_time_features = total_slices_features_embryo[min_target_time]
    min_target_z = first_time_features[0][0]
    max_target_z = first_time_features[-1][0]
    
    # Collect slice features
    target_embryo_slices = []
    for target_time in range(min_target_time, max_target_time + 1):
        if is_empty_slices_removed:
            l_bound = data_z_slices_bound_embryo[target_time][0]
            u_bound = data_z_slices_bound_embryo[target_time][1]
        else:
            l_bound = min_target_z
            u_bound = max_target_z
            
        for target_z_val in range(l_bound, u_bound + 1):
            # z value starts from 1
            feature_info = f"{target_time}z{target_z_val}"
            feature_vector = total_slices_features_embryo[target_time][target_z_val - 1][1]
            
            slice_info = [feature_info, target_time, target_z_val]
            slice_info.extend(feature_vector.squeeze().tolist())
            target_embryo_slices.append(slice_info)
    
    # Create headers for the features
    feature_dimension = 2048 if feature_type == 'resnet' else 1536
    headers = ['info', 'target_time', 'target_z']
    headers.extend(f'feature_{i}' for i in range(feature_dimension))
    
    return target_embryo_slices, headers