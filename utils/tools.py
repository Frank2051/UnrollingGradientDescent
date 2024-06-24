def middle_percent(data, x):
    """
    Sort and retrieves the 100-x percent of floats from the middle of a list.
    """
    sorted_data = sorted(data)
    exclude_count = int(len(sorted_data) * (x / 200)) 
    start_index = exclude_count
    end_index = len(sorted_data) - exclude_count
    middle_percent = sorted_data[start_index:end_index]
    return middle_percent

def middle_exclude(data, x):
    """
    Retrieves the 100-x percent of floats from the middle of a list.
    """
    exclude_count = int(len(data) * (x / 200)) 
    start_index = exclude_count
    end_index = len(data) - exclude_count
    middle_percent = data[start_index:end_index]
    return middle_percent
