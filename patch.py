def should_include_large_file(file_path, file_size, max_file_size, test_mode=False):
    """Check if large file should be included based on importance"""
    file_type = get_file_type(file_path)
    return is_important_implementation_file(file_path, file_type) and file_size < max_file_size * 2
