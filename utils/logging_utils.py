import logging
import os
import time
from datetime import datetime
from utils.file_utils import FileType, ExperimentMeta, get_run_filename

def setup_logger(name, dataset, known_cls_ratio, labeled_ratio, log_file=None, timestamp=None):
    """Set up logger
    Args:
        name: Logger name
        dataset: Dataset name
        known_cls_ratio: Known class ratio
        labeled_ratio: Labeled data ratio
        log_file: Log file path, create new if None
        timestamp: Timestamp, create new if None
    """
    # Create new timestamp if not provided
    if timestamp is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Create experiment metadata
    exp_meta = ExperimentMeta(
        dataset=dataset,
        known_ratio=known_cls_ratio,
        labeled_ratio=labeled_ratio,
        seed=0,  # Can be passed as parameter later
        timestamp=timestamp
    )
    
    # Create new log file if not specified
    if log_file is None:
        log_file = get_run_filename(
            exp_meta=exp_meta,
            file_type=FileType.LOG,
            stage=name.lower(),  # Use lowercase stage name
            logger=None
        )
    
    # Set log format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Set file handler
    if name == 'pretrain':
        # Handler for pretrain stage
        file_handler = logging.FileHandler(log_file, mode='w')  # 'w' mode creates new file
    else:
        file_handler = logging.FileHandler(log_file, mode='a')  # 'a' mode appends to file
        with open(log_file, 'a') as f:
            f.write('\n' + '*' * 50 + '\n')
    
    file_handler.setFormatter(formatter)
    
    # Set console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
        
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_file
