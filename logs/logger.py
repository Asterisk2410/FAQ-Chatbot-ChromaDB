import logging
import os

# Define the directory where logs will be stored
log_directory = "logs/logger_files"
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

# Configure the logging
log_file = os.path.join(log_directory, 'chatbot.log')

logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()  # Also output to console
    ]
)

def get_logger(name):
    """Create and return a logger with the specified name."""
    return logging.getLogger(name)
