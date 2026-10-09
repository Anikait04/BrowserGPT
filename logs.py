import logging
import os
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

load_dotenv()


# Function to create IST timezone
def ist_time(*args):
    ist = timezone(timedelta(hours=5, minutes=30))  # IST is UTC+5:30
    return datetime.now(ist).timetuple()


def log_separator(section_name=None, char="*", line_length=50, spacer_lines=2):
    """
    Logs a visually clear separator to enhance readability.

    Args:
        section_name (str, optional): A title to include in the separator.
        char (str, optional): The character to use for the separator line. Default is '*'.
        line_length (int, optional): Length of the separator line. Default is 50.
        spacer_lines (int, optional): Number of blank lines before and after the section. Default is 2.
    """
    blank_line = " "
    separator_line = char * line_length

    # Add blank lines before the separator
    for _ in range(spacer_lines):
        logger.info(blank_line)

    # Add the separator and section name
    logger.info(separator_line)
    if section_name:
        logger.info(f"{section_name.center(line_length)}")
        logger.info(separator_line)

    # Add blank lines after the separator
    for _ in range(spacer_lines):
        logger.info(blank_line)


def setup_logger():
    # Allow base directory to be set via an environment variable
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.getenv("LOG_DIR"))
    if not os.path.exists(base_dir):
        try:
            os.makedirs(base_dir)
        except OSError as e:
            print(f"Error creating log directory: {e}")
            return None

    # Generate monthly folder name
    month_name = datetime.now().strftime('%b')  # Get the abbreviated month name (e.g., Nov, Dec)
    year = datetime.now().strftime('%Y')  # Get the current year
    month_folder = f"{month_name}_{year}"  # Format folder name as "Nov_2024"
    month_path = os.path.join(base_dir, month_folder)  # Combine base directory and month folder path

    if not os.path.exists(month_path):
        try:
            os.makedirs(month_path)
        except OSError as e:
            print(f"Error creating month directory: {e}")
            return None

    # Generate log file name based on the date
    log_file = os.path.join(month_path, f"{datetime.now().strftime('%d-%m-%Y')}.log")

    # Check environment variable to set the log level
    debug_mode = os.getenv("DEBUG", "False").lower() == "true"
    log_level = logging.DEBUG if debug_mode else logging.INFO

    # Setup logger
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="a"),  # Append to the same file for the day
            logging.StreamHandler(),  # Log to console as well
        ],
    )

    # Configure formatter to use IST time
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s")
    formatter.converter = ist_time  # Set converter to IST

    # Apply formatter to handlers
    for handler in logging.getLogger().handlers:
        handler.setFormatter(formatter)

    return logging.getLogger()


# Setup centralized logger
logger = setup_logger()
 