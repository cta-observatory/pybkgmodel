import datetime

def timestamp(text):
    """
    This function returns the specified text with the prefix of the current date

    Parameters
    ----------
    text: str

    Returns
    -------
    None

    """

    date_str = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    return f"{date_str:s}: {text:s}"


def message(text):
    print(timestamp(text))
