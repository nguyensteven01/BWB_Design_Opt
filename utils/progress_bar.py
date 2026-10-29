# Print iterations progress
import time

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))  # This is the text displayed at end of bar
    filledLength = int(length * iteration // total)   # This is the length of the bar that is filled.
                                                      # iteration // total gives percent completion, then multiply by length of completed bar
    bar = fill * filledLength + '-' * (length - filledLength)   # This is the bar itself, a bunch of blocks and dashes
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

if __name__ == '__main__':
    for i in range(10):
        printProgressBar(i, 9, prefix = 'Progress:', suffix = 'Complete', length = 50)
        time.sleep(0.1)

