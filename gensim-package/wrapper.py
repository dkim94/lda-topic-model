from datetime import datetime

def timed(func):
    def wrapper(*args):
        start = datetime.now()
        func(*args)
        end = datetime.now()
        print(f"Elapsed Time for model {args[1]} {args[2]} {args[3]} {args[4]}: {end-start}")
    return wrapper
    