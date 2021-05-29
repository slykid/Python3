def addition(l, r):
    return l + r

def subtract(l, r):
    return l - r

def multiply(l, r):
    return l * r

def division(l, r):
    if l > r and r != 0:
        return l / r
    elif l < r and l != 0:
        return r / l