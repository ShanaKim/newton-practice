def df(f,xn):
    dfxn= (f(xn+0.0001)-f(xn))/0.0001
    return dfxn

def ddf(f,xn):
    ddfxn= (df(f,xn+0.0001)-df(f,xn))/0.0001
    return ddfxn
    
def newton(f,x0, epsilon):
    xn=x0
    fxn=f(xn)
    n=0
    while (abs(0- fxn))>epsilon:
        xn= xn - df(f,xn)/ddf(f,xn)
        fxn=f(xn)
        n= n+1
    print('After', n ,'iterations, the root is', xn)

function1= lambda x: x**2-4*x
approx= newton(function1, 3, 0.01)
print(approx)
