
upTo = 200

prime_numbers = []
numIsPrime = []
count = 0
num = 2

def is_prime(num):
    if num < 2:
        return False
    for i in range(2, int(num ** 0.5) + 1):
        if num % i == 0:
            return False
    return True

while num <= upTo:
    if is_prime(num):
        prime_numbers.append(num)
        numIsPrime.append(True)
    else:
        numIsPrime.append(False)
    num += 1

primes0 = []
chosens = []

primeProb = 0
predictedPrimeProb = 1
predictedNotPrimeProb = 0
ifPrimePredictPrimeProb = 1
ifPrimePredictNotPrimeProb = 0
numPrimes = 0
lastPrime = 1

d1 = False

for step in range(2, upTo):
    i = step - 1
    quanto = 1 / step

    effectivePrimeProb = numPrimes / (step)
    effectiveIfPrimeProb = (numPrimes+1) / (step)

    primeStamp = quanto
    primeStamp *= predictedPrimeProb

    ifPrimePredictPrimeProb = predictedPrimeProb
    ifPrimePredictPrimeProb -= primeStamp
    ifPrimePredictNotPrimeProb = 1 - ifPrimePredictPrimeProb

    reallyPrime = numIsPrime[i-1]
    isPrime = False
    chosen = 0

    '''
    if False:
        d0 = primeProb + ifPrimePredictNotPrimeProb
        d1 = d0 / ifPrimePredictNotPrimeProb
        d2 = d1 / d0
        isPrime = d2 >= d0


    if reallyPrime is not isPrime and False:
        d0 = primeProb - ifPrimePredictPrimeProb
        d2 = predictedNotPrimeProb + d0
        d3 = d2 + d0
        isPrime = primeProb > d3
        chosen = 1

    if reallyPrime is not isPrime or True:
        d0 = predictedNotPrimeProb - ifPrimePredictPrimeProb
        isPrime = predictedPrimeProb > d0
        chosen = 2

    if reallyPrime is not isPrime:
        d0 = effectivePrimeProb - ifPrimePredictPrimeProb
        isPrime = ifPrimePredictPrimeProb >= d0
        chosen = 2

    if reallyPrime is not isPrime:
        d00 = effectivePrimeProb - ifPrimePredictPrimeProb
        d0 = predictedNotPrimeProb - (1 - primeProb)
        isPrime = ifPrimePredictPrimeProb >= d00 # (d0 if (lastPrime+i) % 2 == 1 else d00)
        chosen = 3

    if reallyPrime is not isPrime:
        d0 = (1-primeProb) + predictedPrimeProb
        b0 = ifPrimePredictNotPrimeProb >= d0
        isPrime = not b0
        chosen = 4

    if reallyPrime is not isPrime:
        d0 = primeProb - ifPrimePredictPrimeProb
        isPrime = predictedPrimeProb >= d0
        chosen = 5

    if reallyPrime is not isPrime:
        d0 = predictedPrimeProb / (1-effectivePrimeProb)
        d1 = predictedNotPrimeProb - d0
        isPrime = (1-effectiveIfPrimeProb) >= d1
        chosen = 6
    '''

    '''
    d0 = predictedNotPrimeProb - ifPrimePredictPrimeProb
    isPrime = predictedPrimeProb > d0

    d0 = (1 - primeProb) + predictedPrimeProb
    b0 = ifPrimePredictNotPrimeProb >= d0
    isPrime1 = not b0

    d0 = predictedPrimeProb / (1 - effectivePrimeProb)
    d1 = predictedNotPrimeProb - d0
    isPrime2 = (1 - effectiveIfPrimeProb) >= d1
    '''

    #d1 = lastPrime / i
    #isPrime = d1 >= (1-primeProb)

    d0 = effectivePrimeProb - ifPrimePredictPrimeProb
    isPrime0 = ifPrimePredictPrimeProb >= d0

    d0 = (1 - primeProb) + predictedPrimeProb
    b0 = ifPrimePredictNotPrimeProb >= d0
    isPrime1 = not b0

    d0 = predictedPrimeProb / (1 - effectivePrimeProb)
    d1 = predictedNotPrimeProb - d0
    isPrime2 = (1 - effectiveIfPrimeProb) >= d1

    d0 = lastPrime % (1-effectiveIfPrimeProb)
    isPrime3 = d0 > predictedNotPrimeProb

    d0 = ifPrimePredictPrimeProb * lastPrime
    d5 = d0 - 1
    isPrime4 = d5 > primeProb

    if isPrime1 is reallyPrime:
        chosen = 1
        isPrime = reallyPrime

    if isPrime2 is reallyPrime:
        chosen = 2
        isPrime = reallyPrime

    if isPrime3 is reallyPrime:
        chosen = 3
        isPrime = reallyPrime

    if isPrime4 is reallyPrime:
        chosen = 4
        isPrime = reallyPrime

    [["d$", 0, "MOD", "d#", 5, "d#", 12], ["b$", 0, "NOT", "b#", 0], ["b$", 1, "OR", "b#", 1, "b$", 0],
     ["d$", 1, "ADD", "d$", 0, "d#", 1], ["d$", 2, "ADD", "d$", 0, "d#", 16], ["b$", 2, "CMP", "d#", 16, "d$", 2],
     ["IF", "b$", 0], ["b$", 1, "GET", "d#", 5, "d#", 3], ["b$", 1, "NOT", "b#", 1], ["b$", 0, "OR", "b#", 0, "b$", 1],
     ["d$", 2, "DIV", "d#", 2, "d$", 0], ["b$", 1, "GT", "d$", 1, "d$", 2], ["END"]]

    if chosen == 6:
        print("check")

    chosens.append(chosen)

    #isPrime = reallyPrime

    if isPrime:
        predictedPrimeProb = ifPrimePredictPrimeProb
        predictedNotPrimeProb = ifPrimePredictNotPrimeProb
        numPrimes += 1
        lastPrime = 0
        primes0.append(step)
        print(step, " is prime")
    else:
        print(step, " is not prime")

    primeProb = numPrimes / (step)
    lastPrime += 1

print("end")