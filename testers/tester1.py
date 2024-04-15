
upTo = 1000

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
serieses = []

primeProb = 0
predictedPrimeProb = 1
predictedNotPrimeProb = 0
ifPrimePredictPrimeProb = 1
ifPrimePredictNotPrimeProb = 0
numPrimes = 0
lastPrime = 1

accVar0 = 0

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

    d0 = 0
    if effectivePrimeProb != 0:
        d0 = (1 - effectivePrimeProb) % effectivePrimeProb
    isPrime5 = 0 >= d0

    d0 = 0
    if effectivePrimeProb > 0:
        d0 = step % effectivePrimeProb
    d1 = 0
    if d0 > 0:
        d1 = i / d0
    isPrime6 = ifPrimePredictNotPrimeProb >= d1

    d0 = effectivePrimeProb * primeProb
    isPrime7 = ifPrimePredictPrimeProb > d0

    series = []
    if isPrime1 is reallyPrime:
        chosen = 1
        isPrime = reallyPrime
        series.append(True)
    else:
        series.append(False)

    if isPrime2 is reallyPrime:
        chosen = 2
        isPrime = reallyPrime
        series.append(True)
    else:
        series.append(False)

    if isPrime3 is reallyPrime:
        chosen = 3
        isPrime = reallyPrime
        series.append(True)
    else:
        series.append(False)

    if isPrime4 is reallyPrime:
        chosen = 4
        isPrime = reallyPrime
        series.append(True)
    else:
        series.append(False)

    if isPrime5 is reallyPrime:
        chosen = 5
        isPrime = reallyPrime
        series.append(True)
    else:
        series.append(False)

    if isPrime6 is reallyPrime:
        chosen = 6
        isPrime = reallyPrime
        series.append(True)
    else:
        series.append(False)

    if isPrime7 is reallyPrime:
        chosen = 7
        isPrime = reallyPrime
        series.append(True)
    else:
        series.append(False)

    serieses.append(series)

    '''
    d0 = ifPrimePredictPrimeProb * lastPrime
    d1 = d0 - (1 - effectivePrimeProb)
    d2 = predictedNotPrimeProb - d0
    d3 = d2 * d1
    isPrime = d3 > d2
    '''

    #[["d$", 0, "MUL", "d#", 13, "d#", 6], ["b$", 0, "GT", "d#", 11, "d$", 0]]
    #d0 = effectivePrimeProb * primeProb
    #isPrime = ifPrimePredictPrimeProb > d0

    #[["d$", 0, "MOD", "d#", 14, "d#", 9], ["d$", 1, "ADD", "d$", 0, "d#", 3], ["b$", 0, "GT", "d#", 12, "d$", 0]]
    d0 = effectiveIfPrimeProb % predictedPrimeProb
    d1 = d0 + i
    isNotPrime = quanto <= d0

    '''[["d$", 0, "DIV", "d#", 15, "d#", 8], ["d$", 1, "ADD", "d$", 0, "d#", 13], ["d$", 2, "MUL", "d$", 1, "d$", 0],
     ["b$", 0, "GET", "d#", 15, "d$", 2], ["b$", 1, "GT", "d$", 2, "d#", 10], ["IF", "b$", 0],
     ["b$", 1, "NOT", "b$", 1], ["END"]]'''

    '''
    d0 = 0
    if predictedNotPrimeProb != 0:
        d0 = (1-effectivePrimeProb) / predictedNotPrimeProb
    d1 = d0 + effectivePrimeProb
    d2 = d1 * d0
    b0 = (1-effectivePrimeProb) >= d2
    b1 = d2 > ifPrimePredictNotPrimeProb
    if b0:
        b1 = not b1
    isPrime = b1
    '''

    #[["d$", 0, "DIV", "d#", 15, "d#", 8], ["d$", 1, "DIV", "d$", 0, "d#", 8], ["b$", 0, "GET", "d$", 1, "d#", 8]]
    '''
    d0 = 0
    if predictedNotPrimeProb > 0:
        d0 = (1-effectivePrimeProb) / predictedNotPrimeProb
        d1 = d0 / predictedNotPrimeProb
    isPrime = d1 >= predictedNotPrimeProb
    '''
    #[["d$", 0, "ASSIGN", "d#", 15], ["b$", 0, "CMP", "d#", 3, "d$", 0], ["b$", 1, "DEFAULT", "b$", 0], ["IF", "b$", 1], ["b$", 0, "AND", "b#", 1, "b#", 3], ["END"]]

    d0 = effectivePrimeProb - ifPrimePredictPrimeProb
    isPrime = ifPrimePredictPrimeProb > d0

    d0 = effectivePrimeProb - ifPrimePredictPrimeProb
    isPrime = ifPrimePredictPrimeProb >= d0

    #if step % 2 == 0 and step > 2:
    #    isPrime = isPrime and not isNotPrime

    if chosen == 6:
        print("check")

    chosens.append(chosen)

    #isPrime = isPrime1

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