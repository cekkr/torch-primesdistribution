
upTo = 1000

primes = []

primeProb = 0
predictedPrimeProb = 1
predictedNotPrimeProb = 0
ifPrimePredictPrimeProb = 1
ifPrimePredictNotPrimeProb = 0
numPrimes = 0

for step in range(2, upTo):
    i = step - 1
    quanto = 1 / step

    primeStamp = quanto
    primeStamp *= predictedPrimeProb

    ifPrimePredictPrimeProb = predictedPrimeProb
    ifPrimePredictPrimeProb -= primeStamp
    ifPrimePredictNotPrimeProb = 1 - ifPrimePredictPrimeProb

    d0 = primeProb + ifPrimePredictNotPrimeProb
    d1 = d0 / ifPrimePredictNotPrimeProb
    d2 = d1 / d0
    isPrime = d2 >= d0

    '''
    d0 = primeProb - ifPrimePredictPrimeProb
    d2 = predictedNotPrimeProb + d0
    d3 = d2 + d0
    isPrime = primeProb > d3
    '''

    '''
    d0 = predictedNotPrimeProb - ifPrimePredictPrimeProb
    isPrime = predictedPrimeProb > d0
    '''

    if isPrime:
        predictedPrimeProb = ifPrimePredictPrimeProb
        predictedNotPrimeProb = ifPrimePredictNotPrimeProb
        numPrimes += 1
        primes.append(step)
        print(step, " is prime")
    else:
        print(step, " is not prime")

    primeProb = numPrimes / (step)