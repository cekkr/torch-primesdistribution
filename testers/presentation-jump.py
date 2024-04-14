upTo = 10000
primes = []

predictedPrimeProb = 1
predictedNotPrimeProb = 0
numPrimes = 0
primeProb = 0

d0 = 0

for step in range(2, upTo):
    i = step - 1
    quanto = 1 / step

    effectivePrimeProb = numPrimes / step

    primeStamp = quanto
    primeStamp *= predictedPrimeProb

    ifPrimePredictPrimeProb = predictedPrimeProb
    ifPrimePredictPrimeProb -= primeStamp
    ifPrimePredictNotPrimeProb = 1 - ifPrimePredictPrimeProb

    d0 = predictedNotPrimeProb - (1-primeProb)
    isPrime = ifPrimePredictPrimeProb > d0

    if isPrime:
        predictedPrimeProb = ifPrimePredictPrimeProb
        predictedNotPrimeProb = ifPrimePredictNotPrimeProb
        primes.append(step)
        numPrimes += 1

    primeProb = numPrimes / (step)

print("Predicted primes: ", primes)