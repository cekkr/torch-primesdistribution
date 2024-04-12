
upTo = 10000

prime_numbers = []
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
    num += 1

primes0 = []

primeProb = 0
predictedPrimeProb = 1
predictedNotPrimeProb = 0
ifPrimePredictPrimeProb = 1
ifPrimePredictNotPrimeProb = 0
numPrimes = 0
lastPrime = 1

for step in range(2, upTo):
    i = step - 1
    quanto = 1 / step

    effectivePrimeProb = numPrimes / (step)

    primeStamp = quanto
    primeStamp *= predictedPrimeProb

    ifPrimePredictPrimeProb = predictedPrimeProb
    ifPrimePredictPrimeProb -= primeStamp
    ifPrimePredictNotPrimeProb = 1 - ifPrimePredictPrimeProb

    '''
    d0 = primeProb + ifPrimePredictNotPrimeProb
    d1 = d0 / ifPrimePredictNotPrimeProb
    d2 = d1 / d0
    isPrime = d2 >= d0
    '''

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

    '''
    d0 = effectivePrimeProb - ifPrimePredictPrimeProb
    isPrime = ifPrimePredictPrimeProb >= d0
    '''

    d00 = effectivePrimeProb - ifPrimePredictPrimeProb
    d0 = predictedNotPrimeProb - (1 - primeProb)
    isPrime = ifPrimePredictPrimeProb >= d00 # (d0 if (lastPrime+i) % 2 == 1 else d00)

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