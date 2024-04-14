# -*- coding: utf-8 -*-
"""
Imports:
"""

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib.pyplot as plt

import random
import math

import json

import psutil
import gc

from datetime import datetime

from main import *

import torch.optim as optim
import torch

device = 'mps'

enableML = True

"""## Costum classes"""

myFloat = np.double
if hasattr(np, 'float128'):
    myFloat = np.float128

output = open("./output.txt", "w")
lastPrintNewLine = True


def myPrint(*argv, **kwargs):
    global lastPrintNewLine

    print(*argv, **kwargs)

    time = ''
    if lastPrintNewLine:
        now = datetime.now()
        time = now.strftime("%m/%d/%Y, %H:%M:%S") + ': '

    end = '\n'
    if 'end' in kwargs:
        end = kwargs['end']
        lastPrintNewLine = False
    else:
        lastPrintNewLine = True

    output.write(time + ''.join(str(v) for v in argv) + end)


"""First, calculate prime numbers up to `upTo`, beginning from 2."""


def is_prime(num):
    if num < 2:
        return False
    for i in range(2, int(num ** 0.5) + 1):
        if num % i == 0:
            return False
    return True


upTo = 1000  # 50.000

numIsPrime = []

prime_numbers = []
count = 0
num = 2

while num <= upTo:
    if is_prime(num):
        prime_numbers.append(num)
        numIsPrime.append(True)
    else:
        numIsPrime.append(False)

    num += 1

myPrint("First " + str(upTo) + " numbers calculated")

"""Then calculate distribution"""

distribution = []
numPrimes = 0
nextPrime = prime_numbers[numPrimes]

n = 2
while n <= upTo:
    if nextPrime == n:
        numPrimes += 1
        if len(prime_numbers) > numPrimes:
            nextPrime = prime_numbers[numPrimes]
        else:
            nextPrime += upTo

    distribution.append(numPrimes)
    n += 1

myPrint("Distribution calculated (", len(distribution), ")")

"""# Q-Learning library

Forking code from https://github.com/farizrahman4u/qlearning4k
"""

batch_size = 30


class Agent:

    def __init__(self, model, input_shape, output_shape):
        self.input_shape = input_shape
        myPrint("input_shape: ", self.input_shape)

        self.output_shape = output_shape
        myPrint("output_shape: ", self.output_shape)

        self.model = model

        self.loss = torch.nn.MSELoss()

        self.optim = optim.SGD([x for x in model.parameters() if x.requires_grad], lr=0.1, weight_decay=1e-3)

        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"Layer: {name}, Size: {param.size()}")

        self.fileTraining = './outputTrain.txt'
        self.dirOutputs = './outputs/'

    def get_game_data(self, game):
        frame = game.get_frame()
        return frame

    def saveJson(self, file, var):
        f = open(file, "w")
        f.write(json.dumps(var))
        f.close()

    def readJson(self, file):
        if (os.path.exists(file)):
            f = open(file, "r")
            return json.loads(f.read())
        return None

    def predictOptions(self, game):
        reqs = []
        for o in range(0, game.optionsLen):
            game.selOption = o
            frame = game.get_frame()
            frame = np.array(frame)
            if frame.shape[0] > 0:
                prediction = self.model(torch.tensor(frame, dtype=torch.float32, device=device).view(1, frame.shape[0], game.ideWidth*2))
                reqs.append(prediction.cpu().detach())

        return reqs

    def checkRamUsage(self):
        usedRam = psutil.virtual_memory()[2]  # in %
        return usedRam > 75

    def train(self, game, nb_epoch=100000, epsilon=[1.0, 0.0], epsilon_rate=1.0, observe=0, checkpoint=100,
              weighedScore=False):
        if type(epsilon) in {tuple, list}:
            delta = ((epsilon[0] - epsilon[1]) / (nb_epoch * epsilon_rate))
            final_epsilon = epsilon[1]
            epsilon = epsilon[0]
        else:
            final_epsilon = epsilon

        model = self.model
        win_count = 0

        observeModel = False
        totEpochs = nb_epoch
        minEpochsControl = totEpochs / 10
        limitTrainingCount = 99999

        epoch = 0

        avgTotalIsolatedLines = game.num_lines / 2

        bestScore = 1.0
        bestScoreLines = 0

        lastTrain = self.readJson(self.fileTraining)
        if lastTrain != None:
            delta = lastTrain['delta']
            epsilon = lastTrain['epsilon']
            final_epsilon = lastTrain['final_epsilon']
            epoch = lastTrain['epoch']
            nb_epoch = lastTrain['nb_epoch']
            observeModel = lastTrain['observeModel']
            limitTrainingCount = lastTrain['limitTrainingCount']
            avgTotalIsolatedLines = lastTrain['avgTotalIsolatedLines']
            bestScore = lastTrain['bestScore']
            bestScoreLines = lastTrain['bestScoreLines']
        
        while epoch < nb_epoch:
            posVal = epoch/(nb_epoch)
            if posVal > 1:
                posVal = 1

            epoch += 1

            loss = 0.
            accuracy = 0.

            game.reset()

            ### Saved scores
            ###
            linesScores = [0] * (game.num_lines + 1)
            isolatedHashScores = {}

            def getArrHash(arr):
                return hash(arr.tobytes())

            def checkViewScore(view, score):
                h = getArrHash(view)
                if h in isolatedHashScores:
                    s = isolatedHashScores[h]

                    if s < score:
                        isolatedHashScores[h] = score
                        return True
                    else:
                        return False
                else:
                    isolatedHashScores[h] = score
                    return True

            ### End saved scores
            ###

            avgNumberIsolatedLines = 0
            avgNumberIsolatedLinesCount = 0

            cycles = 0
            game_over = False
            a = 0
            while not game_over:

                options = game.optionsLen
                if np.random.random() < epsilon or epoch < observe:
                    if 'IF' in game.options:
                        retry = True
                        while retry:
                            a = int(np.random.randint(options))
                            if a == game.options.index('IF'):
                                retry = int(np.random.randint(3)) > 0
                            else:
                                retry = False
                    else:
                        a = int(np.random.randint(options))
                else:
                    p = self.predictOptions(game)
                    a = 0
                    if len(p) > 0:
                        a = int(np.argmin(p))

                game.goDown(a)

                game.goRight()  # next parameter

                if game.inNewLine:
                    score = game.get_score()

                    if score >= 0:
                        # Train only the working algorithm
                        isolatedInstructions = game.curWinnerInstructions

                        if abs(score) < bestScore or (score == bestScore and bestScoreLines > len(isolatedInstructions)):
                            if abs(score) < bestScore/2:
                                bestScore = abs(score)
                                bestScoreLines = len(isolatedInstructions)

                            with open(self.dirOutputs + '/' + str(score) + '.txt', 'w') as file:
                                file.write(json.dumps(isolatedInstructions))

                        scoreWeight = score
                        if weighedScore and score > 0 and False:
                            lineWeight = len(isolatedInstructions) * score
                            avgNumberIsolatedLines = (avgNumberIsolatedLines * avgNumberIsolatedLinesCount) + lineWeight
                            avgNumberIsolatedLinesCount += score
                            avgNumberIsolatedLines /= avgNumberIsolatedLinesCount

                            weight = len(isolatedInstructions)

                            if weight <= avgTotalIsolatedLines:
                                weight = (1 - (weight / avgTotalIsolatedLines)) * -1
                            else:
                                weight = (weight - avgTotalIsolatedLines) / (game.num_lines - avgTotalIsolatedLines)

                            weight *= -1  # is not perfect, but it should work...
                            scoreWeight = (math.sin(score * (math.pi / 2)) * weight) + (score * (1 - weight))
                            myPrint("Lines: ", len(isolatedInstructions), "\t Weight: ", weight, "\t scoreWeight:",
                                    scoreWeight,
                                    "\t avgLines:", avgNumberIsolatedLines)

                        print("Working on score ", scoreWeight)
                        if enableML:
                            for i in range(0, game.countInstructionsElements(isolatedInstructions)):
                                view = game.get_state(i + 1, isolatedInstructions)
                                if checkViewScore(view, scoreWeight):
                                    repeat = int((20 * (1-scoreWeight))+1)
                                    for r in range(0, repeat):
                                        #modelsGen.trainInput(view, scoreWeight)
                                        self.optim.zero_grad()
                                        tensor = torch.tensor(view, dtype=torch.float32).to(device=device).view(1, view.shape[0], game.ideWidth*2)
                                        pred = model(tensor)
                                        target = torch.tensor([[scoreWeight]], dtype=torch.float32).to(device=device)
                                        err = self.loss(pred, target)
                                        err.backward()
                                        self.optim.step()
                                        loss += float(err)
                                        print("Train with loss ", err)


                        # Save working lines max score
                        for i in game.workingLines:
                            if linesScores[i] < scoreWeight:
                                linesScores[i] = scoreWeight

                    game_over = game.is_over()

                lineVal = (game.focus_y/game.num_lines)**2
                if lineVal > posVal:
                    game_over = True

            # Train the best scores of the total script
            if enableML:
                totElements = 0
                for i in range(0, len(game.instructions)):
                    instr = game.instructions[i]
                    instrLen = len(instr)

                    if linesScores[i] > 0:
                        for u in range(totElements, totElements + instrLen):
                            view = game.get_state(u + 1)
                            self.optim.zero_grad()
                            pred = model(torch.tensor(view, dtype=torch.float32).to(device=device).view(1, view.shape[0], game.ideWidth*2))
                            err = self.loss(pred, torch.tensor([[linesScores[i]]], dtype=torch.float32).to(device=device))
                            err.backward()
                            self.optim.step()
                            loss += float(err)

                    totElements += instrLen

            #gc.collect()

            if checkpoint and ((epoch + 1 - observe) % checkpoint == 0 or epoch >= nb_epoch):
                model.save(self.dirOutputs)

                save = {
                    'delta': float(delta),
                    'epsilon': float(epsilon),
                    'final_epsilon': float(final_epsilon),
                    'epoch': epoch,
                    'nb_epoch': nb_epoch,
                    'observeModel': observeModel,
                    'limitTrainingCount': limitTrainingCount,
                    'avgTotalIsolatedLines': float(avgTotalIsolatedLines),
                    'bestScore': float(bestScore),
                    'bestScoreLines': float(bestScoreLines)
                }

                self.saveJson(self.fileTraining, save)

            if game.is_won():
                win_count += 1

            if epsilon > final_epsilon and epoch >= observe:
                epsilon -= delta

                if not observeModel and epsilon < delta:
                    if (nb_epoch - epoch) < minEpochsControl:
                        nb_epoch = minEpochsControl
                        epoch = 0
                    observeModel = True

            if cycles > 0:
                loss /= cycles
                #loss /= upTo
                accuracy /= cycles

            avgTotalIsolatedLines = (avgTotalIsolatedLines + avgNumberIsolatedLines) / 2
            myPrint("avgTotalIsolatedLines: ", avgTotalIsolatedLines)

            myPrint("=========================================")
            myPrint(
                "Epoch {:03d}/{:03d} | Loss {:.4f} | Epsilon {:.2f} | Win count {}".format(epoch + 1, nb_epoch, loss,
                                                                                           epsilon,
                                                                                           win_count))
            myPrint("=========================================")

    def play(self, game, nb_epoch=10, epsilon=0., visualize=False):
        model = self.model
        win_count = 0

        for epoch in range(nb_epoch):
            game.reset()
            self.clear_frames()
            S = self.get_game_data(game)

            game_over = False
            while not game_over:
                if np.random.rand() < epsilon:
                    myPrint("random")
                    action = int(np.random.randint(0, game.nb_actions))
                else:
                    q = model(torch.tensor(np.array([S]), dtype=torch.float32, device=device).view(1, S.shape[0], game.ideWidth*2))
                    q = q[0]

                    possible_actions = game.get_possible_actions()
                    q = [q[i] for i in possible_actions]
                    action = possible_actions[np.argmax(q)]

                game.play(action)
                S = self.get_game_data(game)

                game_over = game.is_over()

            if game.is_won():
                win_count += 1

        myPrint("Accuracy {} %".format(100. * win_count / nb_epoch))

        if visualize and False:  # todo: handle this possibility on colab
            if 'images' not in os.listdir('.'):
                os.mkdir('images')
            for i in range(len(frames)):
                plt.imshow(frames[i], interpolation='none')
                plt.savefig("images/" + game.name + str(i) + ".png")


###
### Game parent class
###

class Game(object):

    def __init__(self):
        self.reset()

    @property
    def name(self):
        return "Game"

    @property
    def nb_actions(self):
        return 0

    def reset(self):
        pass

    def play(self, action):
        pass

    def get_state(self):
        return None

    def get_score(self):
        return 0

    def is_over(self):
        return False

    def is_won(self):
        return False

    def get_frame(self):
        return self.get_state()

    def draw(self):
        return self.get_state()

    def get_possible_actions(self):
        return range(self.nb_actions)


"""
## Labels
"""

labels = []

storeTypes = []
storeTypes.append('d#')
storeTypes.append('b#')
storeTypes.append('d$')
storeTypes.append('b$')
labels.extend(storeTypes)

alternativeStartInstruction = ['IF', 'END'] #  (removed for simplicity)
labels.extend(alternativeStartInstruction)

"""
## Operations
"""

neutralOps = ['ASSIGN', 'DEFAULT'] # ASSIGN removed after 'DEFAULT'
labels.extend(neutralOps)

decimalOps = []
decimalOps.append('ADD')
decimalOps.append('SUB')
decimalOps.append('MUL')
decimalOps.append('DIV')
decimalOps.append('MOD')
labels.extend(decimalOps)

boolOps = []
boolOps.append('NOT')
boolOps.append('OR')
boolOps.append('AND')
boolOps.append('CMP')
boolOps.append('GT')
boolOps.append('GET')
# boolOps.append('LT')
# boolOps.append('LET')
labels.extend(boolOps)

boolOps_bool = ['NOT', 'OR', 'AND']
boolOps_decimal = ['GT', 'GET', 'LT', 'LET']

oneArgOps = ['NOT']

"""
# Default costants and variables
"""

storesNames = {}
storesNamesAssoc = {}
stores = {}

boolCostNames = [
    'false',
    'true',
    'isPrime0',
    'isPrime1'
]

boolVarsNames = [
    # 'isPrime'
]

decimalCostNames = [
    'zero',
    'one',
    'step',
    'i',
    'numPrimes',
    'lastPrime',#5
    'primeProb', #6
    'notPrimeProb',
    'predictedNotPrimeProb', #8
    'predictedPrimeProb',
    'ifPrimePredictNotPrimeProb', #10
    'ifPrimePredictPrimeProb', #11
    'quanto',
    'effectivePrimeProb', # 13
    'effectiveIfPrimeProb',
    'effectivePrimeNotProb',
    'effectiveIfPrimeNotProb',
    #'quantoSum',
    #'primeQuantoSum',
    'primeStamp',
    #'lastPrimeDistance'
]

decimalVarsNames = [
    'jumpBy'
]

engineFault = False

def getNameType(name):
    if name in storesNamesAssoc:
        return storesNamesAssoc[name]
    else:
        return None


def setStore(name, value):
    isCost = name.startswith('#')
    name = name[1:]

    isBool = isinstance(value, (np.bool_, bool))

    storeTypeSaved = getNameType(name)

    storeType = ''
    if isBool:
        storeType += 'b'
    else:
        storeType += 'd'

    if isCost:
        storeType += '#'
    else:
        storeType += '$'

    typeNames = None
    pos = -1

    if storeTypeSaved is not None:
        if storeTypeSaved != storeType:
            raise Exception("Excepted store of type " + storeTypeSaved + ", received " + storeType)

        typeNames = storesNames[storeType]
        pos = typeNames.index(name)

    else:
        typeNames = storesNames[storeType]

        pos = len(typeNames)
        typeNames.append(name)
        storesNamesAssoc[name] = storeType

    if not isBool and not isinstance(value, myFloat):
        value = myFloat(value)

    st = stores[storeType]

    if len(st) <= pos:
        st.append(value)
    else:
        st[pos] = value


def getStore(name):
    isCost = name.startswith('#')
    isVar = name.startswith('$')

    if isCost or isVar:
        name = name[1:]

    storeTypeSaved = getNameType(name)

    if storeTypeSaved is None:
        raise Exception("Store " + name + " not found")

    typeNames = storesNames[storeTypeSaved]
    pos = typeNames.index(name)

    return stores[storeTypeSaved][pos]


def setStoreFields(fields, value):
    storeType = fields[0]
    pos = fields[1]

    reqBool = storeType.startswith('b')
    if reqBool:
        try:
            value = bool(value)
        except:
            myPrint('neu')

    valBool = isinstance(value, bool)

    if reqBool != valBool:
        raise Exception("Different type assignation for ", storeType, " (valueIsBool:", valBool, ", reqBool:", reqBool,
                        ", value:", value, ")");

    if not valBool and not isinstance(value, myFloat):
        value = myFloat(value)

    st = stores[storeType]

    if len(st) <= pos:
        st.append(value)
    else:
        st[pos] = value

def assocNames(names, storeType):
    storesNames[storeType] = names
    stores[storeType] = []

    pos = 0
    for name in names:
        storesNamesAssoc[name] = storeType
        setStoreFields([storeType, pos], False if storeType.startswith('b') else 0)
        pos += 1

def resetStores():
    global storesNames
    global stores
    global storesNamesAssoc
    storesNames = {}
    storesNamesAssoc = {}
    stores = {}

    assocNames(boolCostNames, 'b#')
    assocNames(boolVarsNames, 'b$')
    assocNames(decimalCostNames, 'd#')
    assocNames(decimalVarsNames, 'd$')


resetStores()

def getStoreFields(fields):
    if len(fields) != 2:
        myPrint("Error: as fields I got a ", fields)

    storeType = fields[0]
    pos = fields[1]

    if not checkStoreFields(fields):
        engineFault = True
        myPrint("Error for fields", fields, stores[storeType])
        raise Exception("Fields fault: check it")

        if storeType.startswith('d'):
            return 0
        else:
            return False

    return stores[storeType][pos]


def checkStoreFields(fields):
    storeType = fields[0]
    pos = fields[1]

    return pos < len(stores[storeType])


"""# Engine
This is the engine that interpret commands and executes the cycles.
"""


def fieldIsStore(f):
    return f in storeTypes


def instructionsToByteCode(instructions):
    res = []
    parent = []
    context = res
    end = False

    for instr in instructions:
        if end:
            break

        resInstr = []

        acc = ''  # accumuling
        nextIsStore = False
        isCondition = False

        def addToResInstr(val):
            if isCondition:
                resInstr['condition'] = val
            else:
                resInstr.append(val)

        for f in range(0, len(instr)):
            field = instr[f]

            if field == '':  # it depends how is done the instruction array
                break

            # myPrint('Reading field ',field, nextIsStore)

            if f == 0:  # if it's the first field of the line
                if field == 'IF':
                    resInstr = {'statement': field, 'condition': [], 'context': []}
                    isCondition = True
                    context.append(resInstr)
                    parent.append(context)
                    context = resInstr['context']
                    continue

                if field == 'END':
                    if len(parent) == 0:
                        end = True
                        break

                    context = parent.pop()

            if not nextIsStore:  # is not store second value
                nextIsStore = fieldIsStore(field)

                if not nextIsStore:
                    addToResInstr([field])  # add normally
                else:
                    acc = field  # accumulate store first value for next field
            else:  # is store second value
                addToResInstr([acc, field])
                nextIsStore = False

        if not isCondition:  # if it's a condition it's already added to context
            context.append(resInstr)

    return res


###
### Interpret bytecode
###

def interpretBytecode_condition(condition):
    # myPrint('Condition: ', condition)
    match condition['statement']:
        case 'IF':
            if getStoreFields(condition['condition']):
                interpretBytecode(condition['context'])

        case _:
            raise Exception("Condition " + condition['statement'] + " not found")


def interpretBytecode_line(line):
    lineLen = len(line)

    if lineLen == 0:
        return

    if lineLen < 3:
        cmd = line[0][0]

        if cmd == 'END':
            return  # todo: Handle better the END command

    assignTo = line[0]

    operation = line[1][0]
    arg1 = getStoreFields(line[2])

    if lineLen > 3:
        arg2 = getStoreFields(line[3])

    # Here there is not control about the right type used for the operation:
    # it's assumed that Calculon automatically removed wrong types from selectionable options

    write = True
    res = None
    match operation:
        # Neutral
        case 'ASSIGN':
            res = arg1

        case 'DEFAULT':
            res = arg1
            write = not checkStoreFields(assignTo)

        # Decimal
        case 'ADD':
            res = arg1 + arg2

        case 'SUB':
            res = arg1 - arg2

        case 'MUL':
            res = arg1 * arg2

        case 'DIV':
            if arg2 == 0:
                res = 0  # wrong but temporary response
                # todo: Handle this condition blocking the game
            else:
                res = arg1 / arg2

        case 'MOD':
            if arg2 == 0:
                res = 0  # wrong but temporary response
                # todo: Handle this condition blocking the game
            else:
                res = arg1 % arg2

        # Bool
        case 'NOT':
            res = not arg1

        case 'OR':
            res = arg1 or arg2

        case 'AND':
            res = arg1 and arg2

        case 'CMP':
            res = arg1 == arg2

        case 'GT':
            res = arg1 > arg2

        case 'GET':
            res = arg1 >= arg2

        case _:
            raise Exception('Operation ' + operation + ' not found')

    if write:
        setStoreFields(assignTo, res)


def interpretBytecode(bytecode):
    for line in bytecode:
        isCondition = isinstance(line, dict)

        if isCondition:
            interpretBytecode_condition(line)
        else:
            interpretBytecode_line(line)


###
### Execute Cycles
###

def resetEngine():
    engineFault = False

    resetStores()

    setStore('#zero', 0)
    setStore('#one', 1)

    setStore('#false', False)
    setStore('#true', True)

    setStore('#step', 2)
    setStore('#i', 1)

    setStore('#numPrimes', 0)
    setStore('#lastPrime', 0)

    setStore('#primeProb', 0)
    setStore('#notPrimeProb', 0)

    setStore('#predictedNotPrimeProb', 0)
    setStore('#predictedPrimeProb', 1)

    setStore('#ifPrimePredictNotPrimeProb', 0)
    setStore('#ifPrimePredictPrimeProb', 1)

    setStore('#effectivePrimeProb', 0)
    setStore('#effectiveIfPrimeProb', 0)

    setStore('#effectivePrimeNotProb', 1)
    setStore('#effectiveIfPrimeNotProb', 0)

    setStore('#quanto', 1)

    setStore('#isPrime0', False)
    setStore('#isPrime1', False)

    setStore('#primeStamp', 0)
    setStore('$jumpBy', 0)


resetEngine()  # reset by default


def executeCycles(instructions, isPrimeVar=0):
    bytecode = instructionsToByteCode(instructions)

    resetEngine()

    # Score variables
    distributionDiff = 0
    distributionMaxDiff = 0

    # Cache stores
    step = getStore('#step')
    i = getStore('#i')
    quanto = None
    ifPrimePredictNotPrimeProb = 0
    ifPrimePredictPrimeProb = 0
    predictedPrimeProb = getStore('#predictedPrimeProb')
    predictedNotPrimeProb = getStore('#predictedPrimeProb')
    isPrime = None
    numPrimes = getStore('#numPrimes')
    lastPrime = getStore('#lastPrime')
    primeProb = 0
    notPrimeProb = 1
    quantoSum = 0
    primeQuantoSum = 0

    jumpBy = 0

    while (step <= upTo):
        i = step - 1
        quanto = 1 / step

        effectivePrimeProb = numPrimes / (step)
        effectiveIfPrimeProb = (numPrimes + 1) / (step)

        ifPrimePredictNotPrimeProb = getStore('#predictedNotPrimeProb')
        predictedNotPrimeProb = ifPrimePredictNotPrimeProb

        ifPrimePredictPrimeProb = getStore('#predictedPrimeProb')
        predictedPrimeProb = ifPrimePredictPrimeProb

        # Calculate if prime prediction probability
        quantoSum += quanto
        primeStamp = quanto
        primeStamp *= predictedPrimeProb
        ifPrimePredictPrimeProb -= primeStamp
        ifPrimePredictNotPrimeProb = 1 - ifPrimePredictPrimeProb

        d0 = predictedNotPrimeProb - ifPrimePredictPrimeProb
        isPrime0 = predictedPrimeProb > d0

        d0 = effectivePrimeProb - ifPrimePredictPrimeProb
        isPrime1 = ifPrimePredictPrimeProb >= d0

        setStore('#i', i)
        setStore('#quanto', quanto)
        setStore('#ifPrimePredictNotPrimeProb', ifPrimePredictNotPrimeProb)
        setStore('#ifPrimePredictPrimeProb', ifPrimePredictPrimeProb)
        setStore('#effectivePrimeProb', (numPrimes / step))
        setStore('#effectiveIfPrimeProb', ((numPrimes+1) / i))
        setStore('#effectivePrimeNotProb', 1-(numPrimes / step))
        setStore('#effectiveIfPrimeNotProb', 1 - ((numPrimes+1) / i))
        #setStore('#quantoSum', quantoSum)
        setStore('#primeStamp', primeStamp)

        setStore('#isPrime0', isPrime0)
        setStore('#isPrime1', isPrime1)

        setStore('$jumpBy', 0)

        ###
        ### Cycle
        ###

        isPrime = False
        if jumpBy == 0:
            interpretBytecode(bytecode)

            ###
            ### End cycle
            ###


            isPrime = getStoreFields(['b$', isPrimeVar])
        else:
            jumpBy -= 1

        if isPrime:
            numPrimes += 1
            lastPrime = 0

            predictedNotPrimeProb = ifPrimePredictNotPrimeProb
            predictedPrimeProb = ifPrimePredictPrimeProb

            primeQuantoSum += quanto

        primeProb = numPrimes / (step)
        notPrimeProb = 1 - primeProb

        lastPrime += 1

        if jumpBy == 0:
            jumpBy = getStore('$jumpBy')
            try:
                jumpBy = int(jumpBy)
            except:
                jumpBy = 0

            if jumpBy < 0:
                jumpBy = 0

        step += 1

        setStore('#numPrimes', numPrimes)
        setStore('#lastPrime', lastPrime)
        setStore('#predictedNotPrimeProb', predictedNotPrimeProb)
        setStore('#predictedPrimeProb', predictedPrimeProb)
        setStore('#primeProb', primeProb)
        setStore('#notPrimeProb', notPrimeProb)
        setStore('#step', step)
        #setStore('#primeQuantoSum', primeQuantoSum)
        #setStore('#lastPrimeDistance', (1/lastPrime))

        # Add score
        i = int(i)

        if True:
            effectiveNumPrimes = distribution[i - 1]
            numPrimesDiff = abs(numPrimes - effectiveNumPrimes) / step
            numPrimesDiff = numPrimesDiff ** (1/2)

            distributionDiff += numPrimesDiff
            distributionMaxDiff += 1

        '''
        if numPrimesDiff < 0:
            numPrimesDiff /= effectiveNumPrimes
            numPrimesDiff *= -1
            distributionDiff += numPrimesDiff

        elif numPrimesDiff > 0:
            numPrimesDiff /= i - effectiveNumPrimes
            distributionDiff += numPrimesDiff

        distributionMaxDiff += 1
        '''

    distributionDiff /= distributionMaxDiff
    return distributionDiff


"""# The 'game'
This is the implementation of *Game* class for implementing the IDE. This 'game' is called Calculon.
"""

actions = {0: 'right', 1: 'down'}  # down action is currently unused in the q learning model
ideWidth = 7

# Game options
drawLineNumber = False
dontAllowEndOnDepth0 = True
forceAssignToNewVar = False
allowAssign = True

if not allowAssign:
    neutralOps.remove('ASSIGN')

rewardVars = [
    # ['b$', storesNames['b$'].index('isPrime'), 3],
    ['d#', storesNames['d#'].index('primeProb')],
    ['d#', storesNames['d#'].index('predictedPrimeProb')],
    ['d#', storesNames['d#'].index('lastPrime')]
]


def cloneDictOfArr(arr):
    res = {}
    for i in arr:
        res[i] = arr[i].copy()
    return res


def checkVarReward(stype, num=-1):
    reward = 0
    for var in rewardVars:
        if num >= -1:
            if stype == var[0] and num == var[1]:
                varReward = 1
                if len(var) > 2:
                    varReward = var[2]
                return varReward
        else:
            if stype == var[0]:
                reward += 1
                return 1  # disable store type reward weight
    return reward


class Calculon(Game):

    def __init__(self, num_lines=100):
        self.num_lines = num_lines
        self.reset()
        self.state_changed = True

        self.maxScore = 0
        self.maxScoreLen = 0

        global ideWidth
        self.ideWidth = ideWidth
        if drawLineNumber:
            self.ideWidth += 1

    @property
    def name(self):
        return "Calculon"

    @property
    def nb_actions(self):
        return 2

    def play(self, action):
        if action not in range(self.nb_actions):
            myPrint('Called wrong action: ', action)
            self.currentReward = -2  # punish the agent ^-^
        else:
            self.currentAction = action
            self.currentReward = 1  # currently unused

            try:
                match action:
                    case 0:  # right
                        self.goRight()

                    case 1:  # down
                        self.goDown()

            except Exception as error:
                myPrint('instruction: ', self.instructions[self.focus_y])
                raise Exception('Error: ', error)

        self.lineReward += self.currentReward
        return self.currentReward

    def extractWinnerVarInstructions(self):
        wv = self.lastBoolVarAssign

        instructionsCp = self.instructions.copy()

        # Compensate missing ends
        missingEnds = 0
        for instr in instructionsCp:
            if len(instr) > 0:
                if instr[0] == 'IF':
                    missingEnds += 1
                elif instr[0] == 'END':
                    missingEnds -= 1

        if missingEnds > 0:
            for i in range(0, missingEnds):
                instructionsCp.append(['END'])

        # Find first important instruction
        startFrom = len(instructionsCp)
        while startFrom > 0:
            startFrom -= 1
            instr = instructionsCp[startFrom]

            if len(instr) == 0:
                continue

            if instr[0] == 'b$' and instr[1] == wv:
                break

        usedVars = {'b$': [wv], 'd$': []}
        assigns = {'d$': [], 'b$': []}
        instructions = []

        def getAssignIndex(var):
            nonlocal assigns
            ass = assigns[var[0]]
            ivar = -1
            for i in range(0, len(ass)):
                if ass[i] == var[1]:
                    ivar = i
                    break
            return ivar

        def checkAssign(var):
            nonlocal assigns
            ass = assigns[var[0]]
            ivar = getAssignIndex(var)

            if ivar >= 0:
                ass.pop(ivar)

            ass.insert(0, var[1])

        self.workingLines = []
        i = startFrom

        previousStackWasRelevant = False

        def stack():
            nonlocal usedVars
            nonlocal i
            nonlocal instructions
            nonlocal previousStackWasRelevant

            stackInstructions = []
            stackWorkingLines = []
            stackIsRelevant = False

            stackLen = 0
            while i >= 0:
                stackLen += 1
                instr = instructionsCp[i]
                i -= 1

                assign = None
                vars = []

                relevant = False
                endOfStack = False
                isAssign = True
                isVar = False
                lastVarType = ''
                isCondition = False
                for field in instr:
                    if isVar:
                        vars.append([lastVarType, field])
                        if isAssign:
                            assign = [lastVarType, field]
                        isVar = False

                    elif str(field).endswith('$'):
                        isVar = True
                        lastVarType = field
                    else:
                        if isAssign:
                            # Check stack
                            if field == 'END':
                                line = i + 1
                                if line < len(self.instructions):
                                    stackWorkingLines.append(line)

                                stack()
                            elif field == 'IF':
                                relevant = stackIsRelevant
                                stackInstructions.append(['END'])
                                endOfStack = True
                                isCondition = True

                            isAssign = False

                # Check if it's a relevant instruction
                if not isCondition:
                    for var in vars:
                        if var[1] in usedVars[var[0]]:
                            relevant = True
                            break

                if relevant:
                    stackIsRelevant = True

                    for var in vars:
                        if var[1] not in usedVars[var[0]]:
                            usedVars[var[0]].append(var[1])

                    if assign != None:
                        checkAssign(assign)

                    stackWorkingLines.append(i + 1)
                    stackInstructions.insert(0, instr.copy())
                    stackIsRelevant = True

                if endOfStack:
                    break

            if stackIsRelevant:
                nonlocal instructions
                instructions = stackInstructions + instructions
                self.workingLines = self.workingLines + stackWorkingLines

            return stackIsRelevant

        while i >= 0:
            stack()

        # Reorder assigns
        for i in range(0, len(instructions)):
            instr = instructions[i]

            isVar = False
            lastVarType = ''
            for f in range(0, len(instr)):
                field = instr[f]
                if isVar:
                    newIndex = getAssignIndex([lastVarType, field])
                    if newIndex == -1:  # just for debug purposes
                        myPrint("newIndex error")
                    instr[f] = newIndex
                    isVar = False
                elif str(field).endswith('$'):
                    isVar = True
                    lastVarType = field

        self.curWinnerInstructions = instructions
        self.curWinnerInstructions_winner = getAssignIndex(['b$', wv])

        return instructions  # finally!

    def newLine(self):
        if self.focus_y >= 0:
            # Calculate current score
            self.inNewLine = True
            self.lastLineLen = len(self.curLine)

            self.usedStores = self.lineUsedStores

        myPrint('Written line: ', self.curLine, '   ', self.focus_y, '/', self.num_lines)

        if self.curLine_isCondition:
            self.newStack()

        self.resetCurLine()

        self.focus_y += 1
        self.focus_x = 0

        if self.focus_y >= self.num_lines:
            self.checkGameEnd()

    def resetCurLine(self, removeLine=False):
        if removeLine:
            self.instructions.pop(len(self.instructions) - 1)
            self.lastBoolVarAssign = None

        self.curLine = []
        self.instructions.append(self.curLine)

        self.endOfLine = 7

        self.focus_x = 0

        self.curLine_previousIsStoreTypes = False
        self.curLine_isOperation = False
        self.curLine_isCondition = False
        self.curLine_isAssign = False
        self.curLine_argNum = 0

        self.totalReward += self.lineReward
        self.lineReward = 0

        self.forkUsedStores()

    def forkUsedStores(self):
        self.lineUsedStores = self.usedStores.copy()

    def loadOptions_variables(self):
        for t in range(2, len(storeTypes)):  # Add just variables
            self.options.append(storeTypes[t])

    def loadOptions_decimal(self):
        for t in range(0, len(storeTypes)):  # Add just variables
            if t % 2 == 0:
                self.options.append(storeTypes[t])

    def loadOptions_bool(self):
        for t in range(0, len(storeTypes)):  # Add just variables
            if t % 2 == 1:
                self.options.append(storeTypes[t])

    def checkGameEnd(self):
        if self.current_score == 1:
            self.game_won = True
        else:
            self.game_won = False

        self.game_over = True

    def getNumStores(self, stype, excepted=-1):
        res = self.usedStores[stype]

        if res == excepted:
            self.lineUsedStores[stype] += 1

        return res

    def loadOptions_Condition(self):
        stores = []

        numVars = self.getNumStores('b$')
        for s in range(0, numVars):
            if s not in self.varUsedInIf:
                stores.append(s)

        return stores

    def countConditionOptions(self):
        return len(self.loadOptions_Condition())

    def loadOptions(self):
        self.options = []
        self.selOption = 0

        hasStore = False
        self.curLine_isAssign = self.focus_x == 1 and self.curLine_previousIsStoreTypes

        # Load available stores
        if self.curLine_previousIsStoreTypes:
            if self.curLine_isAssign:

                if forceAssignToNewVar:
                    if self.depth > 0:
                        for i in range(0, self.getNumStores(self.curLine_storeType)):
                            self.options.append(i)
                else:
                    for i in self.assignUsedStores[self.curLine_storeType]:
                        self.options.append(i)

                if self.depth == 0:
                    self.options.append(self.getNumStores(self.curLine_storeType))

            elif self.curLine_isCondition:
                self.options = self.loadOptions_Condition()
            else:
                for i in range(0, self.getNumStores(self.curLine_storeType)):
                    self.options.append(i)

                if self.curLine_argNum == 2:
                    if self.curLine_2argSt == self.curLine_storeType:
                        self.options.remove(self.curLine_2argNum)

        else:
            match self.focus_x:
                case 0:
                    self.loadOptions_variables()
                    self.options.extend(alternativeStartInstruction)
                    hasStore = True

                case 1:  # In case of condition
                    if self.curLine_isCondition:
                        self.options = ['b$']

                case 2:  # In case of operation
                    self.options.extend(neutralOps)

                    # Handle DEFAULT operation
                    if self.depth > 0:
                        self.options.remove('DEFAULT')
                    elif self.getNumStores(self.curLine_assignType) != self.curLine_assignNum:
                        self.options.remove('DEFAULT')

                    if self.curLine_assignIsBool:
                        self.options.extend(boolOps)
                    else:
                        self.options.extend(decimalOps)

                    self.curLine_isOperation = True
                    hasStore = True

                case 3:
                    match self.curLine_opType:
                        case 'n':
                            if self.curLine_assignIsBool:
                                self.loadOptions_bool()
                            else:
                                self.loadOptions_decimal()

                        case 'b':
                            if self.curLine_opReq == 'b':
                                self.loadOptions_bool()
                            elif self.curLine_opReq == 'd':
                                self.loadOptions_decimal()
                            else:
                                self.options.extend(storeTypes)

                        case 'd':
                            self.loadOptions_decimal()

                        case _:
                            myPrint('Wrong curLine_opType: ', self.curLine_opType)

                    if not self.curLine_isOperation:
                        raise Exception('This line is not an operation...')

                    hasStore = True

                case 5:
                    if self.curLine_argsAreBool:
                        self.loadOptions_bool()
                    else:
                        self.loadOptions_decimal()

                        hasStore = True

            if hasStore:
                remove = []
                for opt in self.options:
                    if opt in stores:
                        if self.focus_x == 0:
                            if self.depth > 0 and len(self.assignUsedStores[opt]) == 0:
                                remove.append(opt)
                        else:
                            if self.getNumStores(opt) == 0:
                                remove.append(opt)

                for rem in remove:
                    self.options.remove(rem)

        if dontAllowEndOnDepth0 and self.depth <= 0:  # and not self.maxScoreSurpass:
            if 'END' in self.options:
                self.options.remove('END')

        if 'IF' in self.options and (self.countConditionOptions() == 0):
            self.options.remove('IF')

        # todo: Delete these lines (just for debug purposes)
        # if self.focus_x > 1:
        #  self.options = []

        if len(self.options) == 0:
            myPrint("All options excluded...")
            myPrint("curLine: ", self.curLine)

            # Reset curLine
            self.resetCurLine(True)
            self.loadOptions()

        #random.shuffle(self.options)

    def newStack(self):
        self.assignStoresStack.append(self.assignUsedStores)
        self.initAssignStores()

    def initAssignStores(self):
        if self.depth == 0:
            self.assignUsedStores = {'d$': [], 'b$': []}

            self.setAsUsedStoreType('d$')
            self.setAsUsedStoreType('b$')
        else:
            self.assignUsedStores = self.assignUsedStores.copy()

    def setAsUsedStoreType(self, stype):
        for i in range(0, self.getNumStores(stype)):
            self.varIsUsed(stype, i)

    def varIsAssigned(self, stype, var):
        if stype == 'b$':
            self.lastBoolVarAssign = var

            if var in self.varUsedInIf:
                self.varUsedInIf.remove(var)

        if stype in self.assignUsedStores and var in self.assignUsedStores[stype]:
            self.assignUsedStores[stype].remove(var)

    def varIsUsed(self, stype, var):
        if stype in self.assignUsedStores and var not in self.assignUsedStores[stype]:
            self.assignUsedStores[stype].append(var)

    def oldStack(self):
        self.assignUsedStores = self.assignStoresStack.pop()

    @property
    def optionsLen(self):
        return len(self.options)

    def saveOption(self):
        if self.selOption == -1:
            return

        if self.selOption >= len(self.options):
            myPrint('Error on selOption: ', self.focus_x, self.focus_y)
            raise Exception('selOption out of bounds: ', self.selOption, ' in ', self.options);

        opt = self.options[self.selOption]
        self.curLine.append(opt)

        if opt == 'END':
            self.endOfLine = 0

            self.depth -= 1
            if (self.depth < 0):
                self.currentReward = -1
                self.checkGameEnd()
            else:
                self.oldStack()

            return

        if self.curLine_previousIsStoreTypes:
            self.getNumStores(self.curLine_storeType, opt)
            self.currentReward += checkVarReward(self.curLine_storeType, opt)

            if self.curLine_argNum == 1:
                self.curLine_2argSt = self.curLine_storeType
                self.curLine_2argNum = opt

            self.curLine_argNum += 1

            if self.curLine_isAssign:  # is assignation
                self.varIsAssigned(self.curLine_storeType, opt)
                self.curLine_assignNum = opt
                self.curLine_isAssign = False
            else:
                if self.curLine_isCondition:
                    self.varUsedInIf.append(opt)

                self.varIsUsed(self.curLine_storeType, opt)

            self.curLine_previousIsStoreTypes = False
            return

        self.curLine_previousIsStoreTypes = opt in storeTypes

        if self.curLine_previousIsStoreTypes:
            self.currentReward += checkVarReward(opt)
            self.curLine_storeType = opt
            self.curLine_storeIsBool = opt.startswith('b')

        match self.focus_x:
            case 0:
                if self.curLine_previousIsStoreTypes:
                    self.curLine_assignIsBool = self.curLine_storeIsBool
                    self.curLine_assignType = self.curLine_storeType
                else:
                    # In case of condition
                    self.curLine_isCondition = True
                    self.endOfLine = 2
                    self.depth += 1

            case 2:
                if not self.curLine_previousIsStoreTypes:
                    self.curLine_opReq = 'NaN'

                    if opt in neutralOps:
                        self.curLine_opType = 'n'
                    elif self.curLine_assignIsBool:
                        self.curLine_opType = 'b'

                        if opt in boolOps_bool:
                            self.curLine_opReq = 'b'

                        if opt in boolOps_decimal:
                            self.curLine_opReq = 'd'

                    else:
                        self.curLine_opType = 'd'

                    if opt in oneArgOps:
                        self.endOfLine = 2

                    if opt in neutralOps:
                        self.endOfLine = 2

            case 3:
                self.curLine_argsAreBool = self.curLine_storeIsBool

    def goRight(self):
        self.inNewLine = False

        self.saveOption()

        self.focus_x += 1
        self.endOfLine -= 1

        self.totalReward *= 0.9

        if self.endOfLine < 0 or self.focus_x >= 7:
            self.newLine()

        self.loadOptions()

    def goDown(self, selOption=-1):

        if selOption == -1:
            self.selOption += 1
        else:
            self.selOption = selOption

        if self.selOption >= len(self.options):
            self.currentReward = -1
            self.selOption -= 1
        else:
            opt = self.options[self.selOption]
            if self.curLine_previousIsStoreTypes:
                self.currentReward += checkVarReward(self.curLine_storeType, opt)
            else:
                if opt in storeTypes:
                    self.currentReward += checkVarReward(opt)

    def countInstructionsElements(self, instructions=None):
        if instructions == None:
            instructions = self.instructions

        tot = 0
        for instr in instructions:
            tot += len(instr)

        return tot

    def get_state(self, until=0, instructions=None):
        if instructions == None:
            instructions = self.instructions

        startFrom = 0
        if drawLineNumber:
            startFrom = 1

        totElements = self.countInstructionsElements(instructions)

        if until == 0:
            until = totElements
        elif until < 0:
            until += totElements

        # Draw the current view
        canvas = []

        drawFocus = False

        elNumber = 0
        for y in range(0, self.num_lines): # or self.num_lines
            elNumber += 1
            if elNumber > until:
                break

            line = []

            instruction = []
            if y < len(instructions):
                instruction = instructions[y]

            if drawLineNumber:
                line.append([0, 1, y])

            for x in range(startFrom, self.ideWidth):
                xx = x - startFrom
                pixel = [0, 0, 0] if drawFocus else [0, 0]

                def setPixelVal(val):
                    isNum = isinstance(val, int)

                    if isNum:
                        pixel[1 if drawFocus else 0] = 1
                    else:
                        val = labels.index(val)

                    pixel[2 if drawFocus else 1] = val

                if elNumber == until or (until == totElements and y == self.focus_y and xx == self.focus_x):
                    if drawFocus:
                        pixel[0] = 1
                    setPixelVal(self.options[self.selOption])
                elif elNumber < until and xx < len(instruction):
                    setPixelVal(instruction[xx])

                line.append(pixel)

            canvas.append(line)

        return np.array(canvas, dtype=np.int32)

    def get_score(self):
        # Execute instructions
        self.current_score = -1

        if self.lastBoolVarAssign != self.lastCalculatedBoolVar and self.lastBoolVarAssign != None:
            self.extractWinnerVarInstructions()

            try:
                self.current_score = executeCycles(self.curWinnerInstructions, self.curWinnerInstructions_winner)
            except:
                self.reset()
                self.current_score = 1
                print("STUPID FAULT")

            self.lastCalculatedBoolVar = self.lastBoolVarAssign
            self.lastCalculatedScoreLine = len(self.instructions)

        myPrint("Score: ", self.current_score)

        if self.current_score == 0:
            self.checkGameEnd()

        pow = self.current_score **1
        return pow

    def reset(self):
        self.instructions = []
        self.curLine = ['WELCOME :)']

        self.focus_x = 0
        self.focus_y = -1

        self.game_over = False
        self.game_won = False
        self.maxScoreSurpass = False

        self.depth = 0

        self.current_score = 0

        self.selOption = -1
        self.endOfLine = 0

        self.currentReward = 0  # currently unused
        self.lineReward = 0
        self.totalReward = 0

        resetEngine()
        self.usedStores = {'d#': 0, 'b#': 0, 'd$': 0, 'b$': 0}
        for t in self.usedStores:
            self.usedStores[t] = len(stores[t])
        self.forkUsedStores()

        self.assignStoresStack = []
        self.initAssignStores()

        self.varUsedInIf = []

        self.curLine_isCondition = False

        # Init first line
        self.currentAction = -1
        self.goRight()

        self.lastBoolVarAssign = None
        self.lastCalculatedBoolVar = None
        self.lastCalculatedScoreLine = 0

        myPrint('Reset')

    def is_over(self):
        return self.game_over

    def is_won(self):
        return self.game_won

    ### Commands
    def right(self):
        self.play(0)


"""# Execute"""

### Execution
'''
res = executeCycles([["d$", 0, "MOD", "d#", 5, "d#", 12], ["b$", 0, "NOT", "b#", 0], ["b$", 1, "OR", "b#", 1, "b$", 0],
     ["d$", 1, "ADD", "d$", 0, "d#", 1], ["d$", 2, "ADD", "d$", 0, "d#", 16], ["b$", 2, "CMP", "d#", 16, "d$", 2],
     ["IF", "b$", 0], ["b$", 1, "GET", "d#", 5, "d#", 3], ["b$", 1, "NOT", "b#", 1], ["b$", 0, "OR", "b#", 0, "b$", 1],
     ["d$", 2, "DIV", "d#", 2, "d$", 0], ["b$", 1, "GT", "d$", 1, "d$", 2], ["END"]],2)
'''

drawFocus = False
els = 3 if drawFocus else 2

actions = 1
grid_size = 60
game = Calculon(grid_size)
input_shape = (grid_size, game.ideWidth, els)

totalDim = game.ideWidth * els

"""## Run"""
model = SuccessPredictorLSTM(totalDim, 840, 1, device=device).to(device=device)

if os.path.exists('outputs/model.pth'):
    model.load_state_dict(torch.load('outputs/model.pth'))


agent = Agent(model, input_shape, (1))
agent.train(game)
agent.play(game)
