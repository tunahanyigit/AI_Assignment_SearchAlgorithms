import math
import pandas as pd
import numpy
import random
import copy

global globalDepth
globalDepth = 0
global globalNodes
globalNodes = 0

class Node:
    def __init__(self, state, f=0, g=0, h=0):
        self.state = state
        self.f = f
        self.g = g
        self.h = h
    def __repr__(self):
        return "Node(" + repr(self.state) + ", f=" + repr(self.f) + \
               ", g=" + repr(self.g) + ", h=" + repr(self.h) + ")"

def displayPuzzleStates(state):
    """
    Displays the current state of a list by squaring it, and displaying it nicely.
    :param state: list of numbers
    :return: displays list as a square
    """
	# Substitute '' for 0, to make the puzzle look better
    display = [x if x != 0 else '' for x in state]
    # Iterate through the list, display tab-delimited
    for x in range(3):
        print(*display[(x * 3):(x * 3) + 3], sep='\t')

def getItemPositionInState(x, state):
    # get the index of the sepec number
    index = state.index(x)
    indexInList = int(index / 3), index % 3
    return indexInList

def discover_blank(state):
    """
    Finds location of 0 in puzzle
    :param state: current state of the puzzle
    :return: coordinates of 0
    """
    return getItemPositionInState(0, state)

def getPossibleActions(state):
    """
    In this function we will:
    Determine all the valid moves of 0 (the blank) state in a puzzle.

    if y != 0: then ("left")
    if y != 2: then ("right")
    if x != 0: then ("up")
    if x != 2: then ("down")
    """
    (x, y) = discover_blank(state)
    sqrt = int(math.sqrt(len(state)))
    actions = []
    if y != 0:
        actions.append(("left", 1))
    if y != sqrt - 1:
        actions.append(("right", 1))
    if x != 0:
        actions.append(("up", 1))
    if x != sqrt - 1:
        actions.append(("down", 1))
    return actions

def takeAction(state, action):
    """
    Moves the blank piece the appropriate direction, and returns the new state.
    :param state: Current state of puzzle
    :param action: Which direction to move the blank piece
    :return: New state after the 0 piece has moved.
    """
    childState = copy.copy(state)

    (x, y) = discover_blank(childState)
    reshapedState = numpy.reshape(childState, (3, 3))
    direction = action[0]
    if direction == "left":
        return (swap(reshapedState, (x, y), (x, y - 1)), 1)
    if direction == "right":
        return (swap(reshapedState, (x, y), (x, y + 1)), 1)
    if direction == "up":
        return (swap(reshapedState, (x, y), (x - 1, y)), 1)
    if direction == "down":
        return (swap(reshapedState, (x, y), (x + 1, y)), 1)

def swap(state, location1, location2):
    """
    swap two piece locations
    :param state: current state of puzzle
    :param location1: First location
    :param location2: Second location
    :return: new state with location1 and location2 swapped.
    """
    state[location1[0]][location1[1]], state[location2[0]][location2[1]] = state[location2[0]][location2[1]], \
                                                                           state[location1[0]][location1[1]]
    return list(state.flat)

def recursiveDepthLimitedSearch(startState, goalState, possibleActions, takeAction, depthLimit):
    """
    Recursive function which performs a depth-first search, with a depth limit.  Used by iterativeDS to repeatedly
    do depth-first searches at greater and greater depth.
    :param startState: starting state of graph
    :param goalState: desired completion state of graph
    :param possibleActions: function returning valid actions
    :param takeAction: function which implements those actions
    :param depthLimit: maximum depth to traverse for this pass
    :return: solution path of solution, 'cutoff' if we reach the depthLimit, 'failure' if not found.
    """
    
	#check if the current state is the goal state
    #check if the depth is 0, then cutoff detected
    #initially the cutoff-occurred is false	
	if goalState == startState:
		return 'cutoff'
	else:
		cutoffOccurred = False
	for action in possibleActions(startState):
		childState = takeAction(startState, action)
		globalNodes += 1
		
		#result <- call recursively the function by reducing the depth by 1 (depth -1)
		result = recursiveDepthLimitedSearch(childState[0], goalState, possibleActions, takeAction, depthLimit - 1)

		#check if the result is cutoff, then cutoff then cutoff-occurred? = true
		if result is 'cutoff':
			cutoffOccurred = True
		#else if result != failure then return result
		elif result is not 'failure':
			result.insert(0, childState)
			return result
	
	#if cutoff-occurred? then return cutoff else return failure	
	if cutoffOccurred:
		return 'cutoff'
	else:
		return 'failure'
		

def iterativeDS(startState, goalState, possibleActions, takeAction, maxDepth):
    """
    Performs an Iterative Deepening Search, using a depth-first search, with depth limit, to optimize the time to
    find a valid solution.
    :param startState: Starting state of the graph
    :param goalState:  Goal state of the graph
    :param possibleActions: Function listing actions graph can take
    :param takeAction: Function performing those actions
    :param maxDepth: Maximum depth for this search.  Search can return earlier, but not later than this depth.
    :return:
    """
	solutionPath = []
	solutionPath.append(startState)
	
	for depth in range(maxDepth):
		globalDepth = depth
		
		#result <- call the Depth limit Search function 
        result = recursiveDepthLimitedSearch(startState, goalState, possibleActions, takeAction, depth)

		#check if result is failure then return the failure statement
		#check if result is cutoff then return result
		if result is 'failure':
			return 'failure'
		if result is not 'cutoff':
			solutionPath.extend(result)
			return solutionPath
	
	return 'cutoff'
	
		
def ebf(nNodes, depth, precision=0.01):
    '''
        N: Total number of nodes processed.
        d: Depth at which the solution node was found.
        b*: Effective branching factor.
        N = b* + (b*)2 + ... + (b*)d

        No closed-form solution
        Solution 1: A Close Guess
                 N^(1/d)
        Solution 2:
                Requires N and d
                Select an error tolerance
                Select a high and low estimate
                Average the estimates to provide a guess for b*
                Calculate N' using the guess for b* and d
                If abs(N' - N) > error, modify the low or high estimate accordingly
                Otherwise, it is within the error, so return the guess for b*
        '''
    if (depth == 0):
        return 0

    first = 0
    last = nNodes - 1
    found = False
    midpoint = 0

    while first <= last and not found:
        midpoint = (first + last) / 2
        nPrime = ((1 - midpoint ** (depth + 1)) / (1 - midpoint)) if midpoint != 1 else 1
        if abs(nPrime - nNodes) < precision:
            found = True
        else:
            if nNodes < nPrime:
                last = midpoint - precision
            else:
                first = midpoint + precision

    return midpoint

def heuristicFunction1(state, goal):
    return 0

def heuristicFunction2(state, goal):
    return manhattanDistance(0, state, goal) 

def manhattanDistance(number, state, goal):
    statePos = getItemPositionInState(number, state)
    goalPos = getItemPositionInState(number, goal)
    return abs(statePos[0] - goalPos[0]) + abs(statePos[1] - goalPos[1])

def checkIfStateIsGoal(state, goal):
    return state == goal

def aStarSearch(startState, possibleActions, takeAction, checkIfStateIsGoal, hF):
    h = hF(startState)
    startNode = Node(state=startState, f=0 + h, g=0, h=h)
    return aStarSearchHelper(startNode, possibleActions, takeAction, checkIfStateIsGoal, hF, float('inf'))

def aStarSearchHelper(parentNode, possibleActions, takeAction, checkIfStateIsGoal, hF, fmax):
	
	#Initialize expanded to be an empty dictionary
	expanded = {}
	#Initialize unExpanded to be a list containing the startState node.
	unExpanded = {[parentNode]}
    
	#Its h value is calculated using hF, its g value is 0, and its f value is g+h.
	h = hF(parentNode.state)
	g = 0
	f = g + h
	temp = []
	
	#If startState is the goalState, return the list containing just startState and its f value to show the cost of the solution path.
	if checkIfStateIsGoal(parentNode.state, goalState):
        return [[parentNode.state], parentNode.f]
	
	#Repeat the following steps while unExpanded is not empty:
	while len(unExpanded) != 0:
		
		#Pop from the front of unExpanded to get the best (lowest f value) node to expand.
		node = unExpanded.pop(0)
		#Generate the children of this node.
		children = []

		
		#Update the g value of each child by adding the action's single step cost to this node's g value.
		#Calculate hF of each child.
		#Set f = g + h of each child.
		for action in getPossibleActions(node.state):
			state = takeAction(node.state, action)[0]
			h = hF(state)
            g = node.g + 1
	
            childnode = Node(state=state, f=g + h, g=g, h=h)
            children.append(childnode)
			
		#Add the node to the expanded dictionary, indexed by its state.
		expanded[str(node.state)] = [node, temp]
        temp = node.state
		
        for child in children:
            #Remove from children any nodes that are already either in expanded or unExpanded, unless the node in children has a lower f value.
			if str(child.state) in expanded and child.f > expanded[str(child.state)][0].f:
                children.pop(children.index(child))

            for i in unExpanded:
                if child.state == i.state:
                    if child.f < i.f:
                        unExpanded.pop(unExpanded.index(i))
                    else:
                        children.pop(children.index(child))
						
			#If goalState is in children:
			if checkIfStateIsGoal(child.state):
                solutionPath = []
                currentNode = child
                currentParent = node.state
                
				while True:
                    #Build the solution path as a list starting with goalState.
					solutionPath.append([currentNode.state, currentNode.f])
                    
					if not currentParent:
                        #Reverse the solution path list and return it.
						solutionPath.reverse()
                        print(solutionPath)
                        return solutionPath

                    currentNode = expanded[str(currentParent)][0]
                    currentParent = expanded[str(currentNode.state)][1]

                    alternativef = children[1].f if len(children) > 1 else float('inf')

                    globalDepth = min(fmax, alternativef)
                    globalNodes += 1
		
		#Insert the modified children list into the unExpanded list and sort by f values.
		for child in children:
            unExpanded.append(child)
        unExpanded = sorted(unExpanded, key=lambda x: x.f)


def testMySolution(goalState, heuristic_functions):
    heuristicFunction1 = heuristic_functions[0]
    heuristicFunction2 = heuristic_functions[1]
    heuristicFunction3 = heuristic_functions[2]

    global globalDepth
    globalDepth = 0
    global globalNodes
    globalNodes = 0

    idsSolutionPath1 = iterativeDS(startState, goalState, getPossibleActions, takeAction, 10)
    idsDepth1 = globalDepth
    idsNodes1 = globalNodes


    globalNodes = 0
    globalDepth = 0
    astarsolutionpath1 = aStarSearch(startState, getPossibleActions, takeAction, lambda s: checkIfStateIsGoal(s, goalState),
                                     lambda s: heuristicFunction1(s, goalState))
    astar1Depth1 = globalDepth
    astar1Nodes1 = globalNodes

    globalNodes = 0
    globalDepth = 0
    astarsolutionpath1 = aStarSearch(startState, getPossibleActions, takeAction, lambda s: checkIfStateIsGoal(s, goalState),
                                     lambda s: heuristicFunction2(s, goalState))
    astar1Depth2 = globalDepth
    astar1Nodes2 = globalNodes

    globalNodes = 0
    globalDepth = 0
    astarsolutionpath1 = aStarSearch(startState, getPossibleActions, takeAction, lambda s: checkIfStateIsGoal(s, goalState),
                                     lambda s: heuristicFunction3(s, goalState))
    astar1Depth3 = globalDepth
    astar1Nodes3 = globalNodes

    state1DataFrame = pd.DataFrame(
        [[idsDepth1, idsNodes1, "{0:.3f}".format(ebf(idsNodes1, idsDepth1))],
         [astar1Depth1, astar1Nodes1, "{0:.3f}".format(ebf(astar1Nodes1, astar1Depth1))],
         [astar1Depth2, astar1Nodes2, "{0:.3f}".format(ebf(astar1Nodes2, astar1Depth2))],
         [astar1Depth3, astar1Nodes3, "{0:.3f}".format(ebf(astar1Nodes3, astar1Depth3))]],
        index=["IDS", "A*h1", "A*h2", "A*h3"], columns=["Depth", "Nodes", "EBF"])
    print(state1DataFrame)
    print()
    print()

def printPath(startState, goalState, path):
    length = len(path)
    print("Path from")
    displayPuzzleStates(startState)
    print()
    print("To")
    displayPuzzleStates(goalState)
    print()
    print("is ", length, "nodes long:")
    for p in path:
        displayPuzzleStates(p)
        print()

def randomStartState(goalState, possibleActions, takeAction, nSteps):
    state = goalState
    for i in range(nSteps):
        state = takeAction(state, random.choice(possibleActions(state)))
    return state
if __name__ == "__main__":
    startState = [1,2,5,3,4,0,6,7,8]
    goalState = [0 ,1 ,2 ,3 ,4 ,5 ,6 ,7 ,8]

    displayPuzzleStates(startState)
    # # print(discover_blank(startState))
    # # print(getPossibleActions(startState))
    # # print(takeAction(startState, 'down'))
    path = iterativeDS(startState, goalState, getPossibleActions, takeAction, 5)
    printPath(startState, goalState, path)

    testMySolution(goalState, [heuristicFunction1, heuristicFunction2, heuristicFunction3])