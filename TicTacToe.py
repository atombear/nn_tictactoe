# Play Tic-Tac-Toe
from __future__ import division
import scipy
import random
import copy
import flattenTest
import pickle
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer, BiasUnit
from pybrain.structure import FullConnection
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
#from pybrain.tools.shortcuts import buildNetwork

#curdir = '/Users/alexanderpapageorge/Documents/PythonScripts/edu/'

class TicTacToe_Board(object):
    
    def __init__(self, net_loc = None, DS_loc = None, num_hidden = 1):
        self.Board = scipy.array([[0 for i in range(3)] for j in range(3)])
        self.isWinner = False
        self.num_hidden = num_hidden
        self.gam = .3
        
        # Initialize Network
        if net_loc is not None:
            self.nn = pickle.load(open(net_loc,'rb'))
        else:
            self.ResetNN()
        
        # Initialize Training Set
        if DS_loc is not None:
            self.DS = pickle.load(open(DS_loc,'rb'))
        else:
            self.DS = SupervisedDataSet(9,1)
            
        self.Online_DS = SupervisedDataSet(9,1)

    def GameOver(self,board):
        if 3 in abs(board.sum(0)):
            self.isWinner = True
            return True
        if 3 in abs(board.sum(1)):
            self.isWinner = True
            return True
        if 3 == abs(board.trace()):
            self.isWinner = True
            return True
        if 3 == abs(board[::-1].trace()):
            self.isWinner = True
            return True
        if not list(scipy.concatenate(scipy.where(board == 0))):
            return True
        
        return False
        
    def PlaceX(self,board,pos):
        board[pos[0],pos[1]] = 1
        
    def PlaceO(self,board,pos):
        board[pos[0],pos[1]] = -1
        
    def MoveAllowed(self,pos):
        if self.Board[pos[0],pos[1]] == 0:
            return True
        else: return False
        
    def Moves(self,board):
        temp = scipy.where(board == 0)
        return [(temp[0][i],temp[1][i]) for i in range(len(temp[0]))]
        
    def ResetGame(self):
        self.Board = scipy.array([[0 for i in range(3)] for j in range(3)])
        self.isWinner = False
        
    def SaveNet(self,save_dir):
        pickle.dump(self.nn,open(save_dir + 'ttt_net.p','wb'))
        pickle.dump(self.DS,open(save_dir + 'ttt_DS.p','wb'))
        
    def ResetDS(self):
        self.DS.clear()
        
    def ResetNN(self):
        if self.num_hidden == 1:
            self.nn = FeedForwardNetwork(name = 'TicTacToe Net')
            inLayer = LinearLayer(9, name = 'Input Layer')
            hiddenLayer1 = SigmoidLayer(12, name = 'Hidden Layer 1')
            outLayer = LinearLayer(1, name = 'Output Layer')
            bias = BiasUnit(name = 'Bias Layer')
            self.nn.addInputModule(inLayer)
            self.nn.addModule(hiddenLayer1)
            self.nn.addOutputModule(outLayer)
            self.nn.addModule(bias)
            in_to_h1 = FullConnection(inLayer, hiddenLayer1, name = 'in_to_h1')
            bias_to_h1 = FullConnection(bias, hiddenLayer1, name = 'bias_to_h1')
            h1_to_out = FullConnection(hiddenLayer1, outLayer, name = 'h1_to_out')
            self.nn.addConnection(in_to_h1)
            self.nn.addConnection(bias_to_h1)
            self.nn.addConnection(h1_to_out)
            self.nn.sortModules()
        
        if self.num_hidden == 2:
            self.nn = FeedForwardNetwork(name = 'TicTacToe Net')
            inLayer = LinearLayer(9, name = 'Input Layer')
            hiddenLayer1 = SigmoidLayer(7, name = 'Hidden Layer 1')
            hiddenLayer2 = SigmoidLayer(5, name = 'Hidden Layer 2')
            outLayer = LinearLayer(1, name = 'Output Layer')
            bias = BiasUnit(name = 'Bias Layer')
            self.nn.addInputModule(inLayer)
            self.nn.addModule(hiddenLayer1)
            self.nn.addModule(hiddenLayer2)
            self.nn.addOutputModule(outLayer)
            self.nn.addModule(bias)
            in_to_h1 = FullConnection(inLayer, hiddenLayer1, name = 'in_to_h1')
            bias_to_h1 = FullConnection(bias, hiddenLayer1, name = 'bias_to_h1')
            h1_to_h2 = FullConnection(hiddenLayer1, hiddenLayer2, name = 'h1_to_h2')
            bias_to_h2 = FullConnection(bias, hiddenLayer2, name = 'bias_to_h2')
            h2_to_out = FullConnection(hiddenLayer2, outLayer, name = 'h2_to_out')
            self.nn.addConnection(in_to_h1)
            self.nn.addConnection(bias_to_h1)
            self.nn.addConnection(h1_to_h2)
            self.nn.addConnection(bias_to_h2)
            self.nn.addConnection(h2_to_out)
            self.nn.sortModules()
            
    def Retrain(self,N):
        self.ResetNN()
        trainer = BackpropTrainer(self.nn, dataset = self.DS, momentum=0.1, verbose=True, weightdecay=0.01)
        trainer.trainEpochs(N)
        
    def Moretrain(self,Ng,N):
        self.ResetDS()
        self.TrainNN_Random(Ng)
        trainer = BackpropTrainer(self.nn, dataset = self.DS, momentum=0.1, verbose=True, weightdecay=0.01)
        trainer.trainEpochs(N)
        
    def make_o_move_nn(self):
        o_moves = self.Moves(self.Board)
        o_moves_val = scipy.zeros(len(o_moves))
        for ni,i in enumerate(o_moves):
            temp_board = copy.deepcopy(self.Board)
            self.PlaceO(temp_board,i)
            o_moves_val[ni] = self.nn.activate(flattenTest.flatten(temp_board.tolist()))
            
        o_move = o_moves[scipy.where(o_moves_val == o_moves_val.min())[0][0]]
        self.PlaceO(self.Board,o_move)
        
    def Tree_Search(self,board,depth,o_move):
        if (depth == 0) or (len(scipy.where(board==0)[0])<depth):
            if self.GameOver(board):
                return 1
            if o_move:
                o_moves = self.Moves(board)
                o_moves_val = scipy.zeros(len(o_moves))
                for ni,i in enumerate(o_moves):
                    temp_board = copy.deepcopy(board)
                    self.PlaceO(temp_board,i)
                    o_moves_val[ni] = self.nn.activate(flattenTest.flatten(temp_board.tolist()))
                return o_moves_val.min()
        v = 10
        best_move = None
        if o_move:
            for move in self.Moves(board):
                temp_board = copy.deepcopy(board)
                self.PlaceO(temp_board,move)
                #v = min(v,self.Tree_Search(temp_board,depth-1,False))
                new_v = self.Tree_Search(temp_board,depth-1,False)
                if new_v < v:
                    v = new_v
                    best_move = move
            return v, best_move
        else:
            for move in self.Moves(board):
                temp_board = copy.deepcopy(board)
                self.PlaceX(temp_board,move)
                v = min(v,self.Tree_Search(temp_board,depth-1,True))
            return v
            
        
    def make_o_move_nn_Tree_Search(self):
        val, o_move = self.Tree_Search(self.Board,2,True)
        self.PlaceO(self.Board,o_move)

    def PlayRandom(self):
        while not self.GameOver(self.Board):
            x_move_str = raw_input('X, make your move ')
            x_move = (x_move_str[0],x_move_str[2])
            if self.MoveAllowed(x_move):
                self.PlaceX(self.Board,x_move)
                if not self.GameOver(self.Board):
                    o_moves = self.Moves(self.Board)
                    o_move = o_moves[random.randint(0,len(o_moves)-1)]
                    self.PlaceO(self.Board,o_move)
            
            print self.Board
        if self.isWinner:
            if self.Board.sum() == 1:
                print 'X wins!'
            if self.Board.sum() == 0:
                print 'O wins!'
        else: print 'Draw!'
            
        self.ResetGame()
        
    def PlayNN(self,learn = False):
        Game_Boards = []
        #Game_Boards.append(flattenTest.flatten(self.Board.tolist()))
        while not self.GameOver(self.Board):
            x_move_str = raw_input('X, make your move ')
            x_move = (x_move_str[0],x_move_str[2])
            if self.MoveAllowed(x_move):
                self.PlaceX(self.Board,x_move)
                #Game_Boards.append(flattenTest.flatten(self.Board.tolist()))
                if not self.GameOver(self.Board):
                    self.make_o_move_nn()
                    for k in range(4):
                        Game_Boards.append(flattenTest.flatten(scipy.rot90(self.Board,k).tolist()))
                        Game_Boards.append(flattenTest.flatten(scipy.rot90(self.Board[::-1],k).tolist()))
            
            print self.Board
        if self.isWinner:
            if self.Board.sum() == 1:
                print 'X wins!'
                End_Game = 1
            if self.Board.sum() == 0:
                print 'O wins!'
                End_Game = -1
        else:
            print 'Draw!'
            End_Game = 0
            
        # Training step
        if learn:
            for ni,Board in enumerate(Game_Boards):
                self.DS.addSample(Board,(ni+1)*End_Game)
            trainer = BackpropTrainer(self.nn, dataset = self.DS, momentum=0.1, verbose=True, weightdecay=0.01)
            trainer.trainEpochs(10)
            
        self.ResetGame()
        
    def TrainNN_Random(self,N):
        for i in range(N):
            Game_Boards = []
            while not self.GameOver(self.Board):
                x_moves = self.Moves(self.Board)
                x_move = x_moves[random.randint(0,len(x_moves)-1)]
                self.PlaceX(self.Board,x_move)
                if not self.GameOver(self.Board):
                    #self.make_o_move_nn()
                    self.make_o_move_nn_Tree_Search()
                    for k in range(4):
                        Game_Boards.append(tuple(flattenTest.flatten(scipy.rot90(self.Board,k).tolist())))
                        Game_Boards.append(tuple(flattenTest.flatten(scipy.rot90(self.Board[::-1],k).tolist())))
            
            Game_Boards = Game_Boards[-8::]
            Game_Boards = list(set(Game_Boards))

            if self.isWinner:
                if self.Board.sum() == 1:
                    End_Game = 1
                if self.Board.sum() == 0:
                    End_Game = -1
            else:
                End_Game = 0
                
            # Training step
            
            self.Online_DS.clear()
            
            #for ni,Board in enumerate(Game_Boards):
            #    self.DS.addSample(Board,(int(ni/8)+1)**2*End_Game)
            #    self.Online_DS.addSample(Board,(int(ni/8)+1)**2*End_Game)
                            
            for ni,Board in enumerate(Game_Boards[::-1]):
                self.DS.addSample(Board,self.gam**(int(ni/8))*End_Game)
                self.Online_DS.addSample(Board,self.gam**(int(ni/8))*End_Game)
                
            trainer = BackpropTrainer(self.nn, dataset = self.DS, momentum=0.1, verbose=True, weightdecay=0.01)            
            
            #trainer = BackpropTrainer(self.nn, dataset = self.Online_DS, momentum=0.1, verbose=True, weightdecay=0.01)
            
            trainer.trainEpochs(10)

            self.ResetGame()
        
        
    def PlayNetRandom(self,games):
        wins = 0
        losses = 0
        draws = 0
        
        for i in range(games):
            while not self.GameOver(self.Board):
                x_moves = self.Moves(self.Board)
                x_move = x_moves[random.randint(0,len(x_moves)-1)]
                self.PlaceX(self.Board,x_move)
                if not self.GameOver(self.Board):
                    self.make_o_move_nn()
                
            if self.isWinner:
                if self.Board.sum() == 1:
                    losses+=1
                if self.Board.sum() == 0:
                    wins+=1
            else:
                draws+=1
                
            self.ResetGame()
            
        return (wins,losses,draws)
        
#Tie: 4
#Lose: 2,3,4
#Win: 3,4
