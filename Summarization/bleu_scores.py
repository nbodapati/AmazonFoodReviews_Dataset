from __future__ import division
from collections import Counter,defaultdict
#this is an implementation of unigram bleu score.

class bleu_scores(object):
      def __init__(self,pred_list,val_list):
          #inputs: pred_list - list of lists of  predicted words
          #val_list - list of lists of val words.
          self.pred_list=pred_list
          self.val_list=val_list

      def unigram_bleu_score(self):
          score=0
          N=0
          for i in range(len(self.pred_list)):
              pred=self.pred_list[i].split()
              val=self.val_list[i].split()
              c_pred=Counter(pred)
              c_val=Counter(val)
              N+=len(c_val.keys())
              for key,value in c_pred.items():
                  score+=min(value,c_val[key])
          print("Score: ",score,"N: ",N)        
          bleu_score=score/N
          return bleu_score

      def bigram_bleu_score(self):
          score=0
          N=0
          for i in range(len(self.pred_list)):
              pred=self.pred_list[i].split()
              val=self.val_list[i].split()

              pred=list(zip(pred[:-1],pred[1:]))
              val=list(zip(val[:-1],val[1:]))

              c_pred=Counter(pred)
              c_val=Counter(val)
              N+=len(c_val.keys())
              for key,value in c_pred.items():
                  score+=min(value,c_val[key])
          print("Score: ",score,"N: ",N)        
          bleu_score=score/N
          return bleu_score

