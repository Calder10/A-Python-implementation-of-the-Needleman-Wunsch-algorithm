"""
    Implementazione dell'algoritmo Needleman-Wunsch
"""

# Importo le librerie necessarie
import numpy as np
from sklearn.preprocessing import MinMaxScaler
#from Bio.SubsMat.MatrixInfo import blosum62
import matplotlib.pyplot as plt
import seaborn as sns
import os
import importlib

def blos_score(s1,s2,ms):
    """
        La funzione prende in input le 2 sequenze e costruisce la matrice con i punteggi della matrice BLOSUM 62
    """
    pkg = importlib.import_module('Bio.SubsMat.MatrixInfo')
    X=getattr(pkg,ms)
    #X=blosum62
    A=np.zeros((len(s2),len(s1)))
    for j in range (len(s2)):
        for i in range (len(s1)):
            if (s2[j],s1[i]) in X :
                A[j][i]=X[(s2[j],s1[i])]
            else:
                A[j][i]=X[(s1[i],s2[j])]
    return A

def my_min(M):
    min=A[0][0]
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if (M[i][j]<min):
                min=M[i][j]
    return min

def my_max(M):
    max=0
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if (M[i][j]>=max):
                max=M[i][j]
                i_max=i
                j_max=j
    return (max,i_max,j_max)

def norm_matrix(A):
    """
        La funzione prende in input la matrice con i punteggi Blosum62 e gli somma il minimo
        per rendere tutti i valori maggiori di 0.
    """
    m=my_min(A)
    for i in range (0,len(s2)):
        for j in range(0,len(s1)):
            A[i][j]=A[i][j]-m
    return A



def nw_base(An,d,s1,s2):
    """
    La funzione score prende in input:
        1-An (Matrice con i valori blosum62 normalizzata)
        2-S (Matrice dove salverò gli score)
        3-P (Matrice dove per ogni cella memorizzo la direzione di provenienza)
        4-d (valore di delta scelto dall'utente)
    """

    """
        Assegno gli score alla prima riga. Nella matrice dei percorsi inserisco:
            D-Se assegno lo stesso punteggio della matrice blosum
            Left-Se il punteggio l'ho ricavato venendo da sinistra
    """
    S=np.zeros((len(s2),len(s1)))
    P = np.empty((len(s2),len(s1)),dtype=np.dtype('U25'))
    S[0][0]=An[0][0]
    P[0][0]='D'
    i=0
    j=1
    for j in range(1,S.shape[1]):
        v=[An[i][j],(An[i][j-1]-d)]
        S[i][j]=max(v)
        aus=v.index(max(v))
        if aus==0:
            P[i][j]='D'
        else:
            P[i][j]='Left'


    """
        Assegno gli score alla prima colonna. Nella matrice dei percorsi inserisco:
            D-Se assegno lo stesso punteggio della matrice blosum
            Up-Se il punteggio l'ho ricavato venendo dall'alto
    """
    j=0
    i=1
    for i in range(1,S.shape[0]):
        v=[An[i][j],(An[i-1][j]-d)]
        S[i][j]=max(v)
        aus=v.index(max(v))
        if aus==0:
            P[i][j]='D'
        else:
            P[i][j]='Up'


    """
        Assegno gli score a tutto il resto della matrice. Nella matrice dei percorsi inserisco:
            Diag-Se il punteggio l'ho ricavato venendo dalla diagonale
            Left-Se il punteggio l'ho ricavato venendo da sinistra
            Up-Se il punteggio l'ho ricavato venendo dall'alto
    """
    i=1
    j=1
    for i in range(1,S.shape[0]):
        for j in range(1,S.shape[1]):
            #print(str(i)+" "+str(j))
            v=[(An[i][j]+S[i-1][j-1]),(S[i][j-1]-d),(S[i-1][j]-d)]
            #print(v)
            S[i][j]=max(v)
            aus=v.index(max(v))
            if aus==0:
                P[i][j]="Diag"
            if aus==1:
                P[i][j]="Left"
            if aus==2:
                P[i][j]="Up"

    """
        Ricostruzione dell'allineamento.
    """
    s1_a=[]
    s2_a=[]

    #i,j = np.where(S == np.amax(S))
    score,i,j=my_max(S)
    #i=i[0]
    #j=j[0]
    #score=0
    j_max=j
    i_max=i
    X_P=[]
    Y_P=[]
    score=S.max()
    while True:
        if P[i][j]=='Diag':
            X_P.append(i)
            Y_P.append(j)
            i=i-1
            j=j-1
        elif (P[i][j]=='Up'):
            X_P.append(i)
            Y_P.append(j)
            i=i-1

        elif (P[i][j]=='Left'):
            X_P.append(i)
            Y_P.append(j)
            j=j-1

        else:
            X_P.append(i)
            Y_P.append(j)
            break

    for p in range (0,len(X_P)):
        ii=X_P[p]
        jj=Y_P[p]
        if P[ii][jj]=="Diag":
            s1_a.append(s1[jj])
            s2_a.append(s2[ii])
        elif P[ii][jj]=="Up":
            s1_a.append("-")
            s2_a.append(s2[ii])
        elif P[ii][jj]=="Left":
            s2_a.append("-")
            s1_a.append(s1[jj])
        elif P[ii][jj]=="D":
            s1_a.append(s2[ii])
            s2_a.append(s1[jj])


    if i !=0:
        while True:
            s2_a.append(s2[i-1])
            s1_a.append(" ")
            i=i-1
            if i==0:
                break
    if j !=0:
        while True:
            s1_a.append(s1[j-1])
            s2_a.append(" ")
            j=j-1
            if j==0:
                break

    s1_a.reverse()
    if j_max<len(s1):
        for p in range(j_max+1,len(s1)):
            s1_a.append(s1[p])

    s2_a.reverse()
    if i_max<len(s2):
        for p in range(i_max+1,len(s2)):
            s2_a.append(s2[p])


    s1_a=' '.join(map(str,s1_a))
    s2_a=' '.join(map(str,s2_a))

    return S,s1_a,s2_a,score,X_P,Y_P


def nw_gap_linear(s1,s2,A,d):
    F=np.zeros((len(s2)+1,len(s1)+1))
    P = np.empty((len(s2)+1,len(s1)+1),dtype=np.dtype('U25'))
    F[0][0]=0
    P[0][0]="D"

    for j in range(1,len(s1)+1):
        F[0][j]=-d*(j)
        P[0][j]="Left"

    for i in range(1,len(s2)+1):
        F[i][0]=-d*(i)
        P[i][0]="Up"

    i=1
    j=1

    for i in range(1,len(s2)+1):
        for j in range(1,len(s1)+1):
            v=[(A[i-1][j-1]+F[i-1][j-1]),(F[i][j-1]-d),(F[i-1][j]-d)]
            F[i][j]=max(v)
            aus=v.index(max(v))
            if aus==0:
                P[i][j]="Diag"
            if aus==1:
                P[i][j]="Left"
            if aus==2:
                P[i][j]="Up"

    s1_a=[]
    s2_a=[]

    X_P=[]
    Y_P=[]

    i=len(s2)
    j=len(s1)
    while True:
        if P[i][j]=="D":
            break
        if P[i][j]=='Diag':
            X_P.append(i)
            Y_P.append(j)
            s1_a.append(s1[j-1])
            s2_a.append(s2[i-1])
            i=i-1
            j=j-1
        elif (P[i][j]=='Up'):
            X_P.append(i)
            Y_P.append(j)
            s1_a.append("-")
            s2_a.append(s2[i-1])
            i=i-1

        elif (P[i][j]=='Left'):
            X_P.append(i)
            Y_P.append(j)
            s2_a.append("-")
            s1_a.append(s1[j-1])
            j=j-1


    s1_a.reverse()
    s2_a.reverse()
    s1_a=' '.join(map(str,s1_a))
    s2_a=' '.join(map(str,s2_a))
    F=F[1:len(s2)+1,1:len(s1)+1]
    score=F[len(s2)-1][len(s1)-1]
    return s1_a,s2_a,score,X_P,Y_P,F


def nw_gop_gep(s1,s2,A,d,g):
    S=np.zeros((len(s2),len(s1)))
    P = np.empty((len(s2),len(s1)),dtype=np.dtype('U25'))
    S[0][0]=A[0][0]
    P[0][0]='D'

    i=0
    j=1

    # Riempiemnto della prima riga
    for j in range (1,len(s1)):
        l=0
        z=0
        if(P[i][j-1]=="Left"):
            z=j-1
            while(P[i][z]=='Left'):
                l=l+1
                z=z-1
        v=[A[i][j],(A[i][j-1]-d-(g*(l-1)))]
        S[i][j]=max(v)
        aus=v.index(max(v))
        if aus==0:
            P[i][j]='D'
        else:
            P[i][j]='Left'

    i=1
    j=0
    for i in range (1,len(s2)):
        l=0
        z=0
        if(P[i-1][j]=="Up"):
            z=i-1
            while(P[z][j]=='Up'):
                l=l+1
                z=z-1
        v=[A[i][j],(A[i-1][j]-d-g*(l-1))]
        S[i][j]=max(v)
        aus=v.index(max(v))
        if aus==0:
            P[i][j]='D'
        else:
            P[i][j]='Up'

    i=1
    j=1
    for i in range(1,len(s2)):
        for j in range(1,len(s1)):
            l_sx=0
            l_up=0
            z=0
            if(P[i][j-1]=="Left"):
                z=j-1
                while(P[i][z]=='Left'):
                    l_sx=l_sx+1
                    z=z-1

            if(P[i-1][j]=="Up"):
                z=i-1
                while(P[z][j]=='Up'):
                    l_up=l_up+1
                    z=z-1

            v=[(A[i][j]+S[i-1][j-1]),(S[i][j-1]-d-(g*(l_sx-1))),(S[i-1][j]-d-(g*(l_up-1)))]
            S[i][j]=max(v)
            aus=v.index(max(v))
            if aus==0:
                P[i][j]="Diag"
            if aus==1:
                P[i][j]="Left"
            if aus==2:
                P[i][j]="Up"


    s1_a=[]
    s2_a=[]

    """
    i,j = np.where(S == np.amax(S))
    i=i[0]
    j=j[0]
    score=0
    """
    score,i,j=my_max(S)
    j_max=j
    i_max=i
    X_P=[]
    Y_P=[]
    while True:
        if P[i][j]=='Diag':
            X_P.append(i)
            Y_P.append(j)
            i=i-1
            j=j-1
        elif (P[i][j]=='Up'):
            X_P.append(i)
            Y_P.append(j)
            i=i-1

        elif (P[i][j]=='Left'):
            X_P.append(i)
            Y_P.append(j)
            j=j-1

        else:
            X_P.append(i)
            Y_P.append(j)
            break

    for p in range (0,len(X_P)):
        ii=X_P[p]
        jj=Y_P[p]
        if P[ii][jj]=="Diag":
            s1_a.append(s1[jj])
            s2_a.append(s2[ii])
        elif P[ii][jj]=="Up":
            s1_a.append("-")
            s2_a.append(s2[ii])
        elif P[ii][jj]=="Left":
            s2_a.append("-")
            s1_a.append(s1[jj])
        elif P[ii][jj]=="D":
            s1_a.append(s2[ii])
            s2_a.append(s1[jj])

    if i !=0:
        while True:
            s2_a.append(s2[i-1])
            s1_a.append(" ")
            i=i-1
            if i==0:
                break
    if j !=0:
        while True:
            s1_a.append(s1[j-1])
            s2_a.append(" ")
            j=j-1
            if j==0:
                break

    s1_a.reverse()
    if j_max<len(s1):
        for p in range(j_max+1,len(s1)):
            s1_a.append(s1[p])

    s2_a.reverse()
    if i_max<len(s2):
        for p in range(i_max+1,len(s2)):
            s2_a.append(s2[p])
            


    s1_a=' '.join(map(str,s1_a))
    s2_a=' '.join(map(str,s2_a))
    return S,s1_a,s2_a,score,X_P,Y_P



def plot(S,s1,s2,x,y,g,f,ms):
    """
        La funzione prende in input:
        S->Matrice con i relativi score
        s1,s2->Le 2 sequenze che si stanno allineando
        x,y-> acisse e ordinate dei punti che formano il percorso dell'allineamento

        Fornisce in output un grafico che ci mostra il percorso migliore ottenuto
    """
    if f==0 and g==0:
        t=s1+"_"+s2+" Delta="+str(d)+" MS="+ms
    else:
        t=s1+"_"+s2+" Delta="+str(d)+" Gamma="+str(g)+" MS=" +ms
    plt.figure(figsize=(10,10))
    plt.title(t)
    plt.plot(y,x,'r')
    f=sns.heatmap(S,annot=True,cmap="Blues_r",linewidths=.5,xticklabels=s1,yticklabels=s2)
    plt.show()

"""
TFDERILGVQTYWAECLA
QTFWECIKGDNATY

QERTY
QERS

AFGIVHKLIVS
AFGIHKIVS

"""
while True:
    print("""
            1 - Needleman Wunsch Base
            2 - Needleman Wunsch Gap Linear
            3 - Needleman Wunsch Gop Gep
            0 - Esci
          """)
    x=input()
    x=int(x)
    if x==0:
        os.system("clear")
        break
    if x==1:
        os.system("clear")
        print("Inserisci il nome della matrice di sostituzione da utilizzare-->")
        ms=input()
        print("Inserisci la sequenza S1--->")
        s1=input()
        print("Inserisci la sequenza S2--->")
        s2=input()
        A=blos_score(s1,s2,ms)
        A=norm_matrix(A)
        print("Inserisci il valore di delta--->")
        d=input()
        d=float(d)
        (S,s1_a,s2_a,score,X_P,Y_P)=nw_base(A,d,s1,s2)
        os.system("clear")
        print("Allineamento:"+"\n"+s1_a +" \n"+ s2_a +"\n"+ "Score: "+str(score))
        plot(S,s1,s2,X_P,Y_P,0,0,ms)
    if x==2:
        os.system("clear")
        print("Inserisci il nomde della matrice di sostituzione da utilizzare-->")
        ms=input()
        print("Inserisci la sequenza S1--->")
        s1=input()
        print("Inserisci la sequenza S2--->")
        s2=input()
        A=blos_score(s1,s2,ms)
        A=norm_matrix(A)
        print("Inserisci il valore di delta--->")
        d=input()
        d=float(d)
        s1_a,s2_a,score,X_P,Y_P,F=nw_gap_linear(s1,s2,A,d)
        os.system("clear")
        print("Allineamento:"+"\n"+s1_a +" \n"+ s2_a +"\n"+ "Score: "+str(score))
        plot(F,s1,s2,X_P,Y_P,0,0,ms)
    if x==3:
        os.system("clear")
        print("Inserisci il nome della matrice di sostituzione da utilizzare-->")
        ms=input()
        print("Inserisci la sequenza S1--->")
        s1=input()
        print("Inserisci la sequenza S2--->")
        s2=input()
        A=blos_score(s1,s2,ms)
        A=norm_matrix(A)
        print("Inserisci il valore di delta--->")
        d=input()
        d=float(d)
        print("Inserisci il valore di gamma-->")
        g=input()
        g=float(g)
        S,s1_a,s2_a,score,X_P,Y_P=nw_gop_gep(s1,s2,A,d,g)
        os.system("clear")
        print("Allineamento:"+"\n"+s1_a +" \n"+ s2_a +"\n"+ "Score: "+str(score))
        plot(S,s1,s2,X_P,Y_P,g,1,ms)
    os.system("clear")
