#!/usr/bin/python

import sys; import math; import numpy as np
from perceptron import perceptron; from confus import confus
from linmach import linmach
#Si no se pasan 3 argumentos 
if len(sys.argv)!=4:
    print('Usage: %s <data> <alphas> <bs>' % sys.argv[0]);
    sys.exit(1);
#Cargas el dataset que se pasa como primer arg.
data=np.loadtxt(sys.argv[1]);
#Cargas las alphas(factor de aprendizaje) que se pasan como arg.
alphas=np.fromstring(sys.argv[2],sep=' ');
#Cargas las b(margen) que se pasa como arg.
bs=np.fromstring(sys.argv[3],sep=' ');
N,L=data.shape; #Numero de flias(muestras) y columnas(etqiquetas de clase)
D=L-1;  #Numero de etiquetas de clase
labs=np.unique(data[:,L-1]);    #Guardas todas las etiquetas de clase que hay ordenadas y filtradas
perm=np.random.permutation(N);  #Genera un array de N numeros random no repetidos
data=data[perm];    #Estos numeros generados en perm se añadaen a data para representar cada clase.
NTr=int(round(.5*N));       #50%xnumero de muestras
train=data[:NTr,:];     #Obtienes 50% de las muestras para ser entrenadas
M=N-NTr;    #Nuevo valor de muestras que han sido entrenadas
test=data[NTr:,:];      #Para ver el error, te guardas el otro 30% del dataset
print('#   a         b     E    k   Ete  Ete(%)   Ite(%)');
print('#--------   ------ ---- --- ----- ------ -------');
for a in alphas:    #Bucle que prueba el algoritmo perceptron con todas las alphas que se han pasado 
    for b in bs:        #Bucle que prueba el algoritmo perceptron con todas las b que se han pasado
        w,E,k=perceptron(train,b,a,100);    #LLama al algoritmo perceptor con el dataset reducido, en w se guardan los vectores de clases
        rl=np.zeros((M,1));     #Matriz de zeros con 1 columna y m(nº de muetstras) filas
        for n in range(M):      #Para toda muestra
            rl[n]=labs[linmach(w,np.concatenate(([1],test[n,:D])))];    
        Ete,m=confus(test[:,L-1].reshape(M,1),rl);  #Devuelve para las muestras no entrenadas el nº de errores y para cada clase la matriz de errores
    per=Ete/M;      #Holdout=nºerr/M  
    r=1.96*math.sqrt(per*(1-per)/M);    #Intervalo de confianza 
    print('%8.1f %8.1f %3d %4d %5d %6.1f [%.1f, %.1f]' % (a,b,E,k,Ete,per*100,(per-r)*100,(per+r)*100)); 
