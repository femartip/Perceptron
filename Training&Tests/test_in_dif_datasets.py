#!/usr/bin/python
#Script that compares the performance of the Perceptron with different data sets 

import sys; import math; import numpy as np
from perceptron import perceptron; from confus import confus
from linmach import linmach

#Si no se pasan 3 argumentos 
if len(sys.argv)!=3:
    print('Usage: %s <alphas> <bs>' % sys.argv[0]);
    sys.exit(1);
#Cargas las alphas(factor de aprendizaje) que se pasan como arg.
alphas=np.fromstring(sys.argv[1],sep=' ');
#Cargas las b(margen) que se pasa como arg.
bs=np.fromstring(sys.argv[2],sep=' ');
datas = ['OCR_14x14', 'expressions', 'gauss2D', 'gender','videos'];
print('#   tarea      Ete(%)   Ite(%)');
print('#--------- ---------- ------------');
for d in datas:
    data = np.loadtxt(d);
    N,L=data.shape; #Numero de flias(muestras) y columnas(etqiquetas de clase)
    D=L-1;  #Numero de etiquetas de clase
    labs=np.unique(data[:,L-1]);    #Guardas todas las etiquetas de clase que hay ordenadas y filtradas
    perm=np.random.permutation(N);  #Genera un array de N numeros random no repetidos
    data=data[perm];    #Estos numeros generados en perm se añadaen a data para representar cada clase.
    NTr=int(round(.7*N));       #70%xnumero de muestras
    train=data[:NTr,:];     #Obtienes 70% de las muestras para ser entrenadas
    M=N-NTr;    #Nuevo valor de muestras que han sido entrenadas
    test=data[NTr:,:];      #Para ver el error, te guardas el otro 30% del dataset
    perM=1000;
    rM=1000;
    for a in alphas:    #Bucle que prueba el algoritmo perceptron con todas las alphas que se han pasado 
        for b in bs:        #Bucle que prueba el algoritmo perceptron con todas las b que se han pasado
            w,E,k=perceptron(train,b,a);    #LLama al algoritmo perceptor con el dataset reducido, en w se guardan los vectores de clases
            rl=np.zeros((M,1));     #Matriz de zeros con 1 columna y m(nº de muetstras) filas
            for n in range(M):      #Para toda muestra
                rl[n]=labs[linmach(w,np.concatenate(([1],test[n,:D])))];    
            Ete,m=confus(test[:,L-1].reshape(M,1),rl);  #Devuelve para las muestras no entrenadas el nº de errores y para cada clase la matriz de errores
        per=Ete/M;      #Holdout=nºerr/M  
        r=1.96*math.sqrt(per*(1-per)/M);    #Intervalo de confianza
        if(perM > per ): perM=per;rM=r; 
    print('%s %9.1f [%.1f, %.1f]' % (d,perM*100,(perM-rM)*100,(perM+rM)*100)); 
