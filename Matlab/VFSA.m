% Very Fast simulated anhealing
% Implemented by Sergio Abreo
% Computational Geophysics
% October 2018

clear
close all
clc

N=2; % numero de parametros
M=1; % numero de observaciones

intervalo=5;  %32 ackl, 100 boha1,   5 drop, 10 matia
paso=0.5;        % 1 ackl,   1 boha1, 0.5 drop,  1 matia

[X,Y]=meshgrid(-intervalo:paso:intervalo, -intervalo:paso:intervalo);

[k,k]=size(X);

for i=1:k
    for j=1:k
        %Z(i,j)=ackley( [ X(i,j) , Y(i,j) ] );
        %Z(i,j)=boha1( [ X(i,j) , Y(i,j) ] );
        Z(i,j)=drop( [ X(i,j) , Y(i,j) ] );
        %Z(i,j)=matya( [ X(i,j) , Y(i,j) ] );
    end
end

figure(1)
surf(X,Y,Z)
alpha 0.05
grid on
hold on
%view(-17,18)%ackley
%view(-69,67)%boha1
view(-161,29)%drop
%view(-220,32)%matya

for j=1:N*M
    T(j)=10;% Temperatura inicial
    c(j)= 0.1;% coeficientes de cada parametro
    maxi(j)=intervalo; % maximo valor del parametro
    mini(j)=-intervalo;% minimo valor del parametro
    m0(j)=(rand*2 -1)*intervalo;% Punto inicial
end

%Em0=ackley(m0);
%Em0=boha1(m0);
Em0=drop(m0);
%Em0=matya(m0);


% Punto inicial
plot3(m0(1),m0(2),Em0,'b+');
k=0;
m1=m0;
while T>0 % loop over the temperature T
    k=k+1;
    for i=1:100 % loop over a number of random moves/temperature
        for j=1:N*M % loop over model parameters
            u=rand;
            yi=sign(u-0.5)*T(j)*((1+T(j))^(abs(2*u-1))-1);
            m1(j)=m1(j)+yi*(maxi(j)-mini(j));
            if (m1(j)< mini(j))
                m1(j)= mini(j);
            elseif (m1(j)>= maxi(j))
                m1(j)= maxi(j);
           % else
           %     sprintf('%s','OK')
            end
        end
        %Em1=ackley(m1);
        %Em1=boha1(m1);
        Em1=drop(m1);
        %Em1=matya(m1);
        
        Delta_e=Em1-Em0;
        P=exp(-Delta_e/T(1));
        if (Delta_e<=0)
           m0=m1;
           Em0=Em1;
           plot3(m0(1),m0(2),Em0,'b*');
        else
            r=rand;
            if (P > r)
                m0=m1;
                Em0=Em1;
                plot3(m0(1),m0(2),Em0,'b*');
            end
        end
    end
    for j=1:N*M 
        T(j)=T(j)*exp(-c(j)*k^(1/(M*N)));
    end
end
plot3(m0(1),m0(2),Em0,'r*');
plot3(m0(1),m0(2),Em0,'mo');
m0