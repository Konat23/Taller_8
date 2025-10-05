% Metropolis simulated annealing
% Implemented by Sergio Abreo
% Computational Geophysics
% October 2018

clear
close all
clc

% Parámetro para elegir el problema
problem = "drop";  
% Opciones:
% Configuración de parámetros según el problema
switch problem
    case "ackley"
        intervalo = 32;
        paso = 1;
        view_angle = [-17, 18];
    case "boha1"
        intervalo = 100;
        paso = 1;
        view_angle = [-69, 67];
    case "drop"
        intervalo = 5;
        paso = 0.5;
        view_angle = [-161, 29];
    case "matya"
        intervalo = 10;
        paso = 1;
        view_angle = [-220, 32];
    otherwise
        error('Problema no reconocido. Opciones: "ackley", "boha1", "drop", "matya"');
end

[X,Y] = meshgrid(-intervalo:paso:intervalo, -intervalo:paso:intervalo);
[k, k] = size(X);

% Función objetivo según el problema seleccionado
switch problem
    case "ackley"
        objective_function = @ackley;
    case "boha1"
        objective_function = @boha1;
    case "drop"
        objective_function = @drop;
    case "matya"
        objective_function = @matya;
end

% Calcular superficie
for i = 1:k
    for j = 1:k
        Z(i,j) = objective_function([X(i,j), Y(i,j)]);
    end
end

figure(1)
surf(X,Y,Z)
alpha 0.05
grid on
hold on
view(view_angle)

% Algoritmo de simulated annealing
T = 10;
m0 = (rand(1,2)*2 - 1)*intervalo;
Em0 = objective_function(m0);

% Punto inicial
plot3(m0(1), m0(2), Em0, 'b+');

while T > 0 % loop over the temperature T
    for i = 1:100 % loop over a number of random moves/temperature
        m1 = (rand(1,2)*2 - 1)*intervalo;
        Em1 = objective_function(m1);
        Delta_e = Em1 - Em0;
        P = exp(-Delta_e/T);
        
        if (Delta_e <= 0)
            m0 = m1;
            Em0 = Em1;
            plot3(m0(1), m0(2), Em0, 'b*');
        else
            r = rand;
            if (P > r)
                m0 = m1;
                Em0 = Em1;
                plot3(m0(1), m0(2), Em0, 'b*');
            end
        end
    end
    
    if(T >= 2)
        T = T - 1;
    else
        T = T - 0.2;
    end
end

% Punto final
plot3(m0(1), m0(2), Em0, 'r*');
plot3(m0(1), m0(2), Em0, 'mo');

fprintf('Solución encontrada:\n');
fprintf('Coordenadas: (%f, %f)\n', m0(1), m0(2));
fprintf('Valor de la función: %f\n', Em0);

% Save pdf plot with function name
exportgraphics(gcf, sprintf('%s_function.pdf', problem));