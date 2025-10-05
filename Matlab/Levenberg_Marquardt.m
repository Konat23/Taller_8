% Taller 6. Capitulo 9, segunda edicion de Aster. Ejercicio 3.
% Plantilla elaborado por: Sergio A. Abreo Carrillo
% Curso de Geofisica computacional
% Maestria en Geofisica de la Escuela de Fisica , UIS
% Estudiante Juan Sebastian Guerrero
% Limpia las variables del espacio de trabajo
clear
clc
close all

% Carga de la matriz G y el vector de observaciones dn en el espacio de
% trabajo
load vgdata
disp(size(h))
disp(size(theta))

% Valores del enunciado
theta_s=0.44;
theta_r=0.09;
sigma=0.02;

% Ecuacion no lineal
syms alpha n h1 data;
theta_h_= theta_r + (theta_s - theta_r)/(1 + (-alpha*h1)^n)^(1-1/n);



% Creacion de F
F = (theta_h_-data)/sigma;
disp("F Shape: "+ mat2str(size(F)))

% Creacion de la matriz jacobiana nx2
J = jacobian(F, [alpha, n]);
disp("J Shape: "+ mat2str(size(J)))

%%
lambda = 1000;  % Start with non-zero lambda
factor = 10;    % Factor to increase/decrease lambda
tol = 0.0001;
max_iters = 100;

% Condiciones de arranque
alpha0 = 0.013581; 
n0 = 1.815793;    

% Calculate initial cost
current_cost = norm(double(subs(F, {alpha, n, h1, data}, {alpha0, n0, h, theta})))^2;
fprintf('Initial: alpha = %.6f, n = %.6f, cost = %.6f\n', alpha0, n0, current_cost)

k = 1;
converged = false;

% Vectores para guardar historial
lambda_hist = [];
cost_hist = [];
accepted_hist = [];
alpha_hist = [alpha0];
n_hist = [n0];
iter_hist = [0];

while ~converged && k < max_iters
    % Evaluate Jacobian and function
    J_eval = double(subs(J, {alpha, n, h1}, {alpha0, n0, h}));
    F_eval = double(subs(F, {alpha, n, h1, data}, {alpha0, n0, h, theta}));
    
    %Levenberg Marquardt
    JTJ = J_eval'*J_eval + lambda*eye(2);
    JTFT = -J_eval'*F_eval;
    
    % Solve for parameter update
    delta_m = JTJ\JTFT;
    
    % Trial update
    alpha_trial = alpha0 + delta_m(1);
    n_trial = n0 + delta_m(2);
    
    % Calculate new cost
    F_trial = double(subs(F, {alpha, n, h1, data}, {alpha_trial, n_trial, h, theta}));
    new_cost = norm(F_trial)^2;
    
    % Check if update improves solution
    if new_cost < current_cost
        % Accept update
        alpha0 = alpha_trial;
        n0 = n_trial;
        current_cost = new_cost;
        
        converged = (norm(delta_m) < tol);
        
        fprintf('Iteration %d: alpha = %.6f, n = %.6f, cost = %.6f, lambda = %.3e (ACCEPTED)\n', ...
                k, alpha0, n0, current_cost, lambda);
        lambda = lambda / factor;  % Decrease lambda for faster convergence
        
        % Guardar historial
        lambda_hist(end+1) = lambda;
        cost_hist(end+1) = current_cost;
        accepted_hist(end+1) = 1;  % accepted
    else
        % Reject update, increase lambda
        fprintf('Iteration %d: alpha = %.6f, n = %.6f, cost = %.6f, lambda = %.3e (REJECTED)\n', ...
                k, alpha_trial, n_trial, new_cost, lambda);
        lambda = lambda * factor;        
        % Guardar historial
        lambda_hist(end+1) = lambda;
        cost_hist(end+1) = new_cost;
        accepted_hist(end+1) = 0;  % rejected
    end
    
    k = k + 1;
    alpha_hist(end+1) = alpha0;
    n_hist(end+1) = n0;
    iter_hist(end+1) = k-1;
end

if converged
    fprintf('\nConverged after %d iterations!\n', k-1);
else
    fprintf('\nMaximum iterations reached without convergence\n');
end

%% IC
J_final = double(subs(J, {alpha, n, h1}, {alpha0, n0, h}));
cov = inv(J_final'*J_final);
IC = 1.96*sqrt(diag(cov));
fprintf('Final parameters: alpha = %.6f +- %.8f, n = %.6f\n +- %.5f\n', alpha0,IC(1), n0,IC(2));
%% Plot final result
figure;
hh = linspace(min(h),max(h));
theta_h_final = double(subs(theta_h_, {h1,alpha,n}, {hh,alpha0,n0}));
plot(hh, theta_h_final, 'r-', 'LineWidth', 2);
hold on
% Plot the observed data for comparison
plot(h, theta, 'bo', 'MarkerSize', 5);
legend('Model', 'Observed Data');
xlabel('h');
ylabel('Theta_h');
title('Final Result of Theta_h vs h');
hold off
grid on;

%% Plots de lambda y costo por iteracion
figure;
plot(1:length(lambda_hist), lambda_hist, 'b-o','LineWidth',1.5);
xlabel('Iteration');
ylabel('\lambda');
yscale("log")
title('Lambda evolution per iteration');
grid on;

figure;
hold on;
% Graficar puntos aceptados y rechazados en grupos
accepted_idx = find(accepted_hist == 1);
rejected_idx = find(accepted_hist == 0);

h1 = plot(accepted_idx, cost_hist(accepted_idx), 'go', 'MarkerFaceColor','g'); % verde aceptado
h2 = plot(rejected_idx, cost_hist(rejected_idx), 'ro', 'MarkerFaceColor','r'); % rojo rechazado

% Línea del costo
h3 = plot(cost_hist, 'b-');

xlabel('Iteration');
ylabel('Cost');
yscale("log")
title('Cost per iteration');
grid on;

% Leyenda con los handles creados
legend([h1 h2 h3], {'Accepted','Rejected','Cost'}, 'Location','best');

hold off;

figure;
subplot(2,1,1);
plot(iter_hist, alpha_hist, 'b-o', 'LineWidth', 1.5, 'MarkerFaceColor', 'b');
xlabel('Iteración');
ylabel('\alpha');
title('Evolución del parámetro \alpha');
grid on;

subplot(2,1,2);
plot(iter_hist, n_hist, 'r-o', 'LineWidth', 1.5, 'MarkerFaceColor', 'r');
xlabel('Iteración');
ylabel('n');
title('Evolución del parámetro n');
grid on;
