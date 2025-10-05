% Taller 6 adaptado: Lee valores iniciales desde un JSON y ejecuta LM por cada caso
% Autor: Juan Sebastian Guerrero

clear
clc
close all

% Cargar datos
load vgdata
disp(size(h))
disp(size(theta))

% Parámetros del modelo
theta_s = 0.44;
theta_r = 0.09;
sigma = 0.02;

% Ecuación no lineal simbólica
syms alpha n h1 data;
theta_h_ = theta_r + (theta_s - theta_r)/(1 + (-alpha*h1)^n)^(1 - 1/n);

% Función F y Jacobiano
F = (theta_h_ - data) / sigma;
J = jacobian(F, [alpha, n]);

% Configuración del método LM
lambda_init = 1000;
factor = 10;
tol = 1e-4;
max_iters = 100;

% Leer archivo JSON con los casos
json_text = fileread('resultados.json');
data_json = jsondecode(json_text);

% Prealocar arrays para almacenar resultados
num_casos = length(data_json);
resultados = struct();
tiempos = zeros(num_casos, 1);

% Iterar sobre cada conjunto de parámetros iniciales
for idx = 1:num_casos
    fprintf('\n=== CASO %d ===\n', idx);
    init_temps = data_json(idx).init_temps;
    coeficients = data_json(idx).coeficients;
    alpha0 = data_json(idx).avg_optimal(1);
    n0 = data_json(idx).avg_optimal(2);

    fprintf('Valores iniciales -> alpha = %.6f, n = %.6f\n', alpha0, n0);

    % Inicializar variables
    lambda = lambda_init;
    current_cost = norm(double(subs(F, {alpha, n, h1, data}, {alpha0, n0, h, theta})))^2;
    fprintf('Costo inicial = %.6f\n', current_cost);

    k = 1;
    converged = false;

    % Medir tiempo de ejecución
    tic;
    
    % Iteraciones LM
    while ~converged && k < max_iters
        J_eval = double(subs(J, {alpha, n, h1}, {alpha0, n0, h}));
        F_eval = double(subs(F, {alpha, n, h1, data}, {alpha0, n0, h, theta}));

        JTJ = J_eval' * J_eval + lambda * eye(2);
        JTFT = -J_eval' * F_eval;

        delta_m = JTJ \ JTFT;

        alpha_trial = alpha0 + delta_m(1);
        n_trial = n0 + delta_m(2);

        F_trial = double(subs(F, {alpha, n, h1, data}, {alpha_trial, n_trial, h, theta}));
        new_cost = norm(F_trial)^2;

        if new_cost < current_cost
            alpha0 = alpha_trial;
            n0 = n_trial;
            current_cost = new_cost;
            lambda = lambda / factor;
            converged = (norm(delta_m) < tol);
            fprintf('Iter %d: α=%.6f, n=%.6f, cost=%.6f (ACCEPTED)\n', k, alpha0, n0, current_cost);
        else
            lambda = lambda * factor;
            fprintf('Iter %d: α=%.6f, n=%.6f, cost=%.6f (REJECTED)\n', k, alpha_trial, n_trial, new_cost);
        end

        k = k + 1;
    end
    
    % Guardar tiempo de ejecución
    tiempos(idx) = toc;

    % Intervalos de confianza
    J_final = double(subs(J, {alpha, n, h1}, {alpha0, n0, h}));
    cov_matrix = inv(J_final' * J_final);
    IC = 1.96 * sqrt(diag(cov_matrix));

    % Almacenar resultados
    resultados(idx).caso = idx;
    resultados(idx).init_temps=init_temps;
    resultados(idx).coeficients=coeficients;
    resultados(idx).alpha_final = alpha0;
    resultados(idx).n_final = n0;
    resultados(idx).IC_alpha = IC(1);
    resultados(idx).IC_n = IC(2);
    resultados(idx).costo_final = current_cost;
    resultados(idx).iteraciones = k-1;
    resultados(idx).convergio = converged;
    resultados(idx).tiempo_seg = tiempos(idx);

    % Resultados finales
    if converged
        fprintf('✅ Convergió tras %d iteraciones\n', k-1);
    else
        fprintf('⚠️ No convergió (máximo alcanzado: %d iteraciones)\n', k-1);
    end

    fprintf('Parámetros finales:\n');
    fprintf('  α = %.6f ± %.8f\n', alpha0, IC(1));
    fprintf('  n = %.6f ± %.5f\n', n0, IC(2));
    fprintf('Costo final = %.6f\n', current_cost);
    fprintf('Tiempo de ejecución: %.4f segundos\n', tiempos(idx));
end
%%
% Mostrar tabla resumen al final
%fprintf('\n\n' + repmat('=', 1, 100) + '\n');
fprintf('TABLA RESUMEN DE RESULTADOS\n');
%fprintf(repmat('=', 1, 100) + '\n');
fprintf('Caso |init_temps|coeficients|   Alpha Final  |    n Final    |   Costo Final   | Iters | Tiempo (s)\n');
%fprintf(repmat('-', 1, 100) + '\n');

for idx = 1:num_casos
    %conv_char = '✓' if resultados(idx).convergio else '✗';
    fprintf('%4d | %2d | %2d | %2d | %2d | %14.9f | %13.9f | %14.10f | %5d |   %s   | %10.4f\n', ...
            resultados(idx).caso, ...
            resultados(idx).init_temps, ...
            resultados(idx).coeficients, ...
            resultados(idx).alpha_final, ...
            resultados(idx).n_final, ...
            resultados(idx).costo_final, ...
            resultados(idx).iteraciones, ...
            resultados(idx).tiempo_seg);
fprintf('\n');
end
fprintf('\n');
%fprintf(repmat('-', 1, 100) + '\n');

% Estadísticas adicionales
fprintf('\nESTADÍSTICAS GENERALES:\n');
fprintf('Tiempo total de ejecución: %.4f segundos\n', sum(tiempos));
fprintf('Tiempo promedio por caso: %.4f segundos\n', mean(tiempos));
fprintf('Tiempo mínimo: %.4f segundos (Caso %d)\n', min(tiempos), find(tiempos == min(tiempos), 1));
fprintf('Tiempo máximo: %.4f segundos (Caso %d)\n', max(tiempos), find(tiempos == max(tiempos), 1));
fprintf('Casos que convergieron: %d/%d\n', sum([resultados.convergio]), num_casos);